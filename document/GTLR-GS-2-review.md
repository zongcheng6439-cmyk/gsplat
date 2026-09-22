# GTLR-GS 复现 · 二：源码审查发现的问题（R1–R15）

> 系列文档：① [第一版实现](GTLR-GS-1-implementation.md) → ② 审查发现的问题（本文） → ③ [后续修复记录](GTLR-GS-3-fixlog-2026-09-22.md)

**审查结论（2026-09-22）：第一版是已有三模块的实验性实现，尚不能认定为完成、可信的论文复现。** 已发现初始化坐标系、子采样近邻索引、深度梯度等确定问题。

审查日期：2026-09-22。审查代码：`/Users/zongcheng/code/3dgs/gsplat`，分支 `gtlr`，提交 `d67a9225ccd8b039fcfd088474156def8f575f86`。范围：`examples/gtlr/`、`gsplat/strategy/gtlr.py`、相关 Parser、densification、CUDA 像素坐标约定以及 `tests/test_gtlr.py`。本次仅审查与记录，没有修改训练实现或启动服务器训练。

论文依据：[GTLR-GS arXiv v1 正文](https://arxiv.org/html/2603.23192v1)，重点核对 III-B/C/D、式 (1)–(11) 和 IV-B。以下文件行号均针对审查提交，后续修改后可能移动。P1 表示应在继续完整实验前修复；P2 表示影响复现一致性、可扩展性或结论可信度。条件性问题均写明触发条件，未把它们表述为所有数据都会失败。

## R1 · P1 · 外部初始化点云没有进入相机的归一化坐标系

**位置：** `examples/gtlr/simple_trainer_gtlr.py:168–175,201–225`；对照 `examples/datasets/colmap.py:276–318` 和 `examples/gtlr/project_depth.py:77–81`。

默认 `normalize_world_space=True`，Parser 已变换相机和内部 SfM 点，深度生成脚本也会在 `--normalize` 下变换外部 LiDAR 点。但训练器 `_init_points()` 读取 `init_ply` 后直接返回原始 xyz，没有应用 `self.parser.transform`。采样脚本只选择原点，不做坐标变换，因此文档默认流程会把原始 LiDAR 高斯放到归一化相机中渲染；`_estimate_lidar_normals()` 还会沿用这套错误参考坐标。

**影响：** 当归一化变换非单位阵时，几何位置、投影、尺度及法线参考不一致；PSNR 或深度 warmup 无法证明坐标正确。

**建议：** 对外部原始 PLY 统一应用 `transform_points(self.parser.transform, xyz)`，在变换后的点集上求初始尺度及法线。增加输入坐标约定和元数据，避免预归一化 PLY 被二次变换；SfM fallback 已经归一化，不能再变换一次。

**验收：** 用非平凡旋转、平移、缩放的合成相机和点，验证原始/归一化两条路径投影相同，深度仅乘相似变换的尺度；检查初始化点、法线参考点和投影深度三者一致。

## R2 · P1 · 两处子采样 kNN 的局部索引被当成全云索引

**位置 A：** `examples/gtlr/simple_trainer_gtlr.py:219–225`。

点数 >200,000 时，`ref = points[randperm(...)]`，`knn_indices(points, ref, 4)` 返回的是 **ref 的行号**，下一行却使用 `points[idx]` 计算初始尺度。邻居因此变成全云前 20 万行中的其他点，尺度可能接近场景尺寸，而不是局部点间距。应使用 `ref[idx]`，或保存 `ref_idx` 后映射回全云。

**位置 B：** `examples/gtlr/sample_points.py:133–174`。

未安装 Open3D 且点数 >200,000 时，fallback 同样返回 ref 局部行号；`curvature_texture()` 随后使用 `points[idx]` 和 `colors[idx]`。这同时破坏几何与纹理评分，连带影响采样分布。Open3D 分支返回原云索引，不触发此错误；训练尺度的 A 路径不受是否安装 Open3D 影响。

**建议：** 明确近邻 API 的索引空间。采样 backend 应保存并返回 `ref_idx[local_idx]`；初始化尺度直接索引 ref。修复后重做 fallback 生成的采样结果和依赖它的 checkpoint，旧结果不能继续作为有效复现证据。

**验收：** 强制使用乱序子集作为 ref，与独立精确近邻结果逐项对照；覆盖超过 cap 的路径。现有几百点测试不会进入该分支。

## R3 · P1 · 深度误差硬截断使最需要校正的点失去梯度

**位置：** `examples/gtlr/geom.py:193–226`。

当前计算 `abs(D_lidar - depth).clamp(max=2.0)`。当误差 >2 时，loss 保持常数，对预测深度的导数为零。这能压低日志里的 loss，但深度监督本身已无法纠正这些大误差。实测最小例：预测 5、目标 1、权重 1、通过门控，loss=2，`depth.grad=0`。

**建议：** 优先修复 R1/R2；确需鲁棒化时使用大残差仍有梯度的 Huber/Charbonnier，或明确的离群点策略，同时把相应偏离列为工程改动。记录原始残差、截断比例、有效监督覆盖率；以恢复正确几何而不是限制 loss 幅值作为验收目标。

## R4 · P1 · 法线缓存只根据点数变化刷新，可能与高斯身份错位

**位置：** `examples/gtlr/simple_trainer_gtlr.py:423–440`；`gsplat/strategy/ops.py:193–234,252–265`。

一次 refine 会先 duplicate/split 再 prune，即使最终总数等于 `n_before`，高斯身份和顺序也可能改变。外置的 `self.gaussian_normals` 不在 strategy state 中，不会随这些操作重排；点数未变就跳过刷新，随后会把旧法线施加到错误的高斯。另一个问题是，高斯只发生位移但点数不变时，最近邻关联也一直不更新；15k 停止 refine 后尤为明显。

**建议：** 按拓扑变更事件刷新或同步维护 reference ID；额外按周期/位移阈值更新最近邻关联。不能只用长度判断身份一致。

**验收：** 构造“新增数量=删除数量”的 refine；对不同位置分配不同参考法线，验证每行关联正确，再测试纯位移跨越两个法线区域的情况。

## R5 · P2 · 深度门控混入透明度，低覆盖区域会失去监督

**位置：** `examples/gtlr/geom.py:106–107,217–226`；`examples/gtlr/simple_trainer_gtlr.py:386–395`。

`ray_dot` 是累积法线 `Σwᵢnᵢ` 与光线的点积，未去除 opacity/coverage。以它的绝对值 >0.1 作为“非掠射”判断，实际也在按透明度和法线相互抵消程度筛选。例如相同正对相机的平面、相同预测深度 2，累积 alpha=0.05 时被全部丢弃，alpha=0.5 时才有深度损失。opacity reset 后也可能暂时减少监督。

此外，函数用过滤后的集合大小重新求 mean，已经不是仅由有效 LiDAR 像素定义的监督集合；同一帧损失权重会随模型当前状态改变。

**建议：** 将覆盖率、平面法线一致性、掠射角判断分开；角度用归一化法线与归一化光线计算，保留独立的 coverage 安全阈值。明确空渲染区域的处理与损失分母，监控各类门控删除的像素数量。不要直接除一个很小的 alpha 来替代稳定性判断。

## R6 · P1 · 零信息/概率支撑不足的点云会令采样失败

**位置：** `examples/gtlr/sample_points.py:178–196`。

如果曲率、纹理均为常数，min-max 返回零，最后 `probs / probs.sum()` 产生 NaN，`numpy.choice` 抛异常。即使总和非零，当正概率点数少于 M 时，无放回采样仍抛 `Fewer non-zero entries in p than size`。例如概率来自 `[0,0,1]`，请求采满三点就失败。`m=min(m,n)` 不能处理这个问题。

**建议：** 无信息时明确退化为均匀采样；概率支撑不足时用均匀项混合或先采有效部分再补齐，并记录此策略属于退化处理。验证 `0≤M≤N`、有限输入、足够邻居和颜色存在性；同色平面、重复点、M=N 都应有测试。

## R7 · P2 · 置信度的归一化范围与有效 LiDAR 区域不一致

**位置：** `examples/gtlr/geom.py:175–190`；`examples/gtlr/simple_trainer_gtlr.py:391–394`。

`laplacian_confidence()` 不接收深度 mask，直接用整幅图的 `lap.max()`；式 (9) 的最大值范围是有效 LiDAR 像素。无深度区域的强纹理会抬高归一化分母，使实际受监督边界获得过高置信度。数值例中，无深度强边缘使唯一有效深度像素的权重从应有的 0 变成约 0.9。

**建议：** 传入固定 LiDAR 有效 mask，只在该集合计算最大值；另处理空集合、零 Laplacian、非有限深度。这里的 mask 应独立于当前预测，不能直接改为 R5 的动态门控集合。

## R8 · P2 · 随机参考集下无条件删除最近邻，并非排除自身

**位置：** `gsplat/strategy/gtlr.py:86–100,142–146`；`examples/gtlr/geom.py:243–249`；采样 fallback 及尺度初始化同样使用 `[:,1:]`。

只有 query 确实存在于 ref 且对应行是自身时，删第一个近邻才合理。随机子集中的大多数查询点并不存在于 ref，此时删除的是合法最近邻，把 kNN 变成第 2 到第 k+1 近邻，改变邻域半径及曲率/法线统计。重复坐标还会使“距离为零即自身”的判断不可靠。

**建议：** 保留原始点 ID，按身份排除自身；外部查询直接保留前 k 个。参考集还应有最小大小检查。测试既包含自身又不包含自身的 query、重复坐标和少于 k+1 的参考集。

## R9 · P2 · 归一化后的深度单位和损失权重缺少度量尺度说明

**位置：** `examples/gtlr/project_depth.py:79–91`；`examples/gtlr/simple_trainer_gtlr.py:87,113,442–445`；`examples/gtlr/validate.py:125–133,211`。

归一化本身不等于丢失度量信息，但当前只保存裸 `.npy`，checkpoint 也只有 `step/splats`，没有保存相似变换和单位。若 `x_norm=sR x_metric+t`，则 `D_norm=s D_metric`。直接在归一化单位下用 `depth_lambda=1`，相对于 RGB 的约束强度随 s 改变；`max_err=2`、验证 `err<0.05` 也不是固定米制阈值。

**建议：** 保存 transform、s、原始单位、相机及数据版本；需要米制损失时用 `L_norm/s`，或说明采用的是场景归一化损失。验证和导出显式恢复米制。缺少注册/单位校验时，不应宣称结果已具备绝对尺度精度。

## R10 · P2 · 默认训练实际不会执行到 θ=0.3 的分裂阶段

**位置：** `gsplat/strategy/default.py:106–109,183–191`；`gsplat/strategy/gtlr.py:126–130,147–149`；`examples/gtlr/simple_trainer_gtlr.py:95,147–156,487–490`。

阈值函数以总训练步数 30k 为分母，但默认 densification 在 15k 停止，最后一次常规 refine 为 14900，θ≈0.19933。文档“0.1 升到 0.3”描述的是函数范围，不是实际分裂经历的范围。更改 `--max-steps` 也不会自动令 `strategy.total_iters` 等于新的训练长度。

**建议：** 明确 schedule 与 refine 窗口的关系，实际日志输出阈值和候选/通过数量；将 total_iters 与配置同步。若选择在 refine 结束前升到 0.3，应作为调度变体报告，不能为了“达到终值”就说这是论文唯一正确实现。

另有两项应明确标记为复现假设：当前是在原梯度/尺寸候选上附加曲率门控，duplicate 完全保留；在线 k=16 与随机参考集 cap 都需记录和消融。论文公开描述不足以唯一确定所有这些实现细节，不能把所有取舍都当作已验证的等价实现。初始预算 M 与训练中点数也应分别记录；不应仅凭论文的预算措辞断言必须实现全训练硬上限。

## R11 · P2 · 法线参考经过二次稀疏化，缺少可靠性判断

**位置：** `examples/gtlr/simple_trainer_gtlr.py:271–283`；`examples/gtlr/geom.py:235–263`。

法线不是来自独立的完整 LiDAR 参考云，而是在已按几何/纹理采样的初始化云中再随机取最多 10 万点估计；示例 `curvature_ref_max=50000` 还会进一步稀疏邻域。细杆、墙角、相近两层面容易被混合。最近邻关联没有距离、邻域半径或平面可靠性限制，离表面很远的高斯也会获得同等权重的法线约束。

**建议：** 将完整 LiDAR 法线预计算与初始化采样分开；保留对应源点 ID，采用空间索引/体素覆盖降低查询成本，记录邻域半径、特征值质量和关联距离。法线置信筛选属于工程增强，需与基础复现分别评估。先修复 R1/R4，再评估该近似带来的误差。

## R12 · P2 · 深度、RGB 与栅格化像素约定未贯通

**位置：** `examples/gtlr/geom.py:79–95`；`examples/gtlr/project_depth.py:49–51`；`gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu:108–117`；`gsplat/cuda/include/Utils.cuh:606`。

几何光线使用整数 `(x,y)`，而 CUDA 栅格化在 `(x+0.5,y+0.5)` 采样；点投影又使用 `round()` 写入像素。传入相同 K 时，这些约定存在半像素差异，会在倾斜平面、窄结构及低分辨率下引入误差。当前解析测试也调用同一个 `pixel_rays()` 构造“真值”，不能发现与 CUDA 的偏差。

**建议：** 统一相机内参、像素中心、LiDAR 写入位置与 ray 的定义，不要只在某一处随意补 0.5。用独立构造的倾斜平面，经真实 CUDA 渲染验证解析深度，并检查前向和反向。

**另一处数据不一致：** `validate.py:113–118` 直接读取 `parser.image_paths` 原图，而训练 `Dataset` 会去畸变及裁剪（`examples/datasets/colmap.py:467–483`）。有畸变或 ROI 裁剪时，验证权重/叠加图可能错位甚至尺寸不匹配，应共用 Dataset 的图像预处理。训练本身也没有应用 Dataset 返回的有效 `mask`，鱼眼无效边界可能进入 RGB/深度损失。

## R13 · P1（存在同名多相机图片时）· 深度文件名冲突且缺少输入校验

**位置：** `examples/gtlr/project_depth.py:90–91`；`simple_trainer_gtlr.py:260–268`；`validate.py:76–81`。

保存和读取均只使用 `basename` 去扩展名，例如 `cam0/0001.jpg` 和 `cam1/0001.jpg` 都映射成 `0001.npy`，后者覆盖前者，训练可能静默使用另一台相机的深度。该问题不要求归一化错误就能发生，但仅在命名冲突时触发。

**建议：** 使用相对路径或稳定 image ID，生成时拒绝重复 key。保存 manifest 并核对 K、外参、尺寸、factor、normalize/transform、源点云版本、有限深度；当前缺文件或有效点不足时只是少加载，错误 depth_dir 甚至可能让“full”实验没有任何深度监督。基础复现模式应对此明确报错，消融模式才允许主动关闭。

## R14 · P2 · 分块距离矩阵并没有让整个处理流程具备大点云可扩展性

**位置：** `gsplat/strategy/gtlr.py:64–83`；`examples/gtlr/sample_points.py:140–174,208–211`；`simple_trainer_gtlr.py:222–224,271–283`；`geom.py:260–263`。

`knn_indices()` 仍是 O(Q×R) 全距离枚举；chunk=4096、R=200k 时，仅一个 float32 距离矩阵就约 **3.28 GB（3.05 GiB）**，不是总峰值。R=50k 时也约 0.82 GB。表达式中间结果、topk、输出索引和训练状态另占内存。

离线采样还会一次创建全体 `N×64×3` 邻居张量；N=3M 时单个 float32 张量约 2.30 GB，`centered`、颜色邻域、索引等继续增加峰值。Open3D 分支是 Python 逐点 KDTree 查询；无 Open3D 的离线 fallback 输入来自 CPU，也没有搬到 CUDA，不能称为已实现全流程 CUDA 分块采样。每次拓扑变化后，法线关联又在 CPU 上做 N×100k 枚举。

**建议：** 使用精确空间索引或有误差评估的近邻实现，曲率/纹理按 query chunk 求完即写回标量，不保留全云邻域张量。分别测初始化、离线采样、每次 refine/法线刷新时间和 CPU/GPU 峰值；不能用 1000 步耗时外推 30k 完整流程。

## R15 · P2 · 当前证据不足以证明完整复现效果

**位置：** `tests/test_gtlr.py`；`examples/gtlr/validate.py:72,119–162,165–193`；`simple_trainer_gtlr.py:442–484`。

1. 现有测试共 7 个 `test_*` 函数，主要覆盖小规模公式和 gate；缺少外部 PLY 坐标变换、随机 ref 索引、退化采样、深度梯度、拓扑身份更新以及真实 CUDA 深度的集成验证。
2. 历史 1000 步 smoke 发生在 `depth_start_iter=3000` 之前，按默认配置根本不会训练深度模块，因此 PSNR/SSIM 不能证明深度正则已跑通或有效。应另做跨越 warmup、refine、opacity reset 的集成冒烟。
3. 深度验证从所有图片等距抽样，混合训练/验证视图，且复用模型相关门控；无有效像素视图被跳过。虽然会输出保留比例，但最终误差仍是条件子集结果，覆盖率低时可能显得过好。应固定 held-out 集，分别报告 LiDAR 覆盖、可渲染覆盖、失败视图数和原始未截断米制误差。
4. `validate_sampling()` 复用采样实现的 `curvature_texture()`，会继承 R2；退化数据的 min-max 分母也未保护。它衡量的是代理评分富集，不能替代薄结构保留率、空间覆盖或最终渲染质量的独立评价。
5. trainer 只打印 PSNR/SSIM，缺少 LPIPS、度量深度/表面精度、逐视图及聚合 JSON、点数曲线、分项损失和峰值资源。需要固定同一数据划分和初始预算，做 baseline、sampling-only、split-only、depth-only、full 以及 normal/warmup/鲁棒化消融。论文未公开的设置应单列为假设。
6. checkpoint 缺 optimizer、scheduler、strategy state、随机数状态、配置和归一化元数据，且没有 resume 入口。无法精确续训/回溯；离线验证还硬编码 `normalize=True`。采样 CLI 的 `--seed` 只传给 NumPy 抽样，torch 随机参考集没有同步设 seed，重复执行未必产生同样的初始化。

## 本次实际验证及其边界

本机执行了源码检查和纯 CPU 数值反例，所用已有环境为 torch 2.10.0、`torch.cuda.is_available()=False`；没有运行 CUDA、加载真实场景或重做 30k 实验。

尝试使用已有环境执行原测试：

```bash
cd /Users/zongcheng/code/3dgs/gsplat
/Users/zongcheng/code/project/gaussian-scene-reconstruction/.venv/bin/python tests/test_gtlr.py
```

结果：在导入 `gsplat.cuda._backend` 时因缺少 `rich` 中止，**不能报告原测试通过**。系统 `python3` 另外缺少 torch。本次未安装依赖。

为核实核心数学行为，使用已有 torch 环境从源码 AST 提取 `knn_indices`、采样概率、`depth_loss`、`laplacian_confidence` 等纯函数原函数体，绕开包级导入，运行了小型反例。此方式仅验证函数行为，不等于包级或 CUDA 集成测试。关键输出如下：

| 检查 | 实际输出/结论 |
|---|---|
| ref 局部索引直接用于全云 | 同一查询对应的错误邻居 x 为 `[1,2,100]`，正确 ref 邻居 x 为 `[101,102,0]`；证明索引空间不可混用 |
| 平坦同色评分，全零概率 | `ValueError: Probabilities contain NaN` |
| 正概率点数小于 M | `ValueError: Fewer non-zero entries in p than size` |
| 预测深度 5、目标 1、截断上限 2 | loss=2，预测深度梯度=0 |
| 同一平面预测深度 2，改变累积 alpha | alpha=0.05：无监督，loss=0；alpha=0.5：有监督，loss=1 |
| 无深度区存在更强 Laplacian | 唯一 LiDAR 像素实际 confidence≈0.9，按有效域归一化应为 0 |
| query 不在 ref | 正确前三邻居 ID `[0,1,2]` 被无条件删首项改为 `[1,2,3]` |
| 默认最后一次 refine 的 θ | θ(14900/30000)=0.1993333333 |

R1/R4/R9/R12/R13 等是源码路径和触发条件分析，尚未做真实数据上的运行验证。历史服务器工件是否使用与本提交完全相同的代码、预变换数据或不同参数，本次没有证据；不能据此推断历史图像必然无效，也不能用历史数字否定当前源码中的确定错误。

## 建议的修复与验收顺序

1. **先保证输入和初始化正确：** 修复 R1、R2、R6、R8、R13；保存统一数据 manifest，检查投影、尺度分位数与初始化空间覆盖。重生成受错误采样影响的工件。→ 已完成，见文档③
2. **再保证监督能够推动正确几何：** 修复 R3、R4、R7，拆解 R5 的角度/覆盖门控，统一 R9/R12 的度量单位与像素约定。分别检查预期可微路径上的 means/quats/scales 梯度是否有限，并验证大误差确实能下降。
3. **明确所有复现取舍：** 固定阈值调度、近邻定义、参考集、法线权重、warmup 和鲁棒损失；记录实际执行的 schedule、分裂/复制/剪枝数量，以及全训练点数。
4. **最后开展效果实验：** 先完成跨 3000 步及 opacity reset 的小场景集成验证，再在一致的数据划分/预算下运行完整 baseline 和消融；同时报告图像质量、几何误差、覆盖率和资源成本。此前将状态保留为“模块原型与局部验证”，不标为“完整复现成功”。
