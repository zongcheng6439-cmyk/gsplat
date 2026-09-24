# GTLR-GS 复现 · 三：后续修复记录（2026-09-22，第一阶段：输入与初始化）

> 系列文档：① [第一版实现](GTLR-GS-1-implementation.md) → ② [审查发现的问题](GTLR-GS-2-review.md) → ③ 后续修复记录（本文）
>
> 代码：`/Users/zongcheng/code/3dgs/gsplat`，分支 `gtlr`
> 依据：`document/GTLR-GS-2-review.md` 的审查（R1–R15），按其修复顺序的第一步执行
> 提交：`931d4a9`（修复）/ `d4c6058`（审查入库）
> 测试：`python tests/test_gtlr.py` **12/12 通过**（7 个原有 + 5 个新增验收测试）

## 全局决定：normalize_world_space 统一为 False

训练器、深度图生成、离线验证全部默认留在**原始度量坐标系**（LiDAR/COLMAP 米制帧），不做场景相似变换归一化。理由：

- 论文目标是"度量尺度重建"，归一化会让深度损失 λ_depth=1、max_err、验证阈值都变成随场景缩放的相对量（审查 R9）；不归一化则深度直接是米制。
- 外部 LiDAR 点云与相机天然同帧，从根上消除 R1 的坐标系不一致。

代价与对策：如果显式开启 `--normalize_world_space`（归一化模式），外部 init ply 会且只会被 `parser.transform` 变换一次（由 sidecar 元数据防二次变换）。

## 逐项修改

### R1（P1）外部初始化点云坐标系

- `examples/gtlr/sample_points.py`：采样输出 ply 时同时写 `<output>.json` sidecar，记录 `coordinate_frame: "raw"`、源点云、采样数、k、seed。
- `examples/gtlr/simple_trainer_gtlr.py::_init_points`：读取 sidecar 判断帧；`normalize=True + raw` → 应用 `transform_points(parser.transform, xyz)`；`normalize=False + 预归一化 ply` → 直接报错；SfM fallback 不再二次变换。
- 配置默认改为 `normalize_world_space: bool = False`。
- 验收测试：`test_projection_similarity_invariance`（非平凡旋转/平移/缩放 s=2.5 下，原始路径与归一化路径像素投影一致、深度仅乘 s）。

### R2（P1）子采样 kNN 的索引空间

- `gsplat/strategy/gtlr.py::knn_indices`：新增 `query_ids`/`ref_ids` 身份参数；返回恒为 ref 行号，调用方负责映射回全云。
- `examples/gtlr/sample_points.py::knn_indices_backend`：参考集局部索引经 `ref_ids[local]` 映射回原始云索引（修复前采样评分的 κ/τ 邻域取错点）。
- `simple_trainer_gtlr.py::_create_splats_with_optimizers`：初始尺度改为对 `ref[idx]`（而非 `points[idx]`）求距离。
- 验收测试：`test_knn_backend_full_index_space`（exact 路径与暴力全云 kNN 逐项一致；子采样路径索引地址全云且不含自身）。

### R6（P1）退化点云采样

- `sampling_probabilities`：κ、τ 均为常数（或出现非有限值）时退化为均匀分布，不再产生 NaN。
- `sample_indices`：正概率支撑 < M 时混入 1% 均匀地板（已在代码和本文档标注为**退化处理，非论文内容**）；M=N 可直接采满。
- 验收测试：`test_sampling_degenerate`。

### R8（P2）无条件删除最近邻

- `knn_indices` 在提供身份 ID 时按**精确 ID 匹配**排除自身（取 k+1 个后仅剔除同 ID 项），不再 `[:, 1:]` 盲删；query 不在 ref 中时保留真实最近邻；重合坐标点只有 ID 匹配才排除。
- 所有调用方同步更新：`GTLRStrategy.split_gate`、`knn_curvature`、`sample_points.knn_indices_backend`、`geom.estimate_normals`、训练器尺度初始化。
- 验收测试：`test_knn_self_exclusion`（自身在 ref / 不在 ref / 无 ID / 重复坐标四种情形）。

### R13（P1，条件触发）深度文件名冲突与输入校验

- `geom.depth_map_filename()`：深度图文件名编码完整相对路径（`cam0/0001.jpg` → `cam0__0001.npy`），扁平名字保持原样；生成端拒绝重名。
- `project_depth.py` 写出 `manifest.json`：factor、normalize、transform、源点云、图像→文件映射。
- 训练器 `_load_depth_maps`：校验 manifest 的 factor/normalize 与当前配置一致（不一致直接报错）；`depth_dir` 给了但零可用深度图时报错，不再静默退化为无深度监督。
- `validate.py` 同步使用同一命名函数，Parser normalize 改为显式 `--normalize` 开关（默认 False）。
- 验收测试：`test_depth_map_filename`。

### 顺带修正

- `geom.estimate_normals`：改用身份感知的 kNN 路径（原 `[:, 1:]` 同样受 R8 影响）。
- `tests/test_gtlr.py`：补充 `examples/` 到 sys.path；`test_curvature_plane_vs_edge` 改用带身份 ID 的标准用法。

## 因此作废、需重生成的服务器工件（/media/zc/SSD/GS/lutu/GTLR/）

| 工件 | 原因 | 重新生成命令 |
|---|---|---|
| `init_400k.ply` / `init_800k.ply` / `init_1.5m.ply` | 由索引空间有 bug 的采样 fallback 生成，κ/τ 评分失真 | `python -m gtlr.sample_points --input_ply .../extracted_all.ply --output_ply /GTLR/init_1.5m.ply --num_samples 1500000` |
| `depth_maps/`（864 张） | 旧版按 normalize=True 生成，与现在统一的 normalize=False 不匹配（manifest 校验会拦截） | `python -m gtlr.project_depth --data_dir /GTLR/data --ply .../extracted_all.ply --output_dir /GTLR/depth_maps --factor 4`（不加 --normalize） |
| `results/smoke/ckpt_999.pt` | 初始化尺度受 R2 影响、深度图坐标系不匹配 | 仅作历史参考，不可用于效果结论 |

注意：服务器上的代码还是旧版，下次训练前需先把 `gtlr` 分支同步过去（会触发一次 CUDA 重编译，约 18 分钟）。

## 仍未修复（按审查第十节顺序的后续阶段）

- **第二阶段（监督有效性）**：R3（max_err 硬截断使大误差零梯度 → 改 Huber/Charbonnier 或明确离群策略）、R4（法线缓存只按点数刷新、与高斯身份错位）、R5（ray_dot 门控混入透明度/覆盖度）、R7（Laplacian 置信度归一化范围应在有效 LiDAR 像素集上）。
- **第三阶段（复现取舍固化）**：R10（θ 调度与 refine 窗口不同步）、R11（法线参考二次稀疏化）。
- **第四阶段（效果实验）**：R12（像素中心半像素约定）、R14（kNN 可扩展性）、R15（测试/验证/消融/续训缺口）。

---

## 第二轮修改（2026-09-22 下午）

### 深度图投影可视化

- `geom.py` 新增共享的 `depth_to_rgb()`（蓝→红色表，0=无效像素），`validate.py` 删掉本地副本改用它。
- `project_depth.py` 新增 `--vis_every N`（默认 100，0 关闭）：每 N 张往 `<output_dir>/vis/<stem>.png` 存一张可视化——左：原图调暗+深度叠加，右：纯彩色深度图。用于训练前肉眼检查位姿/尺度是否对齐。
- 已在服务器用修复后代码重新生成 `/GTLR/depth_maps/`（864 张，manifest: factor=4, normalize=False），旧目录归档为 `depth_maps_obsolete_20260922/`。

### 点云采样支持上采样

- 原 `sample_indices` 里 `m = min(m, n)`：target 超过点数时退化为全取，不会上采样。
- 现行为（`sample_points.py`）：
  - `M <= N`：不放回采样（同论文 Eq. 4）。
  - `M > N`：保留全部 N 个点，再按 P_i 有放回抽取 `M - N` 个复制点（高分点按比例多复制），并用 `upsample_jitter()` 加高斯抖动——std = `jitter_frac`（默认 0.5）× 父点最近邻距离，避免复制点与父点重合导致初始尺度退化。
  - 新增 `--jitter_frac` 参数；sidecar json 记录 `num_upsampled` / `jitter_frac`。
- 新增 `--device`（默认 cpu，可指定 cuda 加速 kNN/协方差计算）；`knn_indices_backend` / `upsample_jitter` 内部索引统一跟随 points 设备。
- 验收测试：`test_upsampling_indices_and_jitter`（13/13 通过）。

### 服务器同步与预处理脚本

- `gtlr` 分支已 rsync 到 `/media/zc/SSD/GS/lutu/GTLR/gsplat/`（首次运行触发一次 CUDA JIT 编译，约 20 分钟，缓存已建好）。
- 预处理脚本上传至 `/media/zc/SSD/GS/lutu/GTLR/preprocess.sh`，容器内执行：
  ```bash
  docker exec gtlr bash /GTLR/preprocess.sh
  ```
  三步：采样（默认 NUM_SAMPLES=3000000 → 触发上采样）→ 深度图投影（含可视化）→ 采样质量校验。环境变量可覆盖各参数。

### 备注

- 服务器 GPU（RTX 2080 Ti）有其他用户的间歇性任务（6.8~9.5 GiB），采样这种一次性任务建议用 `--device cpu`（28 核足够），训练再等 GPU 空闲。

### 采样策略升级为两阶段混合采样（`--base_ratio`）

- 动机：纯按 Eq. 4 分数采样时，平坦区域（地面、墙面）κ/τ 接近 0，会被采得很稀，覆盖不足；分数只应该用于"细节加密"，不应该决定"是否覆盖"。
- 新函数 `hybrid_sample_indices(probs, m, base_ratio)`，返回三段索引：
  1. `base`：均匀不放回抽取 `base_ratio × M` 个点 —— 打底覆盖；
  2. `real`：按分数从剩余点不放回抽取 —— 高频区域加密；
  3. `dup`：真实点用光后还不够（M > N），按分数有放回复制 + `upsample_jitter` 抖动。
- `base_ratio=0` 完全退化为论文的纯分数采样（兼容旧行为）；`sample_indices` 保留为 r=0 的兼容包装。
- CLI：`--base_ratio`（默认 0）、`--jitter_frac`（默认 0.5）、`--device`（cpu/cuda）；sidecar json 记录 num_base/num_real/num_upsampled/base_ratio。
- `preprocess.sh` 增加 `BASE_RATIO`（默认 0.7）/ `JITTER_FRAC` / `SAMPLE_DEVICE`（默认 cpu，避免 GPU 被占时 OOM）。
- 验收测试：`test_hybrid_sampling`（14/14 通过）。

### 预处理不再触发 CUDA JIT 编译

- 问题：预处理脚本只需要纯 torch 的 `knn_indices`，但 `from gsplat.strategy.gtlr import ...` 会执行 `gsplat/__init__.py` → `cuda/_backend.py` → 没有编译好的 `gsplat/csrc` 时 JIT 全量编译 CUDA 扩展（"Setting up CUDA with MAX_JOBS"，约 20 分钟）。`geom.py` 顶部的 `from gsplat.rendering import rasterization` 同理。
- 修改：
  - `knn_indices` 抽到独立叶子模块 `gsplat/strategy/_knn.py`（纯 torch，无包内导入）；`strategy/gtlr.py` 改为 `from ._knn import knn_indices` 再导出（外部 API 不变）。
  - 新增 `examples/gtlr/knn.py`：按文件路径加载 `_knn.py`，不执行 `gsplat/__init__`；找不到文件才回退包导入。
  - `geom.py` 的 `rasterization` / `normalized_quat_to_rotmat` 改为函数内懒导入；`estimate_normals` / `associate_normals` 走 `knn.py` 加载器。
  - `validate.py` 的 `rasterization` 移入 `validate_depth` 函数内（`validate_sampling` 不再需要 CUDA）。
  - `preprocess.sh` 去掉 `PYTHONPATH=/GTLR/gsplat`（预处理不再需要；即使意外 import gsplat 也会用镜像里已编译的安装版）。
- 验证：容器内 `import gtlr.sample_points / project_depth / validate` 仅 1.9s，`sys.modules` 无 gsplat；kNN / estimate_normals 正常工作。
- 说明：`python setup.py install` 之后 `gsplat/csrc` 存在，`_backend.py` 直接加载编译产物，本来就不会再 JIT——此前的重编译是 `PYTHONPATH` 指向源码树遮蔽了镜像里的安装版导致的。训练脚本仍然需要 CUDA 扩展（首次 JIT 一次，或 setup.py install 永久生效）。

### 采样性能与 GPU 稳定性（2026-09-24）

问题：CPU 采样在共享服务器上极慢（6 分钟未完成 kNN 阶段）；GPU 两次 OOM（其他用户间歇性任务占 6.4~9.5 GiB，默认 4096×200k 分块需 3.3 GiB 瞬时显存）；修复 OOM 后又遇 cusolver 批量特征分解报错。

- `sample_points.py` 新增 `--ref_max`（kNN 参考集大小，默认 200k）和 `--chunk_size`（kNN 分块行数，默认 4096）参数，瞬时显存 = chunk_size × ref_max × 4 字节。
- GPU 实测（RTX 2080 Ti，cu128/sm_75）：`torch.linalg.eigvalsh` 的 cusolver 批量 syev 在 batch ≥ ~32768 时必现 `CUSOLVER_STATUS_INVALID_VALUE`（16384 正常），`curvature_texture` 的特征分解按 8192 分块。
- 推荐组合：`--device cuda --ref_max 50000 --chunk_size 1024`（瞬时约 200 MiB，可与他人任务共存），extracted.ply（83 万点）→ 300 万初始化点全程约 1~2 分钟；CPU 路径在 9 用户共享主机上不可用（>6 分钟未完成）。
- `preprocess.sh` 增加 `REF_MAX` / `CHUNK_SIZE` 环境变量。
- 服务器 `/GTLR/init_3m.ply` 已重新生成（834183 真实点 + 2165817 抖动复制点，meta 含全部参数）。

### 采样耗时定位与优化（2026-09-24 下午）

实测各阶段耗时（extracted.ply 83.4 万点，ref_max=50k, chunk=1024）：

| 阶段 | CPU | GPU |
|---|---|---|
| kNN（k=64，曲率/纹理用） | 76.0s | 3.8s |
| 协方差 + 特征分解 + τ | 0.9s | 0.2s |
| kNN（k=1，jitter 用） | 66.6s | 5.6s |

- 瓶颈完全是 kNN（两遍合计 >98%）；协方差/特征分解可忽略。
- 优化：`curvature_texture` 增加 `return_idx`，`upsample_jitter` 复用主 kNN 的第 0 列（最近邻）计算抖动幅度，**删掉第二遍 kNN**，CPU 路径时间减半，GPU 全程约 1 分钟。
- 另外：服务器上的 `simple_trainer_gtlr.py` docstring 示例被改为 `/data/` 路径，已拉回本地合并。约定：以后每次修改前先 rsync 拉回服务器对应文件 diff，再改。

### 采样默认回归论文模式（M > N 报错退出）

- `sample_points.py` 新增 `--allow_upsample`（默认关闭）：`--num_samples > N` 时直接报错退出，并提示三种选择（降低 M / 换更密的点云 / 显式加 `--allow_upsample` 用"保留全部+高分复制抖动"的扩展）。加载 ply 后立即检查，不做无用计算。
- `preprocess.sh` 默认回到论文模式：`NUM_SAMPLES=1500000`、`BASE_RATIO=0.0`、`ALLOW_UPSAMPLE=0`、`INIT_PLY=/GTLR/init_1.5m.ply`。
- 注意：服务器上现有的 `/GTLR/init_3m.ply`（extracted.ply 83.4 万 → 300 万）是上采样扩展生成的；严格论文模式下 extracted.ply 的 M 必须 < 834183，extracted_all.ply（247 万）必须 < 2475401。

### 上采样改为邻域插值（常规点云上采样）

- `upsample_jitter`（高斯抖动）已删除；新增 `upsample_interpolate`：每个新点落在"父点 ↔ 其最近邻池（默认 8 个）中随机一个邻居"的线段上，`p_new = (1-t)·p_parent + t·p_nbr`，t~U(0,1)，颜色同步插值——新点贴着局部表面，密度自适应，复用主 kNN 结果不产生第二遍 kNN。
- CLI：`--jitter_frac` 移除，新增 `--upsample_neighbors`（默认 8）；meta json 记录 `upsample_neighbors`。
- `preprocess.sh`：`JITTER_FRAC` → `UPSAMPLE_NEIGHBORS`。
- 测试：`test_upsampling_indices_and_jitter` → `test_upsampling_indices_and_interpolation`（验证每个新点确实共线于父点和某邻居、且 t∈[0,1]），14/14 通过。

### sample_points CLI 简化

- `--help` 只显示 4 个参数：`--input_ply` / `--output_ply` / `--num_samples` / `--allow_upsample`（M > N 时必需，上采样为邻域插值）。
- 其余旋钮（knn=64, seed=42, ref_max=200k, chunk_size=4096, upsample_neighbors=8, base_ratio=0）隐藏帮助、保留默认值，脚本仍兼容传参。
- `--device` 默认 auto：有 CUDA 就用 GPU，否则 CPU，不再需要手动指定。

### preprocess.sh 同步简化

- 只保留 6 个环境变量：`DATA` / `SRC_PLY` / `NUM_SAMPLES` / `ALLOW_UPSAMPLE` / `INIT_PLY` / `DEPTH_DIR`，其余全部用脚本默认值（knn=64、factor=4、vis_every=100、device=auto 等）。
- 默认 `NUM_SAMPLES=1500000`（< extracted_all.ply 的 247 万，论文模式）；需要上采样时 `ALLOW_UPSAMPLE=1`。
- 注释里写明：GPU 被占 OOM 时给 sample_points 加 `--ref_max 50000 --chunk_size 1024`。

### 采样 OOM 修复（2026-09-24）

- `curvature_texture` 的邻居 gather/协方差计算按 26 万点分块（原来一次性 gather N×64×3，247 万点时约 3.8 GB 显存峰值）。
- 默认 `ref_max` 200k→50k、`chunk_size` 4096→1024（kNN 瞬时距离矩阵 3.3 GB→约 200 MiB，共享 GPU 可共存）。
- 验证：extracted_all.ply 247 万点 → 150 万，GPU 全程 EXIT:0 无 OOM。

### 深度图改为 factor=1 全分辨率（2026-09-24）

- `preprocess.sh` 恢复 `FACTOR` 环境变量，默认 1（全分辨率 1024×1024）；**训练必须配 `--data_factor 1`**（manifest 校验会拦不一致）。
- 服务器已重新生成 `/GTLR/depth_maps/`（factor=1，864 张，每张约 20~25 万有效像素，是 factor=4 的 10 倍），factor=4 版本保留在 `depth_maps_f4/`。
- 全分辨率可视化确认：车位网格线在深度图上清晰可辨，与图像对齐正确。

### 重大数据问题：images.bin 位姿与 LiDAR 点云不在同一坐标系（2026-09-24）

- 现象：factor=1 深度图 208/864 张零覆盖（如 `1_01_00000442` 拍近处墙面却 0 个有效像素）。
- 排查链：
  1. 同一图像连该 COLMAP 模型自己的 points3D 点也投出 0 像素 → 不是点云裁剪问题；
  2. SfM 点与 LiDAR 云 bbox 吻合（都是 z 跨度 ~6m 的平场景），但相机轨迹 z 跨度 ±26m → 位姿和点云不同帧；
  3. `md5` 证实 `GTLR/data/sparse/0/images.bin` ≠ `parking01_section/sparse/0/images.bin`：前者是**未对齐的原始 COLMAP 模型**（864 张），后者是**对齐到 LiDAR 框架的模型**（11718 张，轨迹 z≈1.1m 平坦，与 `camera_centers.txt` 一致）；
  4. 两个模型内参完全相同（PINHOLE f=512, 1024×1024），只有位姿不同；Umeyama 验证两者不是同一轨迹的相似变换（RMSE 5.2m）→ 不能用变换纠正，必须换位姿。
- 修复：从对齐模型的 images.bin 过滤出 864 张对应位姿，替换 `GTLR/data/sparse/0/images.bin`（原件备份为 `images.bin.unaligned_backup`）。替换后轨迹 z 范围 1.02~1.15m，与 camera_centers.txt 逐张吻合。
- 结果：重新投影后 **0/864 张零覆盖**（原 208 张），平均每张 12.7 万有效像素，可视化对齐正确。
- 影响范围：此前所有 depth_maps（两个旧目录已改名 `depth_maps_unaligned_wrong*`）、smoke ckpt 均基于错误位姿，全部作废；init ply 不依赖位姿，不受影响。
- 教训：深度图可视化（`--vis_every`）是必要的数据检查手段，这次的帧不匹配就是靠它暴露的。
