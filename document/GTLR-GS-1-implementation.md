# GTLR-GS 复现 · 一：第一版实现

> 系列文档：① 第一版实现（本文） → ② [审查发现的问题](GTLR-GS-2-review.md) → ③ [后续修复记录](GTLR-GS-3-fixlog-2026-09-22.md)
>
> 论文：GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction（arXiv:2603.23192，无官方代码）
>
> 代码：`/Users/zongcheng/code/3dgs/gsplat`，分支 `gtlr`。原实验副本：服务器 `/media/zc/SSD/GS/lutu/GTLR/gsplat-gtlr/`
> 测试数据：`/media/zc/SSD/GS/lutu/parking01_section`（864 图 + 247 万点 LiDAR 云）
> 容器镜像：`harbor.aibee.cn/robot/gsplat:V2`（torch 2.8.0+cu128, RTX 2080Ti）

> ⚠️ 本文记录第一版实现及其原始实验过程。之后的源码审查在其中发现了初始化坐标系、近邻索引空间等确定问题（见文档②），本文命令中与之相关的部分已被修复（见文档③）；复现时请以文档③之后的代码状态为准。

## 一、复现思路（论文三模块 → gsplat 映射）

| 论文模块 | 实现位置 | 要点 |
|---|---|---|
| ① 几何-纹理感知采样（式1-4） | `examples/gtlr/sample_points.py` | 离线脚本，纯 PyTorch，不改 gsplat |
| ② 曲率自适应分裂（式5-6） | `gsplat/strategy/gtlr.py` 的 `GTLRStrategy(DefaultStrategy)` | 只改 `_grow_gs`：split 加门控 `κ_i > θ(t)`，θ 从 0.1 线性升到 0.3；duplication/prune 逻辑不动 |
| ③ 置信度深度正则（式8-10）+ 法线损失（式7） | `examples/gtlr/geom.py`、`examples/gtlr/project_depth.py` | 无偏深度用 `rasterization(extra_signals=...)` 渲 4 通道（法线 3 + 平面距离 1）再相除，**无需改 CUDA** |
| 训练入口 | `examples/gtlr/simple_trainer_gtlr.py` | `L = lerp(L1, SSIM, 0.2) + 1.0·L_depth + 0.05·L_normal`（式11 + 式7） |
| 离线验证 | `examples/gtlr/validate.py` | 深度对齐可视化、无偏深度误差统计、采样富集对比 |

### 无偏深度的关键推导（为什么 gsplat 能渲）

式(8)：

```
D̂(p) = Σᵢ wᵢ Dᵢ / Σᵢ wᵢ (nᵢ · rₚ)     rₚ = K⁻¹p̃（像素光线，z=1）
```

- 分子 `Σ wᵢ Dᵢ`：Dᵢ = nᵢ·μᵢ^cam 是 per-Gaussian 标量 → 1 个 extra channel 直接 alpha 累加；
- 分母：`Σ wᵢ (nᵢ·rₚ) = (Σ wᵢ nᵢ)·rₚ`（rₚ 逐像素固定）→ 法线 3 个 extra channel 累加后，图像域逐像素与光线方向点积。

合计 4 个 extra 通道（实现时 pad 到 5，凑齐 kernel 已编译的 8 通道），梯度经 `extra_signals` 回流到 means/quats，不需要写 CUDA kernel。

### 曲率门控（式5-6）

- κ 用原始定义 λ1/(λ1+λ2+λ3)，值域 [0, 1/3]，**不做 min-max 归一化**（与 θ∈[0.1, 0.3] 的阈值区间匹配）；
- θ(t) = 0.1 + 0.2·t/T，T = 总训练步数；
- 性能近似：只对 split 候选（`is_grad_high & is_large`）算曲率，参考集随机子采样（默认 ≤20 万点，共享 GPU 时建议 `--strategy.curvature-ref-max 50000`）。

## 二、运行流程（第一版，后续命令以文档③为准）

```bash
# 0. 启动常驻容器（编译缓存持久化在容器可写层）
docker run -d --name gtlr --gpus all \
  -v /media/zc/SSD/GS/lutu/GTLR:/GTLR \
  -e PYTHONPATH=/GTLR/gsplat-gtlr -e TORCH_CUDA_ARCH_LIST=7.5 \
  harbor.aibee.cn/robot/gsplat:V2 /entrypoint.sh
# 首次 import gsplat 触发 JIT 编译约 18 分钟（sm_75）

cd /GTLR/gsplat-gtlr/examples   # 后续命令都在此目录、容器内执行

# 1. 点云下采样：LiDAR 点云 → 高斯初始化点云
python -m gtlr.sample_points --input_ply /GTLR/data/sparse/0/extracted_all.ply \
    --output_ply /GTLR/init_1.5m.ply --num_samples 1500000

# 2. 深度图生成：点云投影到每个相机（第一版要求 --normalize 与训练器坐标系一致；
#    修复后 normalize_world_space 统一为 False，不再加该参数）
python -m gtlr.project_depth --data_dir /GTLR/data \
    --ply /GTLR/data/sparse/0/extracted_all.ply \
    --output_dir /GTLR/depth_maps --factor 4

# 3. 训练（法线损失默认开启；深度损失默认 step 3000 后启用，见"坑 6"）
python -m gtlr.simple_trainer_gtlr --data_dir /GTLR/data \
    --init_ply /GTLR/init_1.5m.ply --depth_dir /GTLR/depth_maps \
    --result_dir /GTLR/results/full --strategy.curvature-ref-max 50000

# 4. 离线验证（无需训练）
python -m gtlr.validate --data_dir /GTLR/data --ckpt /GTLR/results/full/ckpt_29999.pt \
    --depth_dir /GTLR/depth_maps --full_ply /GTLR/data/sparse/0/extracted_all.ply \
    --sampled_ply /GTLR/init_1.5m.ply --output_dir /GTLR/results/validate
```

## 三、数据检查流程

### 1. 点云下采样（`sample_points.py`）

- **原理**：每点 kNN(k=64) 协方差特征值 λ1≤λ2≤λ3 → 曲率 κ=λ1/(λ1+λ2+λ3+ε)（式1-2）；邻域 RGB 三通道方差均值 → 纹理 τ（式3）；各自 min-max 归一化后按 P=0.5κ̂+0.5τ̂ 无放回采样 M 点（式4）。
- **检查**：`validate.py` 的采样验证——在全云上统一参考集算 κ̂/τ̂，cKDTree 精确匹配采样点索引，对比采样集 vs 随机集均值，采样集应显著更高（enrichment > 1）。
- **注意**：输入 ply 必须 xyz+RGB；M ≤ 云点数 N（本次 N=247 万 < 论文默认 300 万）；无 open3d 时自动用 torch 分块 kNN（参考集 ≤20 万近似）。

### 2. 深度图生成（`project_depth.py`）

- **原理**：点云按 COLMAP 内外参投影，`scatter_reduce(amin)` z-buffer 取每像素最近深度，逐相机存深度 `.npy`（0 = 无效像素，即式10 的 U 集补集）。
- **检查（数值）**：抽查若干 npy 的有效像素数与深度范围。修复后全程为米制坐标（normalize=False），深度应落在场景实际尺度（停车场约 0.1–30m 量级）；同时核对 `manifest.json` 的 factor/normalize 与训练配置一致。
- **检查（可视化）**：`validate.py` 输出的 `depth_check_*.png` 第一格是"图像调暗 + LiDAR 深度彩色叠加"，彩色区域应贴合地面/墙面边界。
- **注意**：部分相机可能完全无 LiDAR 覆盖（原实验 864 张中存在空图），训练器只跳过有效深度像素 <500 的视图的**深度监督**，仍保留其 RGB 训练；深度图分辨率必须与训练 `data_factor` 一致（原数据默认 4 → 256×256）。

### 3. 法向量计算（`geom.py: estimate_normals` / `associate_normals`）

- **原理**：初始化点云上 kNN(k=16)-PCA，最小特征值对应特征向量作为法线（式7 的 n_lidar）；训练中每个高斯按最近邻关联 LiDAR 法线，refine 后刷新关联；高斯自身法线 = 旋转矩阵最短缩放轴（与式8 平面法线同源）。
- **检查**：法线是局部 PCA 结果，用 κ 分布佐证——平坦区 κ≈0、边缘区大（原实验实测：中位 0.06，32% > θ_start=0.1，几乎无 > θ_end=0.3）；若 κ 恒为 0 说明 kNN 失效。
- **注意**：法线参考集 cap 到 10 万点，关联计算放 CPU（避免共享 GPU 上 kNN 瞬时显存 OOM）。

## 四、踩过的坑（复现必读）

1. **sparse 模型与图像数量不一致**：images.bin 注册 11718 张但磁盘只有 864 张（section 子集），gsplat Parser 按排序列表 zip 对齐会**静默错位**。必须先过滤模型（`GTLR/filter_sparse_bin.py`，直接重写 COLMAP 二进制；原始模型备份在 `sparse/0_full`）。
2. **exFAT 不支持软链**：`/media/zc/SSD` 是 exFAT，数据只能实拷（`GTLR/data/` 下约 1GB）。
3. **坐标系**：第一版训练器 `normalize_world_space=True` 而 init ply 未随之变换（即审查 R1）；修复后统一 `normalize_world_space=False`，全程米制坐标（见文档③）。
4. **fork 源码 bug**：本地 `NVSQuality/gsplat` 的 `Utils.cpp`、`SphericalHarmonicsCUDA.cu`、`SphericalHarmonicsL1PlusCUDA.cu` 三处把 `cudaEventCreateWithFlags(&e, flags)` 误写成 `cudaEventCreate(&e, flags)`，任何机器上 JIT 编译都会失败。已在 gtlr 分支修复，原始仓库未动。
5. **torch extension 残留锁**：中断编译后须删除 `/root/.cache/torch_extensions/py310_cu128/gsplat_cuda/lock`，否则后续所有 `import gsplat` 永久阻塞（采样脚本因 `from gsplat.strategy.gtlr import knn_indices` 连带被卡）。
6. **无偏深度的稳定性仍待验证**：原实验记录早期深度爆炸，采用 step 3000 后启用深度损失（`--depth-start-iter`）及门控、截断缓解。warmup 是本实现的工程选择；第一版源码还有坐标、尺度初始化错误，不能将异常全部归因于随机法线，更不能把 loss 变小视为几何已经修复。详见文档② R1–R3、R5。
7. **kNN 分块显存/内存**：`chunk_size=65536` 时距离矩阵 65536×200000×4B = 52GB（曾把 62GB 内存吃到 swap）。已改 4096；共享 GPU 上曲率参考集降到 5 万（瞬时 ~800MB）。
8. **镜像 entrypoint**：`docker run harbor.aibee.cn/robot/gsplat:V2 python xxx` 会报 "cannot execute binary file"，必须用脚本文件作为入口命令。

## 五、原实验状态与验证记录（历史记录，工件已在文档③中作废）

| 项目 | 状态 |
|---|---|
| CUDA 编译 + `extra_signals` 前向/反向 | ✅ 冒烟通过（梯度回流 means/quats/extra） |
| 深度图投影对齐 | ✅ 可视化叠加边界吻合 |
| 曲率门控 | ✅ κ 分布合理，θ=0.1 时约 32% 候选通过 |
| 冒烟训练（400k init，1000 步） | ✅ 85 秒跑完，PSNR 17.44 / SSIM 0.637（注意：1000 步 < depth_start_iter=3000，深度模块实际未参与） |
| 无偏深度早期爆炸 | ⚠️ 原记录采用 warmup + min_ray_dot=0.1 + 逐像素 clamp=2.0 缓解；不能认定已修复，见文档② R3、R5 |
| 完整 30k 训练 | ⏸ 待启动（等显存：另有任务占 6.2GB + 1.8GB / 共 11GB） |
