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
