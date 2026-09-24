# GTLR-GS 数据预处理与训练

这套脚本固定使用注册后的 LiDAR/COLMAP 原始米制坐标。深度投影和训练均不做 world normalization，也不提供相应开关。

## 输入目录

`DATA` 应满足：

```text
DATA/
├── images/                  # COLMAP 使用的原图
├── images_4/                # factor=4 时 Parser 要求该目录存在
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    └── extracted_all.ply    # 与 COLMAP 坐标对齐、包含 xyz+RGB 的 LiDAR 云
```

如果点云不叫 `extracted_all.ply`，通过 `SRC_PLY` 指定。

## 1. 预处理

从仓库的 `examples` 目录或任意目录均可执行：

```bash
DATA=/GTLR/data \
WORK_DIR=/GTLR/artifacts \
NUM_SAMPLES=1500000 \
FACTOR=4 \
bash /GTLR/gsplat/examples/gtlr/preprocess.sh
```

预处理包含：

1. 按论文式 (1)–(4) 计算几何/纹理复杂度并分配初始化点；
2. 将注册 LiDAR 投影到 Parser 实际训练使用的去畸变 pinhole 图像，逐像素 z-buffer 保留最近 camera-z；
3. 输出抽样富集验证及定期深度叠加图。

主要配置：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `SRC_PLY` | `$DATA/sparse/0/extracted_all.ply` | 注册 LiDAR 点云 |
| `WORK_DIR` | `$DATA/gtlr` | 全部生成物根目录 |
| `NUM_SAMPLES` | `1500000` | 初始化高斯预算 |
| `FACTOR` | `4` | 图像缩放倍率，必须与训练一致 |
| `KNN_DEVICE` | `auto` | 采样近邻计算设备 |
| `DEPTH_DEVICE` | `auto` | 深度投影设备 |
| `DEPTH_CHUNK_SIZE` | `1000000` | 每次投影的最大点数 |
| `MIN_DEPTH` / `MAX_DEPTH` | `0` / `inf` | camera-z 有效范围（米） |
| `VIS_EVERY` | `100` | 每隔多少帧保存对齐图，0 为关闭 |

输出：

```text
WORK_DIR/
├── init.ply
├── init.ply.json
├── depth_maps/
│   ├── manifest.json
│   ├── *.npy
│   └── vis/*.png
└── validate_preprocess/
```

`manifest.json` 记录 factor、坐标系、camera-z 类型、像素中心约定和每帧有效像素数。训练器会强制校验这些字段，旧版本深度图需要重新生成。

## 2. 训练

```bash
DATA=/GTLR/data \
WORK_DIR=/GTLR/artifacts \
FACTOR=4 \
CUDA_DEVICE=0 \
bash /GTLR/gsplat/examples/gtlr/train.sh
```

训练脚本默认执行 30k 步，深度监督从 3000 步开始，保持 GTLR 的三部分：几何纹理初始化、曲率自适应分裂、置信度加权 LiDAR 深度正则及 LiDAR 法线约束。

常用覆盖项为 `MAX_STEPS`、`DEPTH_START_ITER`、`DEPTH_LAMBDA`、`NORMAL_LAMBDA`、`CURVATURE_REF_MAX`、`RESULT_DIR`。其余参数可直接追加到脚本末尾并转交给 trainer。

快速检查深度模块时，训练步数必须超过 `DEPTH_START_ITER`：

```bash
DATA=/GTLR/data WORK_DIR=/GTLR/artifacts \
MAX_STEPS=4000 DEPTH_START_ITER=500 \
bash /GTLR/gsplat/examples/gtlr/train.sh
```

训练前先查看 `depth_maps/vis/`。彩色点应贴合墙面、地面和物体边界；若整体错位，应先检查 LiDAR–相机外参和 COLMAP 图像对应关系，不要靠训练补偿。
