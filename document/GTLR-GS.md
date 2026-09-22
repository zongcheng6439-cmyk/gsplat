# GTLR-GS 复现文档索引

> 论文：GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction（arXiv:2603.23192，无官方代码）
>
> 代码：`/Users/zongcheng/code/3dgs/gsplat`，分支 `gtlr`

系列文档按时间顺序分为三篇：

1. **[第一版实现](GTLR-GS-1-implementation.md)**：论文三模块到 gsplat 的映射思路（复杂度采样初始化 / 曲率门控分裂 / extra_signals 无偏深度正则）、运行流程、数据检查流程（点云下采样、深度图、法向量）、第一版踩坑记录与原实验状态。
2. **[中间的问题（源码审查 R1–R15）](GTLR-GS-2-review.md)**：2026-09-22 审查结论——第一版存在初始化坐标系、近邻索引空间、深度梯度截断等确定问题；含 15 项问题的位置/影响/建议/验收标准、审查实际验证边界和建议的修复顺序。
3. **[后续修改（修复记录）](GTLR-GS-3-fixlog-2026-09-22.md)**：第一阶段修复（R1/R2/R6/R8/R13，输入与初始化正确性）、`normalize_world_space` 统一为 False 的全局决定、作废待重生成的服务器工件清单、仍未修复项。

当前状态：**模块原型与局部验证**（审查结论），第一阶段修复完成、测试 12/12；完整效果实验待第二、三阶段修复后进行。
