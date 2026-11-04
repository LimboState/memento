**优化技术概述**

Author: Calcifer
Date: 2026/11/3
Link: [第 2 章：GPU 架构、执行模型与 Roofline](https://leetcuda.readthedocs.io/en/latest/ch01-arch-roofline.html#gpu-roofline)


常见的优化手段可以分为 9 种，在这里列出形成一张路线图。

![[Pasted image 20261104191624.png]]


前三条：

+ **Coalesced Memory Access (合并访存)：** 同一 warp 的线程访问**连续的 128B 对齐地址**，只需 1 次内务事务；否则产生多次事务 (最坏 32 次)。
+ **Tiling (分块)：** 将数据从 HBM 分块加载到 shared memory 复用，减少 HBM 访问；GEMM 的形态是 Block Tile (BM × BN) 加 K Tile (BK)。
+ **Vectorized Memory Access (向量化)：** 使用 `float4` / `half2` 等向量类型，减少 load / store 指令数；`float4` 是 128-bit，但条指令加载 16 bytes。

---



