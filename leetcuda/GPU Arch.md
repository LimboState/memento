Author: Calcifer
Date: 2026/11/3
Link: [第 2 章：GPU 架构、执行模型与 Roofline](https://leetcuda.readthedocs.io/en/latest/ch01-arch-roofline.html#gpu-roofline)


# 1. SM Arch

| 部件                 | 规格 (速查口径)                                               |
| ------------------ | ------------------------------------------------------- |
| Wrap Scheduler × 4 | 每 SM 4 个 warp scheduler，每个每周期可发射 1 条指令                  |
| Register File      | 每 SM 65536 × 32-bit (4 bytes) = 256 KB                  |
| Shared Memory / L1 | 可配置划分，最大 shared memory ～ 228 KB (Hopper)                |
| Tensor Cores       | Hopper 每 SM 4 个；Blackwell 数量随型号/定义不同，建议以官分 ISV guide 为准 |
| Warp               | 32 threads，最小调度单元，SIMT 执行模型                             |

+ **Warp Scheduler × 4：** 每个 SM 有 4 个 warp scheduler，每个每周期可以发射 1 条指令。这意味着一个 SM 每周期最多发出 4 条指令，**发射带宽本身就是一种资源**。

+ **Register File：** 每 SM $65536 \times 32-bit = 256 KB$ 。这是 GPU 上最快也最稀缺的存储：所有活跃线程 (Active Threads) 的寄存器都从这里分配。寄存器分配以 warp 为粒度均分，若 kernel 每线程用 64 个寄存器，一个 256 线程的 block 就要占 $256 \times 64 \times 4B = 64 KB$，SM 里最多驻留 4 个这样的 block。

+ **Shared Memory / L1：** 同一块片上 SRAM 中可配置划分出的共享内存部分，Hopper 上这块 SRAM 为 256 KB，Hopper 上最多约 228 KB/SM 可用做 smem。block 内线程交换数据靠的就是它，后续章节 bank conflict、swizzle 也都在这块 SRAM 上展开。

+ **Tensor Cores：** Hopper 每个 SM 有 4 个第四代 Tensor Core，专门执行矩阵乘加 (MMA)； Blackwell 的数量与形态随型号有差异，源码注释提醒**以官方 ISV guide 为准**，后续我们将大篇幅讨论如何充分利用这些 Tensor Core。

+ **Warp = 32 threads：** warp 是**最小调度与指令发射单元**，32 个线程锁步执行同一条指令 (SIMT)。注意区分两个概念：线程是执行的最小粒度，warp 是调用与发射的最小粒度。硬件把 32 条线程通道捆绑成一个 warp，用一套取指-译码-发射逻辑服务 32 个计算 lane，控制逻辑的开销因此被摊薄，能以少量控制电路驱动大量 ALU。


![[Pasted image 20261103200424.png]]



# 2. Memory Hierarchy

**Memory Hierarchy 规律：** 越靠近计算单元，带宽越高、容量越小，相邻层级的带宽差在 1.5 倍到 5 倍之间 (寄存器按 100+ TB/s 下界计算，实测比值约 5.3)；容量上寄存器与 smem 同量级，再往上每升一级差两个数量级以上。

![[Pasted image 20261103201455.png]]

这张图给出 GPU 编成的一条基本规律：数据每上移一层，带宽就高一级，kernel 优化就是让数据尽量驻留在高层级。[[]]让归约的中间量全部驻留寄存器 (warp shuffle)，[[]] 用 tiling 把矩阵块搬进 smem 复用，[[]] 用 TMA+Warp Specialization 把搬运与计算解耦成流水线，走的都是同一条路。




