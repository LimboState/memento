Author: Calcifer
Update Date: 2026/11/4


# 1. 常见坑

1. **Bytes 漏算写回：** AI 分母必须包含输出写回 (读 $a$、$b$，写 $y$ 是 $3 \times 4N$ 字节，不是 $2 \times 4N$)；漏算会把 memory-bound 算成 compute-bound。[[Roofline 与算术强度]]

2. **sparse 峰值当 dense 用：** NVIDIA 规格表上带 sparsity 的 TFLOPS 是稠密口径的两倍；ridge point 差一倍，bound 类别可能判反。[[Roofline 与算术强度]]

3. **把 occupancy 当唯一 KPI：** occupancy 上限 100% 需要少用寄存器、少用smem，但 GEMM 类 kernel 恰恰要用大 thread tile 与多级 smem 流水换性能；低 occupancy + 高 ILP 常优于高 occupancy + 低 ILP。occupancy 是手段，roofline 位置才是目标。[[Roofline 与算术强度]]

4. **用小数据集标定带宽：** 在测带宽 (microbenchmark) 时，如果数据集太小，测出来的可能是 L2 缓存的速度，而不是 HBM 显存的速度。16 MB 的输入可装入 96 MB 的 L2，在 bench 的 warmup 后的稳定工作集下可保持高 L2 命中，测出的带宽是 L2 的，实测 1862 GB/s > HBM 理论 1344 GB/s，但容量关系本身不保证永不换出。标定 HBM 带宽时数据集至少要数倍于 L2 容量。[[Roofline 与算术强度]]

5. **混淆 FLOP 与 FLOPS：** AI 的分子是运算量 (FLOP)，roofline 纵轴是速率 (FLOP/s)；单位写错，量纲就全乱了。[[Roofline 与算术强度]]


---

# 2. 面试要点速查

+ **Q： warp 是什么？ 为什么是 32？**
	A：最小调度与指令发射单元，32 线程锁步 (SIMT)。32 与 warp scheduler 的每周期发射粒度、寄存器文件 bank 宽度匹配；发散 (divergence) 会让一个 warp 内的分支 两条路径串行执行，等效指令数翻倍。

+ **Q：怎么区分 memory-bound 与 compute-bound？**
	A：算 AI，与 ridge point $AI^* = P_{peak} / BW$ 比较 —— 低于则带宽瓶颈，高于则算力瓶颈。优化方向完全不同：前者靠 coalescing / 向量化 / tiling，后者靠 Tensor Core / 指令效率。

+ **Q：为什么需要 occupancy？**
	A：延迟隐藏。GPU 不做分支预测与深级乱序，靠大量活跃 warp 之间的快速切换填满发射口；occupancy 低则切换不了，HBM 延迟 (约 500 ～ 800 cycle) 直接暴露。但 occupancy 受 寄存器/smem/block 大小三重钳制，是权衡不是越大越好。

+ **Q：寄存器溢出 (spill) 是什么？**
	A：每线程寄存器需求超过分配上限时，编译器把局部变量挪到 local memory (实际在 HBM/L1)，访存量大幅度增加、延迟暴露。用 `--ptxas-options = -v` 观察 spill 次数，出现即优先重估 thread tile 大小。

+ **Q： L1 与 shared memory 的关系？**
	A：同一块片上 SRAM 的两种角色，可配置划分。L1 对所有访存透明缓存；smem 由程序员显式管理、block 内共享。Hopper 上 smem 上限约 228 KB/SM。













