Author: Calcifer
Date: 2026/11/5
Link: [第 1 章：性能分析先于优化：nsys 与 ncu](https://leetcuda.readthedocs.io/en/latest/ch00-profiling.html#nsys-ncu "Link to this heading")

**共有7个核心指标。**

前置知识 [[nsys 与 ncu 工具链与方法论]]、[[GPU Arch]]、[[Roofline 与算术强度]]

---

**==🟣性能分析==** [[Roofline 与算术强度]]

![[Pasted image 20261104193831.png]]

表 1： H100 SXM 参考口径

ridge point： FP16 TC dense $989/3.35 \approx 295$，FP32 $67/3.35 \approx 20$。

对照三个算子：

- GEMM K=4096 (AI 683) 在两条线上都 compute-bound。
- GEMV (AI 0.5) 远低于 $AI^*$，天花板是带宽本身。
- Softmax (AI 0.625) 远低于 $AI^*$，天花板是带宽本身。


---

# 1. Speed-Of-Light (Roofline)

>SM% 与 MEM% (roofline 定位)

`sm_throughput` (SM%) 与 `gpu_compute_memory_throughput` (MEM%) 分别是计算管线与内存管线的峰值利用率，是 NCU 报告的第一行。

在 kernel 的 roofline： SM% 高 MEM% 低是 compute-bound，反之 memory-bound。**双低** (如 SM/MEM 都在 30% 附近) 则是 latency-bound —— 两条管线都没有饱和，问题在并行度或依赖链，这是 roofline 图上看不出来的第三态，必须回到 stall 指标找证据。


---

# 2. Occupancy

>理论、达到与三类限制源

区分两个数：**理论 occupancy** (kernel 资源能驻留多少 warp，受寄存器、smem、block/warp 上限三类限制源钳制) 与**达到 occupancy** (`sm_warps_active`，实际运行均值，受 tail effect 与负载不均拖累)。

NCU 的 Occupancy section 会明确指出 `launch__occupancy_limit_registers` / `__shared_mem` / `__blocks` 哪个是当前限制源。


一个 fp8 attention kernel 的实例：
> H100：$256 KiB = 64K \times 4B = 65536 \times 4B = 65536$个寄存器；
> SM -> CTA (多组 warps，每个 warp 32 线程)；  
> 
> WS 结构 (Warp Specialization，4 producer + 8 consumer，共 384 线程) 的 launch 配置按 $65536/384$ 把静态寄存器限制到 168，`Block Limit Registers` 因此恒为 1；稳态再按 `setmaxnreg` 分成 $128(4 \times 32) \times 32(regs) + 256(8 \times 32) \times 232(regs) = 63488$ 个 (占 65536 的 96.9%，余下的 2048 恰是 [[]] 所述 slack 下限)，第二个 CTA 需要 126976，故 1 CTA/SM、12 warps，理论 occupancy 25% (12/48)，此时想提并行度只有降寄存器或换结构。
> 
> 注意 [[AI Infra 面经]] 的告诫 —— occupancy 是手段不是 KPI，GEMM 类 kernel 用大 tile + 低 occupancy + 高 ILP 换性能是常态。


![[Pasted image 20261105140622.png]]

如图，理论 occupancy 的三限制源取 min(sm_120, 48 warp 槽位口径)：
+ 寄存器约束 $65536/(32r)$ (warp 粒度简化) 随每线程寄存器数 $r$ 下降；
+ smem 与槽位为水平约束 (示例： smem/block=32KB、T=256，sm_120 的 100KB/SM -> 3 block = 24 warps，槽位约束 warps 上限为 48，min 取最小值为 24 warps 上限)。粗黑阶梯为按整个 block (T=256，8 warps) 量化的 min 包络。
+ 实测点为 WS 结构 (producer 32 / consumer 232 regs) 1 CTA/SM = 12 warps (T=384)，理论 occupancy 25% (12/48)。

> **T：** 表示每个 block 的线程数，例如 T=256，表示 block 包含 256/32=8 个 warps。


---


# 3. 寄存器与 spill

>255 墙


