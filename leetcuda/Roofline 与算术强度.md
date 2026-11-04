Author: Calcifer
Date: 2026/11/3
Link: [第 2 章：GPU 架构、执行模型与 Roofline](https://leetcuda.readthedocs.io/en/latest/ch01-arch-roofline.html#gpu-roofline)

前置知识：[[GPU Arch]]
优化手段：[[An Opt Method Overview]]

# 1. 算术强度 (Arithmetic Intensity， AI)

定义一个算法 (或一段 kernel) 的算术强度：

$$AI = \frac{FLOPs}{Bytes}$$
其中 FLOPs 是算法需要的浮点运算次数 (**总量**，单位 FLOP，不是每秒)，Bytes 是它必然要在最外层存储 (通常是 HBM) 上搬运的字节数，口径是每个数据读一次、写一次。AI 的单位是 FLOP/Byte，物理含义是：每搬运 1 字节，能换来多少次浮点运算。

三个注意点：
+ 第一，FLOPs 数的是算法固有的量：乘加各计 1 次，因此一个点积 $a \cdot b$ (N 项) 是 2N 次。
+ 第二，Bytes 是**不可再省的下界**：若算法要读一遍 a 写一遍 y，无论 cache 命中率 (cache 之间、cache 和 HBM 之间) 多高，HBM 流量都不会低于 $4(|a| + |y|)$ 字节 (fp32)。
+ 第三，中间量不算：softmax 的逐元素 max、exp、sum 都在寄存器里完成，不出 HBM，所以不计入 Bytes。


**==🟣三个典型算子的 AI 逐项推导：GEMM、GEMV、Softmax==**

**GEMM** ($C = AB$，尺寸 $M \times K$ 乘 $K \times N$)：输出 C 有 $M \times N$ 个元素，每个元素 $c_{ij} = \sum_{k} a_{ik} b_{kj}$ 需要 K 次乘与 K 次加，共 2K 次；所以

$$ FLOPS = 2MNK.$$
Bytes 按三个矩阵各读写一次计：

$$ Bytes = 4(MK + KN + MN).$$
当 M = N = K 时有简洁的解析形式：

$$ AI_{GEMM} = \frac{2K^3}{3K^2 \cdot 4}= \frac{K}{6}$$
$K = 4096$ 时 $AI \approx 682.7 FLOP/Byte$。AI 随着 K 的线性增长：GEMM 天然是 compute-bound 算法，矩阵越大每字节访存对应的计算量越多，这也是它适合 GPU 与 Tensor Core 的根本原因。

---

**GEMV** ($y = Ax$)：$A$ 是 $M \times K$，$x$ 是 $K$ 维向量，$y$ 是 $M$ 维输出。$y$ 的每个元素要 $K$ 次乘 $K$ 次加，$FLOPs = 2MK$； 而 $Bytes = 4(MK + K + M)$。大 $MK$ 主导下：

$$ AI_{GEMV} \approx \frac{2MK}{4MK} = 0.5 FLOP/Byte$$

矩阵 A 的每个元素只读取一次，没有任何复用，AI 就固定在 0.5，比 GEMM 低三个数量级，是彻底的 memory-bound。[[]] 将看到三种 warp 划分方式如何逼近这 0.5 对应的理论带宽。

---

**Softmax** (逐 token，宽度 $N$)：对长度 $N$ 的行做 softmax —— $N$ 次减法、$N$ 次 $exp$ (每条计 2 FLOP，共 $2N$）、$N$ 次求和、$N$ 次除法，共约 $5N \: FLOPs$ (max 比较不计 FLOP)；访存是读入 $x$、写出 $y$ 各一次，$4 \times 2N$ 字节：

$$ AI_{softmax} = \frac{5N}{8N} = \frac{5}{8} = 0.625 $$

[[]] 的三级递推 (naive -> safe -> online) 优化的是访存趟数与同步次数 (全局扫描与 block 同步同为 1 -> 2 -> 1)，该口径下 AI = 0.625 告诉我们：无论怎么优化，它始终是 memory-bound，最终性能用离理论带宽多近来衡量。

---


# 2. Roofline 模型与 ridge point

把峰值算力 $P_{peak}$ (FLOP/s) 与峰值带宽 $BW$ (Byte/s) 放在一起，Roofline 模型给出任意算子的可达性能上界。

$$ P = min(P_{peak},\: AI \times BW).$$

它的图像是一整条屋顶线：左边是一段斜率为 $BW$ 的斜坡 (强度每多 1 FLOP/Byte)，性能就多$BW$，带宽是瓶颈)，右边是一段水平的天花板 ($P_{peak}$，算力是瓶颈)。两段交点的横坐标叫 **ridge point：**

$$ AI^* = \frac{P_{peak}}{BW}. $$
判断规则由此而来：$AI < AI^*$ 则 memory-bound，$AI > AI^*$ 则 compute-bound。

![[Pasted image 20261104114514.png]]

如图，Roofline 模型：$P = min(P_{peak}, \: AI \times BW)$，脊点 $AI^* = P_{peak} / BW$。GEMM $K=4096$ ($AI = 683$) 落在计算屋顶，softmax (AI = 0.625) 落在带宽斜线区。

---

同一个 GPU 上有多条屋顶线。FP32 CUDA core 与 FP16 Tensor Core 的 $P_{peak}$ 相差约 15 倍，于是 ridge point 也相差 15 倍 —— 在 H100 上，AI=100 的算子用 FP32 跑是 compute-bound (100 > 20)，切到 FP16 TC 反而是 memory-bound (100 < 295)。这个解释了一个反直觉的现象 —— 把 kernel 迁移到 Tensor Core 上未必变快。如果它本身是 memory-bound，算力提高 15 倍也无法提升性能。

同理，把 ridge point 算错 (比如把 sparse 峰值当 dense 用， [[]])，会把 bound 类别判反，整个优化方向就全错了。

![[Pasted image 20261104161501.png]]

如图，同一 GPU 的两条屋顶线 (H100 SXM) —— FP32 ($P_{peak} = 67 \: TFLOP/s, ridge = 20$) 与 FP16 TC dense ($P_{peak} = 989, ridge = 295$) 共用同一条带宽斜坡 (BW = 3.35 TB/s)，脊点相差约 15 倍。

AI = 100 的竖虚线在两条线上落点相反 —— FP32 侧已 compute-bound (100 > 20)，FP16 TC 侧仍 memory-bound (100 < 295)。把 kernel 迁移 Tensor Core 未必变快；ridge 口径算错 (如 sparse 当 dense) 会把 bound 类别判反。

---


# 3. Occupancy 与延迟隐藏

Roofline 描述算法与机器的匹配，occupancy 描述实现的并行度。

GPU 隐藏内存延迟靠的不是 cache (容量太小)，而是 **warp 切换：** 当 warp A 因等待 HBM 数据而停顿时，scheduler 立刻切到 warp B 发射指令；等到 A 的数据到了再切回来。

设单次 HBM 访问约 500 ～ 800 个时钟周期，若要发射口持续有指令可发，SM 上必须常驻足够多的活跃 warp。活跃 warp 数与硬件上限之比就是 occupancy。

$$ occupancy = \frac{active \  warps / SM}{max \  warps / SM}$$

它同时受三类资源钳制，实际值取各约束下的最小值：

+ **每线程寄存器数**。寄存器用的越多，能驻留的 warp 越少。
+ **每 block 的 smem 用量**。smem 用得多，block 数就少。
+ **block 大小**。block 太小则凑不满 SM 的线程槽位。

后续讨论中将会反复出现这类 Trade-Off：[[]] 的寄存器双缓冲用更多寄存器换流水线；[[]] 的 setmaxnreg 干脆在 producer/consumer 之间动态重新分配寄存器池。都是在 occupancy 这个维度上做取舍。

>block (线程块) 是软件层面的线程组织单位，一个 block 包含若干 warp。block 是不可分割地整体调度到一个 SM 上的，一个 block 的所有 warp 必须驻留在同一个 SM 上，不能跨 SM 拆分。



---



最后，这里补充一张一次 HBM 读的全程，可以将上述讲到的 3 个概念串成时间线：

![[Pasted image 20261104193608.png]]

一次访存的全程与延迟隐藏 (蓝 = 发射/执行，橙 = 等待/过渡延迟，绿 = 命中与恢复，红 = 直达 HBM 的最长延迟)。




