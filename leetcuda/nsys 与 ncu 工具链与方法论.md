Author: Calcifer
Date: 2026/11/2
Link: [第 1 章：性能分析先于优化：nsys 与 ncu](https://leetcuda.readthedocs.io/en/latest/ch00-profiling.html#nsys-ncu "Link to this heading")

1. **profile 永远都是最重要的。**
2. **测量工具： Nsight Systems (nsys，全局时间线) 与 Nsight Compute (ncu，单kernel计数器)。**
3. **GPU 上的性能瓶颈很少出现在直觉指向的位置，没有 NCU 数据，不改 kernel。**


==🟣**nsys 与 ncu 的原理完全不同：**==

+ **nsys (Nsight Systems):** trace 型。在驱动层记录 ==🟡API 调用==、==kernel launch==、==内存拷贝==、==流与事件==的时间戳，拼出一张全局时间线。它回答时间花在哪？：哪个 kernel 占比最大、kernel 之间有没有空隙 (gap)、CPU launch 是否跟不上、多流是否真的在重叠。代价低：不需要 GPU 性能计数器权限，任何环境都能跑，被测程序几乎不失真。
+ **ncu (Nsight Compute):** counter replay 型。要读一个 kernel 的成百上千个硬件计数器（SM 吞吐、cache 命中、warp stall 等），而一遍执行次采不完这么多的计数器，于是 ncu 把 kernel 回放多遍，每遍采一组；为了数据可比还会锁定时钟。它回答这个 kernel 内部慢在哪。代价高：单 kernel 深挖、时间被串行化与放大，ncu 报告里的 kernel 耗时 **不能当 bench 时间用**。


==🟣**先 nsys 后 ncu：顺序不是习惯问题，是成本问题。**==

![[Pasted image 20261102162700.png]]

1. **nsys:** 一次采集拿到全部 kernel 的时间占比排序。如果主 kernel 占 86%，那么辅助 kernel 再慢也只有 14% 的优化上限： 先确认该优化谁，再谈怎么优化。nsys 还能立刻暴露三类结构性问题： **launch gap (CPU 跟不上)**、**该并行的流没有重叠**、**意外的 fallback kernel**。
2. **ncu:** 只对 nsys 排出的 Top-1/Top-2 kernel 做 `--set full` 深挖，读 roofline、occupancy、stall、memory 各维指标，形成假设。
3. **单变量修改 -> 复测:** 每次只改一处，re-profile 后用 `--diff` 对照前后指标；无效或负优化立即回退。


How to use nsys and ncu?  [[nsys 与 ncu 实践]]