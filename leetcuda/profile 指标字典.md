Author: Calcifer
Date: 2026/11/4
Link: [第 1 章：性能分析先于优化：nsys 与 ncu](https://leetcuda.readthedocs.io/en/latest/ch00-profiling.html#nsys-ncu "Link to this heading")

共有7个核心指标。

前置知识 [[GPU Arch]]、[[Roofline 与算术强度]]

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



