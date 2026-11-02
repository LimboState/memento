Author: Calcifer
Date: 2026/11/2
Link: [第 1 章：性能分析先于优化：nsys 与 ncu](https://leetcuda.readthedocs.io/en/latest/ch00-profiling.html#nsys-ncu "Link to this heading")

What is nsys and ncu?  [[nsys 与 ncu 工具链与方法论]]

==🟣nsys 命令行实战==

**最小采集：** 原则是只采需要的域、只采最短的窗口。对于采集 CUDA API 和 kernel 活动这一目标，一条 trace 域 `-t cuda` 就够，CPU 采样与上下文切换全部关掉，`--sample=none`、`--cpuctxsw=none`，warmup 与 repeat 降到最小。

```shell
nsys profile -t cuda --sample=none --cpuctxsw=none \ 
	-o .tmp/ch00/report.nsys-rep python bench.py
```

**缓存爆炸：** nsys 的中间数据默认写到 `/tmp/nvidia/nsight_systems`，体积≈单事件字节数 × 事件数，而事件数随时间长线性增长。对大模型推理（多模型链、大 seqlen、多次 repeat）不加限制的全量采集，曾把 `/tmp` 写到 500GiB+ 直到磁盘写满系统卡死。三条铁律：
+ **输出到指定临时文件：** 显式设 `TMPDIR` 指向项目 `.tmp/` 下的目录并用 `-o` 制定输出，事后清理。注意 `TMPDIR` 路径要短，过长的路径会超过 Unix domain socket 的 107 字节路径上限，导致 Ray 的 plasma store 构造失败。
+ **禁止对大模型不加限制地全量采集：** repeat 降到 0-1 ，单域 `-t cuda` ，关采样。
+ **优先 torch.profiler：** 只想看 kernel 时间表时，`torch.profiler` 输出 chrome trace 落在 `.tmp/` ，零外部临时文件风险；nsys 只再必须看时间线/同步事件时使用。

**窗口化采集：** 只想采推理中的一小段时，用 `--capture-range=cudaProfilerApi` 配合代码里的开关，把采集窗口限定再目标区间：

```shell
nsys profile -t cuda --capture-range=cudaProfilerApi \
	--sample=none --cpuctxsw=none -o ./tmp/ch00/win python bench.py
# bench.py 内：
#    torch.cuda.profiler.start() # 窗口开始
#    ... 目标 step ...
#    torch.cuda.profiler.stop()  # 窗口结束
```


**stats 报表：** 

+ 采集产物是 `.nsys-rep`，用 `nsys stats` 导出统计。
+ `cuda_gpu_kern_sum` 按 kernel 聚合总时长与占比，直接回答该优化谁的问题。
+ `cuda_gpu_trace` 逐 launch 列时间戳，用来看相邻 kernel 间的空隙与多流 overlap。

```shell
nsys stats -r cuda_gpu_kern_sum report.nsys-rep # per-kernel 时间占比(最常用)
nsys stats -r cuda_gpu_trace report.nsys-rep    # per-launch 时间线  (看gap/流重叠)

# 首次对旧 report 运行时报 Existing SQLite export found 时，
# 加 --force-export=true 强制重新导出。
```


==🟣ncu 命令行实战==

```shell
ncu --set full -o .tmp/ch00/report_v1 --target-processes all ./kernel_app
ncu --import .tmp/ch00/report_v1.ncu-rep --page raw --csv > report_v1.csv
```

+ `--set full` 采全部 section（roofline、occupancy、warp state、memory workload、instruction mix等），单 kernel 回放多遍、耗时可观。
+ 快速确认可用默认 basic set 。
+ `ncu-ui` 交互式看报告。
+ 命令行用 `--page details` （rule-based 汇总）或 `--page raw` (原始 metric) 。


C++ 模板 kernel 的完整名带一长串后缀（如 `persist_d_ws_fwd_cute_fp8_sm120`，实际实例化名比这更长）。 `--kernel-name` 默认精确匹配，拿名字前缀去匹配会报 `No kernels were profiled` 且生成空 report。必须加 `regex:` 前缀走部分匹配：

```shell
ncu --set full -o rep --kernel-name regex:persist_d_ws_fwd_cute_fp8 \
	--target-processes all ./app 

ncu --import rep.ncu-rep --page raw 2>&1 | \
	grep -A3 "warps_issue_stalled_\(wait\|math_pipe_throttle\|sleeping\)"
```


==🟣**权限限制与同架构代理 GPU**==

读性能计数器需要驱动授权。容器/云主机上常见 `/proc/driver/nvidia/params` 里 `RmProfilingAdminOnly` 标志为 1，此时 `--set full` 直接报 `ERR_NVGPUCTRPERM`，且容器内法驱动 reload 驱动。可行解法是同架构代理：找一块相同微架构的本地 GPU 跑 ncu，例如用本地 RTX PRO 5000（sm_120a）代理远端 RTX 5090（sm_120）。stall reasons、指令 mix、occupancy 这类架构级结论可以迁移，绝对性能数字（SM数、频率、L2 容量）不可迁移；ncu 验证完架构结论后，回目标硬件跑 timing bench 收数。


==🟣**replay 的干扰与 A/B 规则**==

ncu 锁频 + 多遍回放 + 串行执行，因此报告内耗时只用于相对比较。做优化 A/B 时回到真实 bench，并遵守两条规则：

+ **校验公共参照：** 许多 bench 会同时输出被测实现与公共参照两组时间；对比前先核对两轮公共参照是否吻合（±1%）。被 GPU 级干扰（并发进程/时钟态）污染的一轮，两者会同比例膨胀（实测约 2×），直接对比会得出 -50% 全面加速这种物理上不可能的假结论。
+ **min-of-N 口径：**  单轮抖动大的实现取 N 次最小值对比；归因 race 类问题必须 stash -> 重建 -> 同 probe 的 HEAD 对照，单次好结果不可信。


