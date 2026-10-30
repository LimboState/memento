Start Date: 20261029
Author: Calcifer


## 1. Architecture

XiangJiang NPU is a hardware accelerator for ML inferencing.

XiangJiang fetches and decodes instructions, dispatches them to issue queues, issues them for execution, and records their completion before retiring them in program order through the reorder buffer (ROB).
`Fetch -> Decode -> Dispatch -> RS/Issue -> Execute -> Complete to ROB -> Retire/Writeback`

XiangJiang NPU offers the following top-level feature set:
+ RISC-V instruction set (specifically `rv32imf_zve32x_zicsr_zifencei_zbb`);
+ In-order dispatch, out-of-order execution/completion, in-order commit through a ROB;
+ Single-wide unified scalar/vector/matrix dispatch;
+ 256 KiB ITCM memory (tightly-coupled memory for instructions);
+ 1 MiB DTCM memory (tightly-coupled memory for data), organized into four 256 KiB banks;
+ Vector Function Units (VFUs) use 128-bits SIMD datapaths, supporting shorter vectors through partial element activation and wider vectors by decomposing them into multiple 128-bit uops (micro-operation).
+ Both memories are single-cycle-latency TCM SRAM, more efficient than Cache memory;
+ AXI4 bus interfaces, functioning as both manager and subordinate, to interact with external memory and allow external CPUs to configure Coral NPU;



## 2. Instruction Set

| Component  | Declared Capability                                                                                               | Version |                                                Docs Link                                                |
| ---------- | ----------------------------------------------------------------------------------------------------------------- | :-----: | :-----------------------------------------------------------------------------------------------------: |
| `RV32`     | `XLEN = 32`, integer registers and basic address calculations are 32 bits wide.                                   |    -    |        [RV32I Base Integer ISA](https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html)        |
| `I`        | Base integer arithmetic, load/store operations, control flow, and environment instructions.                       |   2.1   |                [RV32I](https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html)                 |
| `M`        | Integer multiplication, division, and remainder operations.                                                       |   2.0   |           [M Extension](https://docs.riscv.org/reference/isa/v20260120/unpriv/m-st-ext.html)            |
| `F`        | Scalar `IEEE 754 binary32 (FP32)` floating-point operations with `FLEN = 32`.                                     |   2.2   |           [F Extension](https://docs.riscv.org/reference/isa/v20260120/unpriv/f-st-ext.html)            |
| `Zve32x`   | Embedded integer vector operations with `SEW = 8, 16, or 32`, vector floating-point operations are not supported. |   1.0   | [Vector Extensions - Zve Subsets](https://docs.riscv.org/reference/isa/v20260120/unpriv/v-st-ext.html)  |
| `Zicsr`    | Atomic read-modify-write operations on Control and Status Registers (`CSRs`).                                     |   2.0   |           [Zicsr Extension](https://docs.riscv.org/reference/isa/v20260120/unpriv/zicsr.html)           |
| `Zifencei` | Provides FENCE.I to synchronize data writes with subsequent instruction fetches.                                  |   2.0   |        [Zifencei Extention](https://docs.riscv.org/reference/isa/v20260120/unpriv/zifencei.html)        |
| `Zbb`      | Basic scalar integer bit-manipulation operations.                                                                 |   1.0   | [Bit-Manipulation Extension - Zbb](https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html) |




## 3. Register Files

### 3.1. Scalar Register Files

+ Scalar Integer Register Files (`x0 ~ x31`): 32 bits wide by default (`XLEN = 32`), primarily used for integer data, addresses, and loop control.
+ Scalar Floating-Point Register File (`f0 ~ f31`): Each register stores a 32-bit FP32 value (`FLEN = 32`), primarily used for scalar floating-point computation and for providing scalar floating-point operands to the vector processing unit.


### 3.2. Vector Register Files

+ Vector Register Files (`v0 ~ v31`): Each register is `VLEN` bits wide, with a total capacity of `32 × VLEN` bits. `VLEN` is a hardware parameter and defaults to 128 bits. The register file primarily provides vector data to the vector processing unit.


### 3.3. Matrix Skew Buffer

+ Matrix Integer Skew Buffers (`M0 ~ M1`): Each buffer stores an `8 × 8` matrix in a skewed layout, with each element represented as an 8-bit `INT8` value. They provide matrix tiles to the Matrix Integer Functional Units.
+ Matrix Float-Point Skew Buffers (`M2 ~ M5`): Each buffer stores an `8 × 8` matrix in a skewed layout, with each element represented as a 16-bit `BF16` value. They provide matrix tiles to the Matrix Floating-Point Functional Units.


## 4. Functional Units

### 4.1. Scalar Functional Units

| Functional Unit | Quantity | Supported Operations                                                                                                                 |
| --------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Integer ALU     | 1        | Integer add/sub, comparison, logical operations, shifts, and `Zbb` bit-manipulation operations.                                      |
| Branch Unit     | 1        | Branches, `JAL/JALR`, and exception control.                                                                                         |
| MLU             | 1        | Integer multiplication: `MUL`, `MULH`, `MULHU`, and `MULHSU`.                                                                        |
| DVU             | 1        | Integer division and remainder: `DIV`, `DIVU`, `REM`, and `REMU`.                                                                    |
| FloatCore       | 1        | Floating-point addition, subtraction, multiplication, fused multiply-add (`FMA`), division, square root, comparison, and conversion. |
| LSU             | 1        | Scalar, vector, and matrix load/store operations.                                                                                    |
| CSR             | 1        | Control and Status Register (CSR) operations.                                                                                        |


The `ADD` and `SUB` operations naturally share the same hardware adder, as subtraction can be implemented by inverting one operand and setting the carry-in. Multiplication and division, however, have significantly different datapaths, latencies, and pipeline structures, so they are implemented in separate functional units.

Scalar floating-point operations are exposed through a single **FloatCore**,  which is further divided into internal sub-units. Floating-point `add`, `mul`, and `FMA` (Fused Multiply-Add) share substantial datapath resources and are therefore integrated into one floating-point compute core, however, `div/sqrt` are handled by dedicated units.


### 4.2. Vector Functional Units

Vector configuration instructions define the execution context used by subsequent vector instructions. They belong to the `OPCFG` instruction class, with `funct3 = 111`, and include `vsetvli`, `vsetivli`, and `vsetvl`.

These instructions are handled directly by the configuration-state update logic in the RVV front end. They are not dispatched to vector functional units and are not decomposed into uops.

| state  | Description                                                     |
| ------ | --------------------------------------------------------------- |
| vl     | Number of vector elements to execute.                           |
| sew    | Selected element width, such as 8, 16, or 32 bits.              |
| lmul   | Vector register-group multiplier.                               |
| ta     | Tail-element policy: agnostic or undisturbed.                   |
| ma     | Masked-off element policy: agnostic or undisturbed.             |
| vill   | Indicates whether the current `vtype` configuration is illegal. |
| vstart | Reset to zero after a vector configuration instruction.         |


 Vector Instruction are decomposed into micro-operations (uops), which are then dispatched to different reservation stations (RS) and execution units (FU) according to their `uop_exe_unit` classification.

| Vector Functional Unit | Quantity | `uop_exe_unit`                                             | Supported Operations                                                                                                                                                                                                  |
| ---------------------- | -------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VALU                   | 1        | `VEU_ALU`, `VEU_CMP`                                       | Integer add/sub, widening and averaging add/subtract, saturating arithmetic, logical operations, shifts, narrowing/clip, min/max, merge/move, integer comparisons, and mask operations.                               |
| VMUL/MAC               | 1        | `VEU_MUL`, `VEU_MAC`                                       | Integer multiply, high-half multiply, widening multiply,  multiply-add, multiply-accumulate, widening MAC, and fixed-point multiply with rounding/saturation.                                                         |
| VDIV                   | 1        | `VEU_DIV`                                                  | Signed/unsigned integer division and remainder.                                                                                                                                                                       |
| VFALU                  | 1        | `VEU_FMA`, `VEU_FCVT`, `VEU_FNCMP`, `VEU_FCMP`, `VEU_FTBL` | FP add/subtract/multiply, fused multiply-add/accumulate, min/max, sign injection, comparisons, classification, integer/FP conversions, reciprocal/reciprocal-square-root estimates, and optional BF16 conversion/MAC. |
| VFDIV/SQRT             | 1        | `VEU_FDIV`                                                 | Floating-point division, reverse division, and square root.                                                                                                                                                           |
| PMTRDT                 | 1        | `VEU_PMT`, `VEU_RDT`, `VEU_FRDT`, `VEU_MISC`               | Vector gather, slide, compress, integer/FP reductions, widening reductions, population count, and index generation.                                                                                                   |
| Vector LSU Path        | 1        | `LSU`                                                      | Unit-stride, constant-stride, ordered/unordered indexed, segmented, whole-register, mask, and fault-only-first vector loads/stores.                                                                                   |
 The `uop_exe_unit` classes are:
 - `VEU_ALU`: General integer vector arithmetic and logical operations.
 - `VEU_CMP`: Integer comparisons and mask-result generation.
 - `VEU_MUL/VEU_MAC`: Integer multiplication and multiply-accumulate operations.
 - `VEU_DIV`: Integer division and remainder operations.
 - `VEU_FMA`: Floating-point addition, subtraction, multiplication, and fused multiply-add operations.
 - `VEU_FCVT`: Integer and floating-point conversions.
 - `VEU_FNCMP`: Floating-point non-mask operations such as min/max, sign injection, and classification.
 - `VEU_FCMP`: Floating-point comparisons that produce vector mask results.
 - `VEU_FTBL`: Table-based floating-point approximations, such as `vfrec7` and `vfrsqrt7`.
 - `VEU_FDIV`: Floating-point division and square-root operations.
 - `VEU_PMT`: Permutation, slide, gather, and compress operations.
 - `VEU_RDT/VEU_FRDT`: Integer and floating-point reduction operations.
 - `VEU_MISC`: Special operations such as population count and index generation.
 - `LSU`: Vector load and store operations.


### 4.3. Matrix Functional Units

 矩阵输入首先由 Matrix LSU 从 DTCM 装载到 Matrix Skew Buffer。执行时，矩阵 A 的元素沿脉动阵列水平方向传播，矩阵 B 的元素
 沿垂直方向传播，使相同 K 维位置的操作数在对应 PE 中对齐。计算结果保存在阵列内部的累加器中，不直接写入 Scalar、
 Floating-Point 或 Vector Register File，必须通过矩阵存储指令写回 DTCM。

 每个 Matrix Functional Unit 包含一个 8 × 8 脉动阵列，共有 64 个 Processing Elements（PE）.

| Matrix Functional Unit     | Quantity | Input Skew Buffer      | Data Type    | Dataflow | Supported Operations                                                                                           |
| -------------------------- | -------- | ---------------------- | ------------ | -------- | -------------------------------------------------------------------------------------------------------------- |
| Matrix Floating-Point Unit | 1        | `M0(A)`, `M1(B)`       | BF16         | OS       | **BF16** matrix multiplication, matrix multiply-accumulate, element-wise multiplication, and matrix transpose. |
| Matrix Integer             | 2        | `M2,M4(A)`, `M3,M5(B)` | INT8(Signed) | OS/WS    | Matrix multiplication, matrix multiply-accumulate, element-wise multiplication, and matrix transpose.          |
| Matrix LSU Path            | 1        | `M0 ~ M5`              | INT8/BF16    | -        | Transfers complete matrix tiles between DTCM, Matrix Skew Buffers, and internal accumulators.                  |

Matrix Integer Unit 对两个有符号 INT8 输入执行乘累加：
```chisel
productINT16 = INT8(A) × INT8(B)
ACCINT32     = ACCINT32 + sign_extend(productINT16)
```
 - INT8 与 INT8 相乘产生完整的有符号 INT16 乘积。
 - INT16 乘积在累加前符号扩展为 INT32。
 - 每个 PE 使用独立的 INT32 累加器保存部分和。
 - MMUL 在计算开始前清零累加器。
 - MMACC 保留已有累加器内容，并继续执行矩阵乘累加。
 - INT8 计算结果以 INT32 格式输出，不自动饱和或窄化为 INT8。
 - INT32 累加溢出按二进制补码回绕处理，除非后续架构版本另行定义饱和模式。



Matrix Floating-Point Unit 接收 BF16 输入，并使用 FP32 累加器保存部分和。每次 PE 更新的数值语义为：
```chisel
productBF16  = RNE_BF16(A × B)
ACCFP32      = RNE_FP32(ACCFP32 + widen_FP32(productBF16))
```
其中：

 - BF16 乘积采用 Round to Nearest, Ties to Even（RNE）舍入。
 - BF16 乘积随后精确拓宽为 FP32，并与 FP32 部分和相加。
 - 该过程不是 BF16 乘法与 FP32 加法的融合 FMA；乘积会在累加前先舍入为 BF16。
 - MMUL 清零 FP32 累加器后开始计算。
 - MMACC 保留已有 FP32 部分和，可用于跨多个 K 维 tile 继续累加。
 - 结果写回 DTCM 时，将 FP32 累加结果以 RNE 转换为 BF16。
 - BF16 乘法路径可以使用乘积寄存器形成流水线，使阵列保持每周期接收一组有效输入，而不要求每两周期才能启动一次乘累加。



| Operation   | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `MSETMODE`  | 设置 INT8 Matrix Unit 的 OS 或 WS 数据流模式。BF16 Matrix Unit 只允许 OS。 |
| `MLOAD.A`   | 从 DTCM 装载一个完整的 A tile 到指定矩阵单元的 A Skew Buffer。                |
| `MLOAD.B`   | 从 DTCM 装载一个完整的 B tile 到指定矩阵单元的 B Skew Buffer。                |
| `MLOAD.ACC` | 从 DTCM 装载已有结果作为内部累加器的初始值。                                    |
| `MMUL`      | 清零内部累加器并执行 ACC = A × B。                                      |
| `MMACC`     | 保留内部累加器并执行 ACC = ACC + A × B。                                |
| `MEW`       | 执行逐元素乘法 ACC[i][j] = A[i][j] × B[i][j]。                       |
| `MTRANS`    | 执行矩阵转置 ACC = Aᵀ。                                             |
| `MSTORE`    | 将内部累加器转换为对应输出格式并写回 DTCM。                                     |
| `MWAIT`     | 等待指定矩阵阵列的当前任务完成。                                             |
| `MBARRIER`  | 等待此前的矩阵或相关内存任务完成。                                            |
| `MFENCE`    | 建立矩阵访存顺序屏障，保证此前矩阵读写已经完成并可被后续操作观察。                            |
矩阵配置指令由配置状态更新逻辑处理，不进入 PE 阵列。MLOAD 和 MSTORE 通过 Matrix LSU 执行；MMUL、MMACC、MEW 和 MTRANS 由矩阵任务控制器发送到选定阵列。



## 5. Architecture Overview

XiangJiang NPU 采用:
```text
Fetch
-> Decode
-> Dispatch / Rename
-> Reservation Station
-> Out-of-Order Issue
-> Execute
-> Complete / Write Result to ROB
-> In-Order Retire
-> Architectural Register Writeback
```

| Stage                          | 含义                                       | 是否修改架构状态 |
| ------------------------------ | ---------------------------------------- | -------- |
| Execution Writeback / Complete | 功能单元产生结果，将结果写入 ROB、结果槽或物理寄存器，并唤醒依赖者.     | 否        |
| Retirement Writeback / Commit  | ROB 队首指令确认无异常并满足提交条件，将结果写入架构寄存器或执行架构副作用. | 是        |


### 5.1. Structure

XiangJiang NPU 将寄存器状态分为两部分：

 1. 已经提交的状态：程序正式可见，即使流水线被清空也必须保留；
 2. 尚未提交的状态：由正在执行的指令产生，可以被后续指令提前使用，但发生异常或 Flush 时可能被丢弃。

 为管理这两类状态，后端使用 ROB、RAT、ARF 和 PRF 四种核心结构：

 - ARF（Architectural Register File，架构寄存器文件）
   保存已经提交的寄存器值。ARF 代表当前精确的程序状态，不会因为取消未提交指令而回滚。

 - PRF（Physical Register File，物理寄存器文件）
   临时保存正在执行或已经完成、但尚未提交的指令结果。这些结果可以提前提供给后续依赖指令，但在提交前仍属于推测状态。

 - RAT（Register Alias Table，寄存器映射表）
   记录每个架构寄存器当前是否存在一个更新但尚未提交的值，以及该值存放在哪个 PRF Entry 中。
   如果不存在未提交的新值，指令从 ARF 读取；如果存在，则从 RAT 指向的 PRF Entry 读取，或者等待其生产者完成。

- ROB（Reorder Buffer，重排序缓冲区）
   按程序顺序记录所有在途指令，并跟踪其执行完成、异常和副作用状态。功能单元可以乱序完成，但只有位于 ROB 队首且满足提交
   条件的指令才能更新 ARF。

 一条写寄存器指令的处理过程如下：

 ```text
   Dispatch：为结果分配 PRF Entry，并由 RAT 记录新位置
   Execute：功能单元执行指令
   Complete：结果写入 PRF，并唤醒依赖该结果的指令
   Commit：指令到达 ROB 队首后，将 PRF 结果写入 ARF
 ```




#### 5.1.1. ROB Entry
ROB 按程序顺序分配和释放条目。每个 ROB Entry 对应一条动态架构指令。

如果一条向量指令被拆分为多个 uop，这些 uop 共享同一个架构 ROB Entry，并通过 uop 完成计数或位图跟踪整条指令的完成状态。

ROB 通常实现为环形队列, ROB 槽位编号不能直接用于比较指令年龄，因为环形队列会发生回绕。年龄比较应使用单调递增的 `seq`.

每条动态指令在分配 ROB Entry 时获得一个 `ROBTag = {rob_slot, generation}` 。其中，`rob_slot` 用于定位指令当前占用的 ROB 槽位的编号，`generation` 用于标识该槽位的本次分配。槽位被释放并重新分配时，`generation` 随之更新。


```text
ROB Entry
  ├── Architectural Destination
  ├── Physical Destination
  ├── Producer Tag
  ├── Completion State
  ├── Exception State
  ├── Memory/Side-Effect State
  └── In-Order Retirement State
```

| Field                       | Type / Width             | Description                                                   |
| --------------------------- | ------------------------ | ------------------------------------------------------------- |
| `valid`                     | 1 bit                    | 表示该 ROB Entry 是否已被有效动态指令占用。                                   |
| `generation`                | Parameterized            | ROB 槽位复用代数。                                                   |
| `tag`                       | `{rob_slot, generation}` | 动态指令唯一标识，用于结果广播、依赖唤醒、异常归属和完成事件匹配。                             |
| `seq`                       | Parameterized            | 动态指令序号，用于比较指令年龄。                                              |
| `pc`                        | 32 bits                  | 当前指令的 PC，用于异常、调试和控制转移处理。                                      |
| `inst`                      | 32 bits                  | 原始指令编码。                                                       |
| `instruction_class`         | Enum                     | Integer、Float、Vector、Load、Store、Branch、CSR、Matrix、System 等类别。 |
| `operation`                 | Enum                     | 已译码的具体操作，例如 `ADD`、`FDIV`、`VADD`、`MLOAD` 或 `MMUL`。             |
| `destination_type`          | Enum                     | `NONE`、`X`、`F`、`V`、CSR 或其他配置状态。                               |
| `architectural_destination` | 5 bits or parameterized  | 架构目标编号，例如 `x5`、`f8` 或 `v16`。无寄存器目标时忽略。                        |
| `physical_destination`      | PRF index                | 为目标分配的 PRF Entry。无寄存器结果或目标为 `x0` 时无效。                         |
| `writes_destination`        | 1 bit                    | 指令是否写普通架构寄存器。                                                 |
| `execution_complete`        | 1 bit                    | 功能单元是否已经完成该指令的计算。                                             |
| `result_ready`              | 1 bit                    | 目标 PRF Entry 中是否已经存在有效结果。无寄存器结果时通常忽略。                         |
| `uop_count`                 | Parameterized            | 架构指令包含的 uop 总数。非分解指令通常为 1。                                    |
| `uop_done_bitmap`           | Parameterized            | 记录各 uop 的完成状态。所有必要 uop 完成后，架构指令才可完成。                          |
| `exception_valid`           | 1 bit                    | 指令是否发生译码、执行、访存或写回异常。                                          |
| `exception_cause`           | Parameterized            | 非法指令、地址未对齐、访问错误等异常原因。                                         |
| `bad_address`               | 32 bits                  | 导致访存或控制转移异常的地址。                                               |
| `next_pc`                   | 32 bits                  | Branch、`JAL` 或 `JALR` 解析出的下一 PC。                              |
| `branch_taken`              | 1 bit                    | 条件分支是否采用。仅控制转移指令使用。                                           |
| `memory_reference`          | MemQ / LSQ index         | 对应的存储器队列条目。非访存指令时无效。                                          |
| `memory_complete`           | 1 bit                    | Load 数据已返回，或 Store/Vector/Matrix 访问已到达规定完成点。                  |
| `side_effect_authorized`    | 1 bit                    | Store、CSR、Matrix Store、DMA 等不可逆操作是否获得 ROB 队首授权。               |
| `side_effect_complete`      | 1 bit                    | 指令的全部架构副作用是否已经完成。                                             |
| `configuration_snapshot`    | Optional                 | 向量或矩阵指令使用的 `vl`、`sew`、`lmul` 或矩阵模式等配置快照。                      |
| `matrix_task_reference`     | Optional                 | 对应矩阵任务记录，用于跟踪阵列、Skew Buffer 和 accumulator。                    |
| `retire_ready`              | Derived                  | 由完成、结果、访存和副作用状态组合得到，通常不需要单独存储。                                |

#### 5.1.2 Register Alias Table

RAT 记录每个架构寄存器的最新未提交版本。不同架构寄存器命名空间分别维护独立的 RAT：

|RAT|Number of Entries|Architectural Namespace|Mapped Physical Resource|
|---|---|---|---|
|`RAT-X`|32|Scalar Integer Registers `x0–x31`|Integer PRF|
|`RAT-F`|32|Scalar Floating-Point Registers `f0–f31`|Floating-Point PRF|
|`RAT-V`|32|Vector Registers `v0–v31`|Vector PRF|
矩阵设计可以通过 Scoreboard 管理 Skew Buffer 与 accumulator，因此不要求实现通用`RAT-M`。

每个 RAT Entry 对应一个架构寄存器，建议包含以下字段：

| Field               | Type / Width | Description                                                             |
| ------------------- | ------------ | ----------------------------------------------------------------------- |
| `valid`             | 1 bit        | 该架构寄存器是否存在最新的未提交版本。`valid=0` 时从 ARF 读取；`valid=1` 时使用 RAT 指向的 PRF Entry。 |
| `physical_register` | PRF index    | 最新未提交版本所在的 PRF Entry。                                                   |
| `producer_tag`      | ROB Tag      | 创建该版本的动态指令标签，格式为 {rob_slot, generation}, 用于依赖跟踪和提交时条件清除。                |


**源寄存器查询流程**：对于每个源寄存器 rs，Rename 阶段执行以下过程：

| RAT State   | PRF State   | Source Resolution                 |
| ----------- | ----------- | --------------------------------- |
| `valid = 0` | 不查询         | 从 ARF 读取已提交值。                     |
| `valid = 1` | `ready = 1` | 从 RAT 指向的 PRF Entry 读取。           |
| `valid = 1` | `ready = 0` | 保存 PRF 编号和生产者 Tag，等待 Complete 广播。 |
| 源为 `x0`     | 不查询         | 直接返回零。                            |

RAT 只保存架构寄存器到最新未提交物理版本的映射，不保存寄存器数据、结果就绪状态或异常状态。寄存器数据和 ready 位保存在 PRF，指令完成和异常状态保存在 ROB。提交时只有当 RAT 的 producer_tag 和 physical_register 仍与当前提交指令匹配时，才能清除该映射。




#### 5.1.3. Architectural Register File




#### 5.1.4. Physical Register File












