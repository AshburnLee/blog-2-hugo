+++
date = '2025-09-16T11:11:34+08:00'
draft = false
title = 'Nsight Compute & Nsight Systems'
tags = ["CUDA", "Nsight Compute"]
categories = ["CUDA"]
+++

## Jetson 上可用工具

- ncu
- ncu-ui
- nsys
- nsys-ui
- cuda-gdb
- cuda-gdbserver


## 常用命令

~~~sh
--mode=launch-and-attach：启动程序并附加分析。
--kernel <kernel_name>：指定要分析的 CUDA 内核（可选，正则表达式匹配）。
--set <set_name>：指定指标集（通过 `ncu --list-sets`）。
  - basic
  - full: 收集所有可用指标，适合全面分析，但耗时较长。
  - detailed: 收集详细指标，覆盖计算、内存、调度等，适合深入优化。
  - nvlink 
  - pmsampling
  - roofline
# 指定要收集的指标, 哪些指标可选呢？`ncu --query-metrics` 给出所有可选的metrics
# 逗号分隔多个指标
ncu --query-metrics  # 输出所有支持的指标 metric 和描述

ncu --mode=launch-and-attach --metric dram__cycles_active.avg  
ncu --mode=launch ./my_cuda_app                # nvcc 编译得到的
ncu --mode=launch-and-attach python kernel.py  # python 程序
~~~


## 常用组合

~~~sh
sudo $(which ncu) --query-metrics > ./ncu-query-metrics.txt  # 输出所有支持的指标 metric 和描述
# 指定分析 kernel
sudo $(which ncu) --mode=launch-and-attach --kernel-name <which-kernel> --metric sm__warps_launched ./obj/softmax
# 产出分析报告
sudo $(which ncu) --mode=launch-and-attach  -o ./ncu/softmax-last.ncu-rep ./obj/softmax
# 然后查看报告
sudo $(which ncu) --import ./ncu/softmax-last.ncu-rep --csv > ./ncu/softmax-last.txt
# 或在 gui上查看
ncu-ui profile_report.ncu-rep
~~~


## KAQ：是否能分析 binding 到 Python api 的程序？

detailed 适合初步分析
~~~sh
# 注意 这里的python 要指定你期望的python，否则是系统python
sudo $(which ncu) --target-processes all --set detailed --export profile_report.ncu-rep /home/junhui/miniforge3/envs/cuda-ops/bin/python test_softmax.py

# 使用 sudo 时，这些变量可能被重置。你可以通过 sudo -E 保留用户环境变量,但是下面返回都是系统python
sudo -E which python
sudo which python
~~~

然后深入关键 metrics ，【确认每一个指标名是否正确】
~~~sh
sudo -E $(which ncu) --metrics sm__cycles_elapsed.avg,mem__bytes_read.sum,dram__bytes_read.sum,sm__warps_active.avg.pct_of_peak_sustained_active --target-processes all --export profile_report.ncu-rep python test_softmax.py
~~~

最后查看报告，ncu-ui profile_report.ncu-rep


## KAQ：ncu-ui profile_report.ncu-rep 可以在linux上查看吗（linux纯命令行/linux桌面）？

需要GUI，纯命令下不可。


## KAQ：ncu 分析 kernel 时，会看到各个线程的行为、存的数值吗？

不行，它主要分析内核的性能指标，如内存访问、SM 利用率、指令吞吐量等，ncu 无法直接查看各个线程的行为或存储的数值。需用调试工具（如 `cuda-gdb`）查看线程行为和数值。


## KAQ： NVIDIA 有哪个工具可以查看各个线程或物理 SP 的行为、存的数值？一定有的，以前就用过一个图形界面的工具可以这样做

NVIDIA Nsight Visual Studio Edition (Nsight VSE) 或 Nsight Compute GUI 的 CUDA 调试器，支持图形界面查看线程行为（Warp 调度、寄存器值）和存储数值（变量/内存视图）。使用 CUDA Warp Watch 和 CUDA Info View 聚焦特定线程/SP。

Linux 上：Nsight for Linux (CLI/GUI)，或 `cuda-gdb` 结合 Visual Profiler。

所以需要一个 IDE，具备调试工具才行。


## KAQ：nsys/nsys UI 分析时，分析的是什么？

- 时间线视图：跟踪 CPU/GPU 活动，包括 CUDA API 调用、内核执行、内存传输。
- 线程/Warp 分析：显示线程块、Warp 调度，识别 SM 占用率和同步问题。
- 性能瓶颈：分析 CPU-GPU 交互、内存带宽、内核延迟。
- 不支持寄存器值：无法直接查看线程/SP 寄存器或内存值（需用 cuda-gdb）。

~~~sh
# 收集数据
nsys profile --trace=cuda,osrt,nvtx --output profile.nsys-rep python test_softmax.py
# 在gui中查看
nsys-ui profile.nsys-rep
# 线程调试
cuda-gdb /home/user/miniconda3/envs/your_env/bin/python
(cuda-gdb) break my_softmax_kernel
(cuda-gdb) run test_softmax.py
~~~

## KAQ：什么情况下需要通过 nsys 分析?
## KAQ：什么情况下需要通过 ncu 分析?

## KAQ：ncu 分析时，一般讲，重点分析哪些指标 ？

重点分析的指标，【原来我的设备是： Device Orin (GA10B)】

1. SM 利用率：

   - 指标：
      - `sm__cycles_elapsed.avg`（SM 时钟周期） # of cycles elapsed on SM 即SM 运行的时钟周期总数
      - `sm__inst_executed.avg`（执行指令数）。  # of warp instructions executed 即每个warp执行的指令数
   - 意义：衡量 SM 的活跃程度，低值可能表示线程块不足或发散问题。**值越大越好**
   - 优化：调整线程块大小、网格配置，减少线程发散。


2. 内存带宽和访问效率：

   - mcc：mcc是 Memory Controller Channel（内存控制通道），负责管理访问显存（DRAM）的请求
   - 指标：
     - `mcc__dram_throughput_op_read.sum`      DRAM 读吞吐量     越大越好
     - `mcc__dram_throughput_op_write.sum`     DRAM 写吞吐量     越大越好
     - `l1tex__t_bytes.avg`                    L1 缓存命中率     越大越好
     - `lts__t_request_hit_rate`               L2 缓存命中率     越大越好
     - `sm__sass_data_bytes_mem_global_op_ld`  sm 级别的全局内存加载字节数总和   本身没有绝对好坏
     - `sm__sass_data_bytes_mem_global_op_st`  sm 级别的全局内存存储字节数总和   本身没有绝对好坏

   - 意义：识别内存瓶颈，高 DRAM 访问或低缓存命中率表明内存访问模式低效。
   - 优化：优化内存合并访问、使用共享内存、调整数据布局。


3. 指令吞吐量：

   - 指标：
      - sm__inst_issued.avg（每周期发出指令数）、
      - sm__inst_executed_pipe_*.avg（特定流水线吞吐，如 pipe_tensor 或 pipe_fma）。
   - 意义：检查指令执行效率，低吞吐量可能因寄存器压力或指令依赖。
   - 优化：减少复杂指令、优化寄存器分配。


4. 占用率（Occupancy）：

   - 指标：`sm__warps_active.avg`  active 的warp数，越多表示GPU利用率越高，有利于隐藏内存及指令延迟
   - 意义：高占用率表明 SM 资源利用充分；低值可能因寄存器或共享内存过高。
   - 优化：调整线程数、减少寄存器使用（-maxrregcount）。

    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        17.34
    Achieved Active Warps Per SM           warp         8.32


5. 延迟原因：

   - 指标：【没找到】
      - sm__cycles_stalled.avg（SM 停顿周期）、
      - sm__warps_stalled_*.avg（具体停顿原因，如 memory_dependency 或 execution_dependency）。
   - 意义：识别内核停顿的主因（如内存延迟或指令依赖）。
   - 优化：减少全局内存访问、优化分支逻辑。



## KAQ：我的 kernel 有关键指标的数值，但是我如很判断这个值是好是坏呢？是否应该有一个基线 baseline？


# KAQ：对于我的cuda kernel，如何有效利用 ncu/ncu-ui 和 nsys/nsys-ui 完全分析明白？给出一步一步的指导

【根据我的 kernel 干起来】

~~~sh
sudo $(which ncu) --target-processes all --set detailed --export profile_report.ncu-rep /home/junhui/miniforge3/envs/cuda-ops/bin/python test_softmax.py
ncu --import profile_report.ncu-rep --csv > profile_report.ncu.csv
~~~

指标太多，哪些是关键？

占用率：

~~~sh
sudo $(which ncu) --metrics sm__warps_active.avg.pct_of_peak_sustained_active --target-processes all --export profile_report.warp-active.ncu-rep /home/junhui/miniforge3/envs/cuda-ops/bin/python ../tests/test_softmax.py
ncu --import profile_report.warp-active.ncu-rep > profile_report.warp-active.ncu.txt
~~~

结果：

~~~txt
    ------------------------------------------------- ----------- ------------
    Metric Name                                       Metric Unit Metric Value
    ------------------------------------------------- ----------- ------------
    sm__warps_active.avg.pct_of_peak_sustained_active           %        16.66
    ------------------------------------------------- ----------- ------------
~~~


## KAQ: 有哪些 --set 可选

~~~sh
ncu --list-sets

sudo $(which ncu) --target-processes all --set full --export profile_report.full.ncu-rep /home/junhui/miniforge3/envs/cuda-ops/bin/python ../tests-v3/test_softmax.py && ncu --import profile_report.full.ncu-rep > profile_report.full.ncu.txt


sudo $(which ncu) --target-processes all --set basic --export profile_report.basic.ncu-rep /home/junhui/miniforge3/envs/cuda-ops/bin/python ../tests-v5/test_softmax.py && ncu --import profile_report.basic.ncu-rep > profile_report.basic.ncu.txt


sudo $(which ncu) --target-processes all --set roofline --export profile_report.roofline.ncu-rep /home/junhui/miniforge3/envs/cuda-ops/bin/python ../tests-v3/test_softmax.py && ncu --import profile_report.roofline.ncu-rep > profile_report.roofline.ncu.txt
~~~


## KAQ：report 信息很多，如何解读？

阅读/分析 basic report 就可以了，报告含有3类内容，1. 统计数值，2. 通过统计数值间接得到的指标，3. 优化建议 OPT。

所以就当前 kernel 的状态，根据 report 上反应出的主要矛盾，进行优化。


## 主要矛盾【优先考虑主要矛盾的，这里内存访问时瓶颈】

回顾优化高优先级:

- 寻找**并行化**顺序代码的方法。
- 最小化主机与设备之间的**数据传输**。
- 调整内核启动配置以**最大化设备利用率**。
- 确保全局内存访问是 **coalesced** 的。
- 尽可能减少对**全局内存**的冗余访问。
- 尽量量避免同一 Warp 中线程执行路径出现长时间的**分支**(sequences of diverged execution)。

应该优先解决主要矛盾, 我的case 内存访问时瓶颈，优化 reduce 计算。

1. 需要改变 kernel 启动配置：多个 SM 没有活动，Achieved Active Warps Per SM = 7.99 （理论是48），占用率（Achieved Occupancy）只有 16.66。，这是 workload 不平衡的结果。

2. 内存访问是瓶颈：（Memory Throughput = 4.89%，L1/TEX Cache Throughput = 5.44%，高全局内存访问


### 1. 版本1中的线程配置：【目的是调整内核启动配置以最大化设备利用率】

(1,1,1),(256,1,1) => (4,1,1),(64,1,1)

~~~cpp
    // 配置CUDA核参数 block数 (1,1,1), thread数 (256,1,1)
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
~~~

优化：减小 thread 的大小，将线程分配给其他 block。

~~~cpp
    // 配置CUDA核参数 block数 (4,1,1), thread数 (64,1,1)
    int threadsPerBlock = 64;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
~~~

> 负载平和和资源利用率：

- SM Active Cycles （同 SMSP Active Cycles）显著增加（从 5,334.50 增加到 18,937），更多的 block 分别了更多的SM。但 block 数还是少。
- Waves Per SM 变化（0.04 -> 0.06）并行程度几乎没有提升：全局内存访问高，Warp 停顿多，内存是瓶颈限制了吞吐和并行性 
- Compute (SM) Throughput 增幅非常小，因为数据量太小（256个float）
- Achieved Active Warps Per SM: 7.99 -> 2.00：SM 没有充分利用。
- Theoretical Active Warps per SM = 32：这个case 中一个 SM 同时运行32个warps。理论上4 SM同时执行128个warps

> 内存吞吐量：

- Memory Throughput（4.48%→4.89%）：低内存吞吐量，表明内存访问效率低。与
- L1/TEX Cache Throughput（19.31%→5.44%）：低 L1 缓存命中率，暗示频繁访问全局内存。
- L2 Cache Throughput (0.49%->0.67%)：低 L2 命中率，进一步确认全局内存访问主导。


#### KAQ：runtime 时数据是如何映射到硬件的！实际映射是有调度器决定的,如何直接得出？

根据信息
~~~sh
Grid Size=4
Block Size=64   # 每个 block 有 64 线程。1 Warp = 32 线程，因此每个 block 有 2 Warp，4个block 就有 8 个warps
Threads = 256
SMs = 4
Block Limit SM = 16  # 示每个 SM 的硬件限制，即每个 SM 最多可同时运行 16 个block，实际上只有1个。
Theoretical Active Warps per SM = 32
Achieved Active Warps Per SM = 2.00   # 8warps 除以 2 = 4 个 SM，表示8个warps 平均分配在4个SM上。
~~~

4 block 仅 8 Warp，每个 SM 约 2 Warp，每个 SM 上 1 个block，故分摊到 4 SM

`sm__ctas_launched` 确认每个SM launch的 block 数量
`sm__warps_launched` 每个 sm 运行了多少个 warp
`smsp__warps_launched` SMSP 是 SM 的子单元，负责线程调度, warp launched 个数
`smsp__average_threads_launched_per_warp` 每个 warp 平均有多少个线程

~~~sh
sudo $(which ncu) --metrics sm__warps_launched,sm__ctas_launched,sm__warps_active.avg.pct_of_peak_sustained_active --target-processes all --export profile_report.cts.ncu-rep /home/junhui/miniforge3/envs/cuda-ops/bin/python ../tests-v5/test_softmax.py  && ncu --import profile_report.cts.ncu-rep > profile_report.cts.ncu.txt
~~~

结果：1个SM 只运行了1个block，一个SM运行了2个warps。所以数据如何映射的？**4 block 分摊给 4 个 SM，每个 SM 一个 block 即 2 个 Warp**


### 2. 使用标准的reduce计算，kernel启动配置不变

改用 shared memory d reduce版本。这个主要是更改了计算逻辑，为了减少全局内存访问，但 kernel 启动配置不变。

Launch 配置没有变化，故 `Launch statistics` 和 `Occupancy` 没有变化。

- Duration (80.06us -> 22.82us)

> 负载平和和资源利用率：
- SM Active Cycles  （5,334 -> 952）
- Waves Per SM 变化（0.04 -> 0.04）
- Compute (SM) Throughput (8.81 -> 2.58): 可能是同步开销 ?
- Achieved Active Warps Per SM: (7.99 -> 7.97)：
- Theoretical Active Warps per SM (32 -> 32)：

> 内存吞吐量：
- Memory Throughput（4.48% -> 2.29%）： size=256 数据量太小，不足以体现共享内存优势， __syncthreads() 的延迟被放大
- L1/TEX Cache Throughput（19.31% -> 13.39%）: 可能因共享内存访问取代部分 L1 缓存访问
- L2 Cache Throughput (0.49% -> 1.29%)

- Average SM Active Cycles（5,334.50 → 952.25）：SM 和 L1 缓存活跃时间大幅减少，表明内核执行时间缩短。
- Total SM Elapsed Cycles（91,900 → 22,268）：总执行周期大幅减少，表明内核运行更快
- Total L2 Elapsed Cycles （97,796 → 27,704）：L2 总周期减少，确认全局内存访问减少
- Total SMSP Elapsed Cycles（367,600 → 89,072）：SMSP 总周期减少，反映整体执行效率提高
- 计算得 SM 资源利用率：952.25 x 4 / 22,268 = 38.09%，远小于100%。

`__syncthreads()`，增加同步开销，抵消部分内存效率提升。

16*26=

### 3. 优化 kernel 逻辑、kernel Launch 配置

~~~cpp
    int threads_per_block = 64;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
~~~

Active Cycles 和  Elapsed Cycles 都显著减小，反应了kernel整体执行时间减少，对全局内存访问减少。

> 应该使用尽量大的数据量，才能明显反应出各个指标随优化进行的变化
> kernel 中对global的访问不是连续的
> kernel中对shared memory 访问是否有conflict

### 4. size = 4096

(32, 1, 1)x(128, 1, 1)

数据量大后，占用率增加，
~~~sh
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        61.19
    Achieved Active Warps Per SM           warp        29.37
~~~

数据到 SM 的映射是均匀的：

~~~sh
Grid Size=32
Block Size=128   # 每个 block 有 64 线程。1 Warp = 32 线程，因此每个 block 有 2 Warp，4个block 就有 8 个warps
Threads = 4096
SMs = 4
Block Limit SM = 16  # 示每个 SM 的硬件限制，即每个 SM 最多可同时运行 16 个block，实际上只有 8 个。
Theoretical Active Warps per SM = 32
Achieved Active Warps Per SM = 29.37  # 8 warps 除以 2 = 4 个 SM，表示8个warps 平均分配在4个SM上。

sm__ctas_launched.avg  8 ：每个 SM 运行了 8 个block
sm__warps_launched.avg  32 ：每个 SM 运行了 32 个warps
~~~

分析两个指标：`Average SM Active Cycles` = 26,624.75 和 `Total SM Elapsed Cycles` = 114,080。我有4个SM，所以：

- Total SM Active Cycles =26,624.75 × 4 SM = 106,499
- 而 Total SM Elapsed Cycles = 114,080。
- 106,499 / 114,080 = 93.35% 的周期 SM 处于活跃状态，表明 SM 利用率很高

这是个好的**资源利用率**的指标。

