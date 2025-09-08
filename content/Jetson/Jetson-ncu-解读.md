+++
date = '2025-08-31T12:47:01+08:00'
draft = false
title = 'Jetson Ncu 解读'
tags = ["Jetson", "Nsight Compute"]
categories = ["Jetson"]
+++




## 常用命令

~~~sh
--mode=launch-and-attach：启动程序并附加分析。
--kernel <kernel_name>：指定要分析的 CUDA 内核（可选，正则表达式匹配）。
--set <set_name>：指定指标集（例如，full、detailed 或自定义集）。
  - full: 收集所有可用指标，适合全面分析，但耗时较长。
  - detailed: 收集详细指标，覆盖计算、内存、调度等，适合深入优化。
  - speed-of-light: 聚焦高层次吞吐量指标（如你的 Speed Of Light Throughput 报告）。
  - occupancy: 聚焦占用率指标（如你的 Occupancy 报告）。
  - scheduler: 分析线程束调度状态。
  - memory: 分析内存访问和缓存性能。
--metric dram__cycles_active.avg

sudo $(which ncu) --query-metrics  # 输出所有支持的指标和描述
sudo $(which ncu) --mode=launch-and-attach --kernel-name ComputeLogSoftmaxForwardInWarp --metric sm__warps_launched ./obj/softmax
~~~

~~~sh
rm -f ./ncu/softmax-last.ncu-rep && sudo $(which ncu) --mode=launch-and-attach  -o ./ncu/softmax-last.ncu-rep ./obj/softmax && sudo $(which ncu) --import ./ncu/softmax-last.ncu-rep > ./ncu/softmax-last.txt
~~~

## ncu -o ./softmax-last.ncu-rep ./softmax

~~~sh
  void ComputeLogSoftmaxForwardInWarp<float, float, 4>(T1 *, const T1 *, int, int) (1024, 1, 1)x(4, 32, 1), Context 1, Stream 7, Device 0, CC 8.7
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    SM Frequency                    Mhz       305.41  # SM的运行频率：表示处于低功耗模式或未满载运行，由于散热或功耗限制）导致频率降低，
    Elapsed Cycles                cycle       32,515  # kernel 执行的总时钟周期数：越小越好，如果过高，优化 kernel 以减少指令数或提高并行性。
    Memory Throughput                 %        27.33  # 内存带宽利用率：内存访问的效率 / 硬件峰值。27.33% 远低于 60% 的推荐阈值，表明内存带宽利用率非常低。检查内存访问模式（使用 NCU 的内存统计部分，查看 Global Memory 访问的合并程度）。
    Duration                         us       106.46  # kernel 执行的总时间（微秒）： 32,515 cycles ÷ 305.41 MHz
    L1/TEX Cache Throughput           %        29.40  # L1 缓存和纹理缓存的带宽利用率：低于 60%，即缓存利用率低。数据访问模式不佳，未充分利用 L1 缓存。缓存命中率低（需要检查 NCU 的缓存命中率统计）。
    L2 Cache Throughput               %        22.36  # 远低于预期，表明 L2 缓存未被有效利用。检查 L2 缓存的命中率（NCU 的缓存统计部分）。
    SM Active Cycles              cycle    28,654.75  # SM 在执行计算任务时的活跃周期数：活跃周期占总周期的比例为 28,654.75 ÷ 32,515 ≈ 88.1% 挺好，但是。查看 NCU 的 Scheduler Statistics，检查线程束调度效率。否存在指令依赖或分支发散。
    Compute (SM) Throughput           %        57.48  # SM 的计算吞吐量：57.48% 低于 60%，计算资源未被充分利用。可能线程块（Block）或网格（Grid）配置不足，导致 SM 未满载。
    ----------------------- ----------- ------------

    ## 优化建议：低计算吞吐量 (57.48%) 和低内存带宽利用率 (27.33%): 表明内核未充分利用 GPU 的计算和内存资源。
    ## 延迟问题 (Latency Issues): 低吞吐量通常由以下原因导致。内存访问延迟（例如，非合并内存访问或缓存未命中）。
    ## 建议分析方向：
    OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance 
          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate    
          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.                 

    ## fp32 的峰值性能是fp64的64倍，但是fp32 的吞吐只有16%，说明内核并未充分利用GPU的计算资源。与上述SM 吞吐 只有57.48% 一致 **
    Section: GPU Speed Of Light Roofline Chart
    INF The ratio of peak float fp32 to double fp64 performance on this device is 64:1. 
          The kernel achieved 16% of this device's fp32 peak performance and 0% of its fp64 peak performance. 
~~~

结论：
  - 305.41 MHz 的 SM 频率 太低，严重限制了吞吐。即使 Warps 活跃，处理速度也慢，间接降低占用率和整体性能。故：提高SM 频率或功率
  - 内存带宽利用率 (27.33%)，并且 缓存效率很低（L1: 29.40%, L2: 22.36%）,这是内存瓶颈，内存吞吐量远低于峰值，属于 Roofline 图的内存受限区域（低计算强度）
  - Waves Per SM = 21.33 表明局部并行性强，但未覆盖更多 SM，考虑到SM只有4个，所以这里的并行程度挺好。
  - Roofline Chart 表明 FP32 峰值性能是 FP64 的 64:1，提示设备对单精度浮点（FP32）优化较强。故：使用fp32 而不是fp64.
  - Compute (SM) Throughput = 57.48% 低于 60% 推荐阈值，表明 SM 计算资源未充分利用。
  - 极低内存和缓存吞吐量表明内存访问效率差，Warp 因等待全局内存或缓存未命中暂停，降低了活跃 Warps 数量，影响占用率。




~~~sh
    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128  # 若SM 占用率低，则考虑增大 Block Size。
    Function Cache Configuration                     CachePreferNone  # L1 缓存配置策略：此处是默认策略，可选 cudaFuncCachePreferShared 优先共享内存，cudaFuncCachePreferL1 优先 L1 缓存。如果内核依赖频繁的全局内存访问，尝试 cudaFuncCachePreferL1 以提高缓存命中率。如果内核大量使用共享内存，尝试 cudaFuncCachePreferShared 以分配更多共享内存。
    Grid Size                                                  1,024  # 决定了任务的并行程度，太小，表示SM没有被充分利用
    Registers Per Thread             register/thread              16  # 小值 有助于提高 SM 占用率。寄存器使用率低，所以不是瓶颈。
    Shared Memory Configuration Size           Kbyte           16.38  # 如果共享内存使用量高，会限制每个 SM 的线程块数量
    Driver Shared Memory Per Block       Kbyte/block            1.02  # 由 CUDA 驱动为每个线程块分配的共享内存，通常用于内核的内部管理（例如，参数传递或临时存储）。小值，故不是瓶颈
    Dynamic Shared Memory Per Block       byte/block               0  # <<<>>>如果内核需要大量线程间数据共享，但未使用动态共享内存，考虑在内核中显式分配动态共享内存。
    Static Shared Memory Per Block        byte/block               0  # __shared__
    # SMs                                         SM               4  # Orin 只有4个sm
    Threads                                   thread         131,072
    Uses Green Context                                             0
    Waves Per SM                                               21.33  # 每个 SM 的“波数”（Waves），表示每个 SM 可以调度的线程块波次（Wave），反映并行性。Waves Per SM = 总线程块数 ÷ SM 数 ÷ 每个 SM 的最大线程块数。（每个SM中，执行时的block数占 每个SM中最大Block数）。21.33 的 Waves Per SM 表明你的内核有足够的线程块（1,024）来保持 SM 忙碌。
    -------------------------------- --------------- ---------------
~~~

结论
  - SM =4，表示 程序的并行性手硬件限制，整体计算能力较低。故：需要最大化每个 SM 的利用率
  - Waves Per SM = 21.33 表示每个 SM 需要调度 21.33 波线程块（256 ÷ 12 ≈ 21.33），表明**局部并行性强**，调度深度足够隐藏部分延迟。
  - Block Size = 128（4 Warps/块）较小，限制了每个 SM 的 Warps 并发性（Block Limit Warps = 12）。故：增大 Block Size 以增加 Warps 并发性。最大化每个SM利用率。



~~~sh 
    ## 占用率是衡量 SM 上同时活跃的线程束（Warps）占硬件最大 Warps 数的比例，是评估 GPU 并行效率的核心指标。
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16  # 16 表示每个 SM 最多能同时驻留 16 个线程块。
    Block Limit Registers                 block           32  # 32 表示如果仅考虑寄存器限制，每个 SM 最多能支持 32 个线程块
    Block Limit Shared Mem                block           16  # 16 表示共享内存限制了每个 SM 最多驻留 16 个线程块。 16.38 KB ÷ 1.02 KB/block ≈ 16
    Block Limit Warps                     block           12  # 12 表示 Warps 限制了每个 SM 最多驻留 12 个线程块。 Block Limit Warps (12) 是所有限制中最低的，直接限制了每个 SM 的线程块数，进而影响占用率。
    Theoretical Active Warps per SM        warp           48  # 基于这个内核配置，的active warp 每个SM。他是硬件和内核配置共同决定的指标，不是硬件固有。
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        84.25
    Achieved Active Warps Per SM           warp        40.44
    ------------------------------- ----------- ------------

    ## 实际占用率 (84.25%) 低于理论占用率 (100%)，性能损失约 15.75%
    ## 100% 表示 内核配置（Block Size、寄存器、共享内存等）理论上支持 SM 完全利用所有可用 Warps，即 48 个/SM
    ## 原因是 warp scheduling overheads or workload imbalances during the kernel execution
    OPT   Est. Local Speedup: 15.75%                                                                                    
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (84.3%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ## Achieved Occupancy = 84.25%（低于理论 100%）表明 SM 的并行性未完全发挥，Warps 受调度开销和负载不平衡限制。
    ## Block Limit Warps = 12 是主要瓶颈，限制了每个 SM 的并发线程块数。
~~~

结论：
  - Achieved Occupancy = 84.25% 低于理论值 (100%)，Warps 因调度开销或负载不平衡未满载。84.25% < 100%: 实际占用率低于理论最大值，存在 15.75% 的性能差距（100% - 84.25%），表明 SM 的并行性未完全发挥。84.25% 虽然不算极低，但在高性能计算中，通常希望占用率接近理论最大值（例如，>90%）。故：提高每个SM的占用率。 

  - 40.44 < 48: 实际活跃 Warps (40.44) 低于理论最大值 (48)，差距为 48 - 40.44 = 7.56 个 Warps。占用率计算: 40.44 ÷ 48 = 84.25%，与 Achieved Occupancy 一致，确认实际并行性未达到理论最大值。实际 Warps 数量不足，**表明 SM 未充分利用所有可用 Warps**，进一步支持“占用率不高”的判断。故：SM的并行性不高。并且 主要由 Warp 调度开销和负载不平衡引起。见OPT

  - Block Limit Warps = 12 是主要瓶颈，即每个 SM 最多驻留 12 个线程块，12*4(SM)=48 个Warps。 故：Block Size = 128（4 Warps/块）较小，限制了 Warps 并发性。

  - 占用率不高：主要原因是 Warps 限制（Block Limit Warps = 12）、内存延迟（低吞吐量）、负载不平衡和低频率。故：增加 BLOCK size；平衡负载；增加SM频率

~~~sh
    ## 助于分析 GPU 的计算和内存访问效率，识别潜在的性能瓶颈。
    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average L1 Active Cycles         cycle    28,654.75  # L1 活跃比例 ≈ 28,654.75 ÷ 30,832 ≈ 92.94%。但是 L1/TEX Cache Throughput = 29.40%，表明 L1 缓存的带宽利用率低。结论：缓存命中率低，缓存效率低。NCU 的内存统计，分析 L1 缓存命中率（Hit Rate）
    Total L1 Elapsed Cycles          cycle      123,328  # 每个 SM 的总周期 = 123,328 ÷ 4 = 30,832。

    Average L2 Active Cycles         cycle    26,799.25  # L2 活跃比例 ≈ 26,799.25 ÷ 32,515 ≈ 82.41%。 结论：82.41% 表明 L2 缓存在大部分时间都在处理请求，存在较多 L1 缓存未命中的请求（L1 Misses）到达 L2。 *** L2 Cache Throughput = 22.36%（Speed Of Light Throughput）表明 L2 缓存带宽利用率极低，浪费 L2 带宽。 结论：1. L2 缓存命中率低，导致请求频繁到达 DRAM。2. 非合并内存访问或数据局部性性 ***
    Total L2 Elapsed Cycles          cycle      130,060  # 每个 SM 的总周期 = 130,060 ÷ 4 = 32,515。

    Average SM Active Cycles         cycle    28,654.75  # 28,654.75 ÷ (123,328 ÷ 4) ≈ 92.94%，与 L1 活跃比例一致。92.94% 表明 SM 大部分时间都在工作，但 Compute (SM) Throughput = 57.48%（Speed Of Light Throughput）表明计算效率低。*** 原因：1. 线程束因内存延迟或指令依赖频繁暂停。2. 低 SM 频率 (305.41 MHz) 限制了计算能力。
    Total SM Elapsed Cycles          cycle      123,328

    # SMSP 是 SM 内的更细粒度执行单元，负责调度和执行线程束（Warps）
    Average SMSP Active Cycles       cycle    28,558.69  # 与SM 活跃比例一致
    Total SMSP Elapsed Cycles        cycle      493,312  # 每个SM有 4 个 SMSP
    -------------------------- ----------- ------------

    WRN   The optional metric dram__cycles_active.avg could not be found. Collecting it as an additional metric could   
          enable the rule to provide more guidance.                                                                     
~~~

结论：
 - L1 (92.94%) 和 L2 (82.41%) 高活跃比例表明内存子系统忙碌，但低吞吐量提示非合并访问或低缓存命中率，导致 Warps 暂停，降低占用率。
 - SM 和 SMSP 高活跃度 (92.94%) 表明 SM 大部分时间在工作，但低计算吞吐量 (57.48%) 提示线程束因延迟暂停，减少活跃 Warps。
 - 低内存吞吐量 (27.33%) 和低缓存效率提示内核位于 Roofline 模型的内存受限区域，Warps 因内存延迟暂停，降低占用率。



## 分支

~~~sh
    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.04  # 分支指令 极少，控制流的开销可以忽略
    Branch Instructions              inst       12,288
    Branch Efficiency                   %          100  # 100% 表明所有分支指令的执行路径都完全一致，没有浪费的指令周期（即没有分支发散）
    Avg. Divergent Branches                          0  # Warp 内线程因分支选择不同路径的情况。所有都是相同的路径没有发散。
    ------------------------- ----------- ------------
~~~

结论：

  - 0.04% 是一个非常低的比例，说明你的内核以直线代码（Linear Code）为主，控制流开销极小。没有分支发散。
  - 性能瓶颈不是由于分支发散带来的。。
  - 分支发散不是 Achieved Occupancy = 84.25% 低的原因
  - 15.75% 的占用率差距 由 Warp 调度开销（如内存延迟）和 负载不平衡（块间工作量不均，而非块内发散）引起。
  - 性能瓶颈更可能来自内存访问或硬件限制


## PM Sampling

给出来性能检测采样，与cuda kernel 没有直接关系
~~~sh
    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte         1.05
    Dropped Samples                sample            0  # 没有数据丢失，所有性能计数器数据都被完整捕获，报告中的数据可信, 
    Maximum Sampling Interval          us            1  # 1 µs 是一个非常短的采样间隔，表明 NCU 采用高频采样以捕获详细的性能数据
    # Pass Groups                                    2  # 这是一个基数性指标，NCU 为收集所有请求的性能计数器数据所需的采样轮次（Pass Groups）数量
    ------------------------- ----------- ------------
~~~

结论：
  - 1 µs 间隔 保证了采样数据的高时间分辨率
  - 没有数据丢失，报告可信


## Compute Workload Analysis

~~~sh
    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         2.47  # 在活跃期，每个周期执行的指令数
    Executed Ipc Elapsed  inst/cycle         2.30  # 表示在整个内核执行期间（Elapsed Cycles，包括活跃和非活跃周期），平均每个周期执行的指令数。
    Issue Slots Busy               %        61.88  # sm 调度器
    Issued Ipc Active     inst/cycle         2.48  # SM 活跃周期内，调度器平均每个周期发出的指令数
    SM Busy                        %        61.88  # 表示 SM 在整个内核执行期间的繁忙百分比，即 SM 有活跃 Warps 执行指令的周期占比
    -------------------- ----------- ------------

    INF   ALU is the highest-utilized pipeline (41.1%) based on active cycles, taking into account the rates of its     
          different instructions. It executes integer and logic operations. It is well-utilized, but should not be a    
          bottleneck.  
~~~

IPC 反映了 SM 的指令执行效率，值越高表示 SM 更高效地利用了计算资源。A100 的理论最大 IPC ≈ 4。

2.47 是一个中等偏高的值，表明 SM 在活跃周期内执行效率较好，但未达到理论峰值。2.47 表示 SM 的计算管道在活跃时利用率较高，但仍未饱和（可能因 Warps 暂停或调度限制）。

2.30 意味着考虑所有周期（包括 SM 空闲或等待的周期），每个周期执行 2.30 条指令。2.30（Elapsed） < 2.47（Active）表明 SM 有非活跃周期，降低了整体 IPC。非活跃周期是性能瓶颈的关键来源，需优化内存访问和 Warps 并发性。

61.88% 意味着调度器有 61.88% 的时间在发出指令，其余时间可能因 Warps 暂停或缺乏可调度指令而空闲。高百分比表示调度器高效利用，低百分比表示 Warps 暂停或指令依赖。61.88% 是一个中等值，表明调度器利用率一般。结论：61.88% 与 Compute Throughput = 57.48% 一致，表明 SM 未充分利用计算资源。

Sm busy：高百分比表示 SM 高效利用，低百分比表示 Warps 暂停或缺乏工作。其实与上层的性能指标结果一致。【这里的就有些深入了】


## Memory Workload Analysis

~~~sh
    Section: Memory Workload Analysis
    --------------------------- ----------- ------------
    Metric Name                 Metric Unit Metric Value
    --------------------------- ----------- ------------
    Mem Busy                              %        27.54
    Max Bandwidth                         %        26.69
    L1/TEX Hit Rate                       %        20.43
    L2 Compression Success Rate           %            0
    L2 Compression Ratio                               0
    L2 Hit Rate                           %        50.04
    Mem Pipes Busy                        %        26.69
    --------------------------- ----------- ------------

    Section: Memory Workload Analysis Chart
    WRN   The optional metric dram__bytes_read.sum.pct_of_peak_sustained_elapsed could not be found. Collecting it as   
          an additional metric could enable the rule to provide more guidance.                                          

    Section: Memory Workload Analysis Tables
    OPT   Est. Local Speedup: 84.6%                                                                                     
          The memory access pattern for global loads from DRAM might not be optimal. On average, only 4.9 of the 32     
          bytes transmitted per sector are utilized by each thread. This applies to the 99.9% of sectors missed in L2.  
          This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced  
          global loads.          
~~~


## Scheduler Statistics

~~~sh
    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        62.09
    Issued Warp Per Scheduler                        0.62
    No Eligible                            %        37.91
    Active Warps Per Scheduler          warp         9.94
    Eligible Warps Per Scheduler        warp         1.82
    ---------------------------- ----------- ------------
~~~

## Warp State Statistics

~~~sh
        Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        16.02
    Warp Cycles Per Executed Instruction           cycle        16.07
    Avg. Active Threads Per Warp                                31.19
    Avg. Not Predicated Off Threads Per Warp                    28.99
    ---------------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 41.77%                                                                                    
          On average, each warp of this kernel spends 6.7 cycles being stalled waiting for a scoreboard dependency on a 
          L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon  
          to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory      
          access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing    
          data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to   
          shared memory. This stall type represents about 41.8% of the total average of 16.0 cycles between issuing     
          two instructions.                                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Kernel Profiling Guide                                                                     
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.      
~~~

## Instruction Statistics
~~~sh
    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst       17,664
    Executed Instructions                           inst      282,624
    Avg. Issued Instructions Per Scheduler          inst    17,722.44
    Issued Instructions                             inst      283,559
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 4.576%                                                                                          
          This kernel executes 69632 fused and 32768 non-fused FP32 instructions. By converting pairs of non-fused      
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 16% (relative to its  
          current performance). Check the Source page to identify where this kernel executes FP32 instructions.  
~~~



## 就这部分，ncu对某一指标深入分析  

  - NCU 给出所在硬件的各个硬件指标
  - NCU 的内存统计部分
  - NCU 的缓存命中率统计，是否调整缓存策略
  - NCU 的 L2 缓存命中率。如果命中率低，优化数据布局（例如，连续内存访问或提高数据局部性）。
  - NCU 的 SM 利用率，是否高的并行程度和高的SM利用率
  - NCU 的 Scheduler Statistics 和 Warp State Statistics，分析 sm 大部分时间在工作，但吞吐很低
  - NCU 的 Spill Store/Load 统计，确认是否存在寄存器溢出


# 所以总上述：

  - 内存延迟：Memory Throughput = 27.33%, L1/TEX Throughput = 29.40%, L2 Throughput = 22.36%，高 L1/L2 活跃度（92.94%, 82.41%）提示非合并访问或低缓存命中率。
  - 低频率：SM Frequency = 305.41 MHz 严重限制计算效率（FP32 性能 = 16%）。
  - Warps 限制：Block Limit Warps = 12（12 块 × 4 Warps = 48）限制了并发性，导致 Achieved Active Warps per SM = 40.44 < Theoretical Active Warps per SM = 48。
  - 负载不平衡：尽管块内无发散（Avg. Divergent Branches = 0），块间工作量可能不均（例如，某些线程块任务量更大）。


