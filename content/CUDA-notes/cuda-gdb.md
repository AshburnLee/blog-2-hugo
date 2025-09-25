+++
date = '2025-09-18T22:55:09+08:00'
draft = false
title = 'cuda-gdb'
tags = ["CUDA", "cuda-gdb"]
categories = ["CUDA"]
+++


# 结论

- 使用 printf 可以最直观表达线程的行为
- 当每个 block 使用更多寄存器时，**SM 能容纳的 block 数量减少**，故减少block大小。
- 所以 block 数量越少（相应的block中thread 就大），这时调度多block的开销就小。故减少block个数。
- 硬件信息
  - 我的设备每个 SM 最大支持 warp 数是 48
  - 硬件最大 每个调度器可调度warps数是 12（同时保持 12 个活跃 warp 可供调度，不是同时执行）
  - 每个 SM 有 4 个 warp 调度器
    - 每个 warp scheduler 每个时钟周期最多可发射 1 条指令（给 1 个 warp）
    - 为了隐藏延迟（如内存访问），它需要 多个 warp 处于“就绪”状态，以便在某个 warp 停顿时切换到另一个。这就是“零开销上下文切换”


## cuda-gdb

cuda-gdb 只能调试 .cu 文件，而不能调试 .py 文件。故 binding 到 python 接口不适用（或可用但复杂，待验证）。

1. nvcc -g -G -o app app.cu
2. cuda-gdb ./app
3. 设置断点: (cuda-gdb) break softmax.cu:25  # max_val = shared_data[0]
4. run
5. 切换到 block 0
    1. (cuda-gdb) cuda block (0,0,0) thread (0,0,0)
    2. (cuda-gdb) print shared_data[0]
6. 切换到 block 1
    1. (cuda-gdb) cuda block (1,0,0) thread (0,0,0)
    2. (cuda-gdb) print shared_data[0]
7. 继续到下一个断点
8. (cuda-gdb) continue
9. (cuda-gdb) quit


## KAQ: 只要 run 就会报错 segfault

确认源码无错误

`0x0000fffff46c168c in ?? () from /lib/aarch64-linux-gnu/libcudadebugger.so.1`

ldd /home/junhui/workspace/my-ops/gdb/build/softmax

没有 libcudadebugger.so，这表明 libcudadebugger.so 是在运行时由 cuda-gdb 动态加载的，而不是 softmax 可执行文件的直接依赖

我的设备是 aarch64 架构（ Jetson Orin Nano 或 AGX Orin），而 cuda-gdb 在 ARM 上的 GPU 调试支持不稳定。【社区有相同的问题】


## printf 获取真正的一线信息，杀手锏

- 一定明确你要打印的内容，然后再打印
- 每一次打印一项你关心的内容。内容太多，脑袋就乱了

打印 第一个 block 中 threadIdx.x 范围，符合预期；每个 grid 中 blockIdx.x 范围，符合预期
~~~cpp
if (blockIdx.x == 0) {
    printf("block %d: threadIdx.x = %d (range: 0 to %d)\n", threadIdx.x, threadIdx.x, blockDim.x - 1);
}

if (threadIdx.x == 0) {
    printf("Block %d: blockIdx.x = %d (range: 0 to %d)\n", blockIdx.x, blockIdx.x, gridDim.x - 1);
}
~~~

不要在 ncu 中观察数值，会多次采样,会有多从从夫打印，所以直接执行 `/softmax > value.txt`

~~~cpp
    float max_val = -INFINITY;
    int loop_count = 0;  // 计数器
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (i < size && d_input[i] > max_val) max_val = d_input[i];
        loop_count++;
    }
    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    // 打印 block 0, thread 0 的循环次数
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Block 0, Thread 0: loop_count = %d\n", loop_count);
    }
    if (blockIdx.x == 1 && threadIdx.x == 0) {
        printf("Block 1, Thread 0: loop_count = %d\n", loop_count);
    }
    if (blockIdx.x == 2 && threadIdx.x == 0) {
        printf("Block 2, Thread 0: loop_count = %d\n", loop_count);
    }
    if (blockIdx.x == 3 && threadIdx.x == 0) {
        printf("Block 3, Thread 0: loop_count = %d\n", loop_count);
    }
~~~

对于case=128, 4 个 block，每个 block 中 thread 32。返回：

~~~txt
Block 1, Thread 0: loop_count = 4
Block 0, Thread 0: loop_count = 4
Block 3, Thread 0: loop_count = 4
Block 2, Thread 0: loop_count = 4
~~~


## 关键* 

上述表示每个 thread 都循环了4次。

关键在 `i` 和 `d_input[i]`, `i` 不是 built-int 的变量，他可以是任何值，与`d_input[i]` 一起，block是透明的，又因为每个 warp 都执行相同的指令，所以每个thread都就执行4次。这个过程与全局 `idx` 无关，与 `blockIdx.x` 无关，也就是说，4个block中相同位置的 thread 都会执行循环体中的内容 4 次。故 4 个 block 中shared memory 中的内容是一样的。每个位置的值是4个block中相同位置上元素的最大值。【已验证】

因为 thread 不知道它在哪个 block。


## shared memory 

打印每个 block 中每个 shared memory 中的值 ，符合预期：
~~~cpp
for (int i = 0; i < blockDim.x; i += blockDim.x) {
    printf("IDX %d, Block %d, threadidx: %d, shared_data[%d] = %f\n", idx, blockIdx.x, threadIdx.x, threadIdx.x, shared_data[threadIdx.x]);
}
~~~


## block 线程 id 和全局线程 id

block 中线程在全局访问的元素

~~~cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (threadIdx.x == 31) {
    printf("d_input[%d] = %f\n", idx, d_input[idx]);
}
if (threadIdx.x == 63) {
    printf("d_input[%d] = %f\n", idx, d_input[idx]);
}
if (threadIdx.x == 95) {
    printf("d_input[%d] = %f\n", idx, d_input[idx]);
}
if (threadIdx.x == 127) {
    printf("d_input[%d] = %f\n", idx, d_input[idx]);
}
~~~


## kernel launch 配置从多维到 1 维，性能的变化

硬件内部将多维线程索引（threadIdx.x, threadIdx.y）展平为 1D 索引。性能主要取决于线程总数和访存模式，而非索引维度。【待验证】


## KAQ：这两种 kernel launch 的配置在硬件上的布局是否一样？ 

检查每一种配置的实际硬件上 thread 布局。

~~~cpp
// config #1
grid(4, 1, 1)，block(64, 1, 1);

// config #4
grid(1, 1, 1)，block(256, 1, 1);   // block中threadIdx.x = 0~255, thread 不可分，导致SM 负载不均衡
~~~

工具

- `sudo $(which ncu) ./softmax`
- `sudo $(which ncu) --metrics sm__warps_launched,sm__ctas_launched,sm__warps_active.avg.pct_of_peak_sustained_active ./softmax`


### grid(4, 1, 1)，block(64, 1, 1)

~~~txt
    sm__ctas_launched.avg                                   block            1
    sm__ctas_launched.max                                   block            1
    sm__ctas_launched.min                                   block            1
    sm__ctas_launched.sum                                   block            4
    sm__warps_launched.avg                                   warp            2
    sm__warps_launched.max                                   warp            2
    sm__warps_launched.min                                   warp            2
    sm__warps_launched.sum                                   warp            8
    sm__warps_active.avg.pct_of_peak_sustained_active           %         4.17

    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %         4.17

    Average SM Active Cycles         cycle    83,842.75
    Total SM Elapsed Cycles          cycle      343,956
~~~

- 每个SM launch 一个 block，每个 block 2 个 warps，负载均衡，很好
- 83,842*4 = 335,368 接近 343,956，表明 SM 绝大部分 cycle 处于活跃状态，好


### grid(1, 1, 1)，block(256, 1, 1)

~~~txt
    sm__ctas_launched.avg                                   block         0.25
    sm__ctas_launched.max                                   block            1
    sm__ctas_launched.min                                   block            0
    sm__ctas_launched.sum                                   block            1

    sm__warps_launched.avg                                   warp            2
    sm__warps_launched.max                                   warp            8
    sm__warps_launched.min                                   warp            0
    sm__warps_launched.sum                                   warp            8

    sm__warps_active.avg.pct_of_peak_sustained_active           %        16.66

    Theoretical Occupancy                         100%
    Achieved Occupancy                           16.66%

    Average SM Active Cycles         cycle    20,096.25  **
    Total SM Elapsed Cycles          cycle      326,232
~~~

- 只有一个 block（cta）被 launch，所以 sm 中** warps 利用率同样很低**。
- 唯一一个 SM 的占用率有 16.66%，这个应该是数据量小的限制。
- 20,096.25*4 = 80,384 远小于 326,232，表明**SM空闲时间很长**。


## KAQ：多维的配置和 1 维的配置在硬件中是否一样？

~~~cpp
// config #1
grid(4, 1, 1)，block(64, 1, 1);

// config #2
grid(2, 2, 1)，block(32, 2, 1);
~~~

### 1. 确认 config#2 的全局 idx 如何计算？

对于config#2，

1. 计算thread 在 block 中的索引：`threadIdx.y * blockDim.x + threadIdx.x`
2. 计算每个block线程数：`blockDim.x * blockDim.y`
3. 计算block 在 grid 中的索引：`blockIdx.y * gridDim.x + blockIdx.x`
4. 全局id =（block 在 grid 中的索引）x（每个block线程数）+（thread 在 block 中的索引）

即 `global_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);`


### 2. config#1 和 config#2 ，与 torch 相比保证 kernel 的正确性

通过 UT（with epsilon = 1e-8）。


### 3. config#2 查看硬件映射

与 config#1 相同。


## KAQ：多维 VS 1维 是否有性能变化

增大数据量 256->4096，观察关键性能指标

### block =（64,1,1）grid =（64,1,1）

~~~txt
sm__ctas_launched.avg                                   block           16
sm__ctas_launched.max                                   block           16
sm__ctas_launched.min                                   block           16
sm__ctas_launched.sum                                   block           64
sm__warps_launched.avg                                   warp           32
sm__warps_launched.max                                   warp           32
sm__warps_launched.min                                   warp           32
sm__warps_launched.sum                                   warp          128
sm__warps_active.avg.pct_of_peak_sustained_active           %        66.11
~~~

硬件上映射符合预期：每个SM 启动了 16 个block，每个SM 有32个 warps, 所以每个 block 有 2 warps。
总thread个数：32 warps/SM * 32 threads/warp * 4 SM = 4096 threads, 完全覆盖元素个数。

~~~txt
Duration                         us       787.39
Compute (SM) Throughput           %        53.12

Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           16
Block Limit Registers                 block           24
Block Limit Shared Mem                block           25
Block Limit Warps                     block           24
Theoretical Active Warps per SM        warp           32
Theoretical Occupancy                     %        66.67
Achieved Occupancy                        %        66.12
Achieved Active Warps Per SM           warp        31.74
------------------------------- ----------- ------------

OPT   Est. Local Speedup: 33.33%
        The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
        hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of blocks that
        can fit on the SM.

8/12=

Average SM Active Cycles         cycle      237,521
Total SM Elapsed Cycles          cycle      958,064
~~~

- 237,521*4 = 950,084 与 958,064 非常接近，表示 SM 都在Active的。SM利用率很高
- 66.11 与 66.67 非常接近，表示占用率很高
- 在硬件资源（如寄存器、共享内存）和launch 配置限制下，该 kernel 理论上可以激活 GPU 上最大可能的 66.67% 的 warps
- 66.11% 表示kernel 运行时实际激活的 warps 比例，接近理论值，说明实际运行时资源利用接近预期
- 66.67% 的占用率虽然较好，但并不是非常高，仍可能有进一步优化空间【如何做】
- 每个 warp scheduler 理论上调度 8 个active Warp，而硬件上每个 warp scheduler 最多可以调度 12 个 active Warp（这个case的理论占用率只有 8/12=66.7%）。也就是说该 kernel 的线程调度水平还没有达到硬件能力的极限，还有提升空间。


### block =（32,2,1）grid =（32,2,1）与上应该是一样的

确认与上一个配置一样。（至少是basic 报告上一样）


### block =（32,4,1）grid =（8,4,1）

~~~txt
sm__ctas_launched.avg                                   block            8
sm__ctas_launched.max                                   block            8
sm__ctas_launched.min                                   block            8
sm__ctas_launched.sum                                   block           32
sm__warps_launched.avg                                   warp           32
sm__warps_launched.max                                   warp           32
sm__warps_launched.min                                   warp           32
sm__warps_launched.sum                                   warp          128
sm__warps_active.avg.pct_of_peak_sustained_active           %        66.25
~~~

硬件上映射符合预期：每个SM 启动了 8 个 block，每个SM 有 32 个 warps, 所以每个 block 有 4 warps。**每个block launch 的 warp 更多**。总thread数：32 warps/SM * 32 threads/warp * 4 SM = 4096 threads, 完全覆盖元素个数。与上一个配置相比，launch的 warp个数相同，但是block 少了。直接体现在 配置上（8,4,1）。

~~~txt
Duration                         us       461.12
Compute (SM) Throughput           %        49.60

Block Limit SM                        block           16
Block Limit Registers                 block           12
Block Limit Shared Mem                block           21
Block Limit Warps                     block           12
Theoretical Active Warps per SM        warp           48
Theoretical Occupancy                     %          100
Achieved Occupancy                        %        66.20
Achieved Active Warps Per SM           warp        31.78

Average SM Active Cycles         cycle   138,068.75
Total SM Elapsed Cycles          cycle      560,884
~~~

- 138,068.75*4=552,275 与 560,884 很接近，表明 SM 大部分时间都在活跃状态。
- Theoretical Occupancy 达100%，【为什么】


## block =（64,1,1）grid =（64,1,1）和 block =（32,4,1）grid =（8,4,1） 比较

launch 配置比较：

- 前者使用较小的 block（64线程），block数量较多（64个），可能导致更高的调度开销。
- 后者使用较大的 block（128线程），block数量较少（32个），可能减少调度开销，但每个block的资源需求更高。

Block Limit Warps 比较：

- 后者由于每个 block 有 4 个warp（128 thread），warp 限制更严格，Block Limit Warps 只有前者的一半。
- 后者寄存器限制更严格，表明每个 block 的寄存器使用量较高（因为block中thread数量多呀）。

Occupancy 比较：

- block =（64,1,1）：实际占用率接近理论值，但理论占用率较低（66.67%），未充分利用SM的并行能力，受限于 block 数量（无法填满 SM 的 warp 容量）。【为什么】
- block =（32,4,1）：100%，理论上完全利用 SM 资源。但实际占用率仅66.20%，远低于理论值，


## 提示

- 如果内核支持，考虑使用 CUDA 动态并行（Dynamic Parallelism）来优化工作负载分配。
- 确保配置针对具体GPU架构（如Volta、Ampere或Hopper）优化，检查 SM 的最大 warp 数和寄存器/共享内存容量。
- 一个指标：每个Warp调度器理论上管理的活跃 Warp 数


## KAQ：SM 上block 的调度需要开销吗？block大小较小（64线程），为什么会导致调度开销较高？

是的，SM 通过调度Warp 调度block，每个block需要分配寄存器、共享内存和其他SM资源。当一个block完成执行或被暂停（例如等待内存访问），SM需要切换到另一个block，这涉及保存和恢复上下文（如寄存器状态），会引入一定开销（很小的开销）。所以 block 数量越少（相应的block中thread 就大），这时调度多block的开销就小。

结论：增加 block 大小。


## KAQ：Block Limit Registers

- block =（64,1,1）grid =（64,1,1）：Block Limit Registers = 24，意味着每个SM受寄存器限制，最多能同时运行 24 个block。即每个block的寄存器使用量较低（thread个数少），允许 SM 容纳较多 block。
- block =（32,4,1）grid =（8,4,1）：Block Limit Registers：12，意味着每个SM受寄存器限制，最多能同时运行12个block。即每个block的寄存器使用量较高，导致SM能容纳的block数量减少。

后者每个 SM 能容纳的 block 数量减少，**限制了并行性**。

$$\text{Block Limit (Registers)} = \left\lfloor \frac{\text{SM的寄存器总数}}{\text{每个block的寄存器使用量}} \right\rfloor$$

这个指标的意思是，当每个block使用更多寄存器时，**SM 能容纳的 block 数量减少**，称为“寄存器限制更严格，使 SM 中block 数量减小”。这意味着寄存器资源成为限制并行性的瓶颈。即 每个SM能**同时运行**的block数量减少。

结论：减少 block 大小。


## KAQ：Jetson 每个 SM 最大支持 warp 数？

一个事实：最大 Warp 数始终等于每个 SM 的最大驻留线程数（Maximum Resident Threads per SM）除以 Warp 大小（32）。

如 GA10B 上 “Maximum number of threads per multiprocessor = 1536”（通过device query 获得），**1536/32 = 48**。所以**Jetson 每一个SM 最大支持warp数是48** ***。


## KAQ：Jetson 每个 SM 有多少个 warp 调度器？

根据“The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12” 这句话，**硬件最大warps/scheduler 是 12**。又根据每一个SM 最大支持 warp 数是 48，所以**每个SM 有 4 个 warp 调度器**。


# 一维和多维 id 在访问 global memory/shared memory时，行为一样吗？

## KAQ：在 2D block（32,2,1）中，相邻线程通常指 threadIdx.x 连续的线程（threadIdx.y 固定），因为硬件按 threadIdx.x 组织 warp（32 线程）。

同一 warp 中的相邻线程通常指 `threadIdx.x` 连续的线程（`threadIdx.y` 固定）。`threadIdx.x` 相邻的线程更有可能在同一个 warp 中，从而可以利用 warp 级别的操作和共享。

硬件上直接验证，不知道如何做，困难，但是可以通过 shuffle指令 间接验证。

Warp 洗牌指令允许 warp 中的线程直接交换数据，而无需通过共享内存。如果硬件按照 `threadIdx.x` 组织 warp，那么使用 `__shfl_sync` 在 `threadIdx.x` 相邻的线程之间交换数据应该会非常高效。

编写 CUDA kernel，使用 `__shfl_sync` 在 `threadIdx.x` 相邻的线程之间交换数据，并测量其性能。 然后，尝试在 `threadIdx.y` 相邻的线程之间交换数据，并比较性能。


## KAQ：block (64,2,1) 和 block（16,8,1），warp 数量和调度效率有什么不同？【todo】

block中thread数相同，都会被展开到1维，直观上看，两者数量和调度效率应该一样。但是，需确认
- 全局内存访问是否合并
- 共享内存访问的bank冲突


## KAQ：GPU warp 以 threadIdx.x 顺序执行，访问连续地址可触发全局内存合并（coalesced access）


