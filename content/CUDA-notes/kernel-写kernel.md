+++
date = '2025-01-31T12:45:05+08:00'
draft = false
title = 'Kernel-写kernel'
tags = ["CUDA"]
categories = ["CUDA"]
+++



## 线程配置配置 最佳实战

- 当前的 GPU 上，一个 block 可能包含多达 1024 个线程。
- 如果每个 block 的 thread 数量为 `[64/128/256/512]`，那么 CUDA 性能会更好。因为 Warp 大小是 32
- 并且 block 数量应该比 SM 的数量多（至少是2到4倍）。因此，要考虑流 SM 的数量，以确定每个块正确的线程数量。
- 激活线程数是 `SM * 2`
- 硬件信息：SM 数， `GetComputeCapability`， `GetMaxThreadsPerBlock`, `GetMaxPhysicalThreadCount`
- 需要激活的线程总数 active threads。


- 得到配置：1.获取硬件信息。2.估计线程数。3.计算block数。4.创建配置

**通常是把经验配置作为配置的起点，作为 baseline**，分析资源占用率，并行程度，吞吐，带宽，甚至功耗，然后通过 Profiling 工具在此基础上进行优化。

Again，在 Profiling 时，收集关键性能指标：

  - 资源占用率 (Occupancy)： GPU 的计算资源利用率，越高越好。
  - 并行程度： 活跃线程块和 warp 的数量，反映了程序的并行性。
  - 吞吐量 (Throughput)： 单位时间内处理的数据量，例如每秒处理的元素数量。
  - 带宽 (Bandwidth)： 内存读写速度，包括全局内存、共享内存和常量内存。
  - 功耗 (Power Consumption)： GPU 的功耗，在某些情况下需要考虑。
  - 延迟 (Latency): kernel 的执行时间。

上述可以通过 **Nsight Compute** 获得。除此之外还有其他工具可以用来分析 CUDA 程序的性能：CUDA API，CUDA 提供的一些环境变量。


## Active block/thread

一个 "active block" 是指**当前正在** SM 上执行的线程块。"active threads" 是指**当前正在**执行的线程。
Active block 的数量直接影响 GPU 的占用率 (occupancy)。 较高的 active block 数量通常意味着更高的 GPU 占用率和更好的性能。

 
## 驻留 VS 活跃

`maxThreadsPerMultiprocessor` 限制的是驻留线程数，实际的活跃线程数会更少。

驻留表示线程已经获得了资源，但没有在执行。驻留线程可以有以下状态：

1. Active：正在执行指令。
2. Ready：准备好了指令，等待 Warp 调度器分配资源。
3. Blocked：等待其他某件事情发生。

**驻留线程的数量影响 GPU 的占用率 (Occupancy)**。 **活跃线程的数量影响 GPU 的利用率 (Utilization)**。 优化 CUDA 程序的目标之一是最大化占用率和利用率。


## 有了配置如何执行 kernel，`grid-stride-loop`

`func<<<4, 8>>>()`：配置启动 32个 threads， 每一个线程执行一个 `func()` 的副本。唯一不同的是每一个 thread 的 ID 。也就是每一个线程有唯一的 ID ，那么线程的 ID 需要更新吗？看情况：

给出更新线程ID 的方法：

~~~cpp
int id = threadIdx.x + blockDim.x * blockIdx.x;
while(id < N){
    // TODO excute operation
    id += blockDIm.x * gridDim.x;  // ID 更新的步长是总线程数
}
~~~

其中 `while()` 判断当前 thread 的 id 需要更新多少次。

### 1. 不需要更新的情况：就是线程配置能够覆盖所有数据：

对于 <<<1, 2048>>> 的 kernel，处理 2024 个数据（N=2024）。假如其中一个 thread 的起始 id 为 0，干完活后，判断 0＜2024，所以这个 thread 的 id 会被跟新为 2048，此时再判断 2048＜2024，返回 false，这个 thread 的工作结束。thread 的 id 未被更新。

当一个CUDA grid 能够一次性处理所有输入数据时，这种 kernel 被称为 `monolithic kernel`

### 2. 需要更新的情况：线程配置中的线程数小于数据个数：

比如 <<<1, 512>>> 的 kernel config，处理 2024 个数据（N=2024）

- 仍假如有一个 thread 是起始 id 为 0，判断 0＜2024 ，执行操作。所以跟新 id 为 512；
- 判断 512＜2024，执行操作，再更新 id 为 1024；
- 判断 1024＜2024，执行操作，再更新 id 为 1536；
- 判断 1536＜2024，执行操作，再更新 id 为 2048；
- 判断 2048＜2024，返回false。该 thread 的工作结束。

关于更新ID， 注意：

- 不管你的数据是一维的二维的还是更高维度的，**在 GPU 端，高维被扁平化，都将被看成一维的**，所以没有必要在 Device 上开辟一个二维数组。
- CUDA code 需要你**并行地思考**：Think parallel.
- 当你在写 CUDA code， 实际上你是在**为一个 thread 写串行 code**，而每一个 thread 都执行这个段相同的串行 code 。唯一区别是每个线程的 ID 各不相同
- 可以这样理解，对于简单问题，把 CPU code 的 for 循环去掉，其实就得到了 GPU code。每个 thread 有自己唯一的 ID，其他都一样。


## 线程配置和 kernel 实现

CUDA kernel 函数的编写主要取决于你的算法，而线程配置则决定了如何将线程映射到数据上以及如何管理线程间的协作。 **所以不同的线程配置（网格维度 gridDim 和块维度 blockDim）不会改变 kernel 函数的编写方式本身，但会影响你如何使用 threadIdx 和 blockIdx 等内置变量来访问数据和控制线程行为**。

高效的内存访问是 CUDA 编程的关键，而 `threadIdx` 和 `blockIdx` 变量是实现这一目标的核心。

**Global 内存访问在多个线程同时访问连续内存块**时效率最高。 这意味着，如果你的线程块中的线程以某种方式访问内存，使得每个线程访问的内存地址在内存中是连续的，那么你就能实现合并内存访问。


# softmax kernel (last axis) 配置设计

给出 input，dims，axis，计算是 softmax，根据什么得到 kernel 的配置？

输入参数：

  维度：dims = {n=32, c=4, h=256, w=256}
  轴：axis = 3（w 维度）
  dim_size：w = 256（softmax 归一化的维度大小）
  outer_size：外层维度大小，n * c * h = 32 * 4 * 256 = 32,768

## Kernel 配置设计原则：

1. warp：线程分配应尽量对齐 warp 大小32，避免线程浪费。

2. block：数通常为 32 的倍数。最大是1024（from device property）。
  - 占用率&并行性：确保 SM 上驻留多个block，充分并行一个SM

3. 并行性分配：softmax 的并行性主要来自 `outer_size`（32,768 个独立的 softmax 计算）和 `dim_size`（每个 softmax 的 256 个元素）。`outer` 和 `dim_size` 是**解耦的，独立并行**。
  - 外层并行：将 `outer_size` 分配到多个线程块或 warp；
  - 内层并行：将 `dim_size` 分配到线程或 warp 内的协作。

4. 合并内存访问：连续线程访问连续内存，对于softmax，w 维度的访问应尽量连续。
  - 合并访问是指一个 warp（32 个线程）在一次内存事务中访问**连续的内存地址**，从而最大化全局内存带宽利用率。
  - 要确定内存访问是否合并，我们需要**验证**每个 warp（32 个线程）的访问模式。当 threadIdx.x 从 0 到 31 递增，地址` w_idx = threadIdx.x` 也从 0 到 31 递增。故是合并访问。
  - 内存事务大小：访问的数据量匹配 GPU 内存事务（通常 32、64 或 128 字节）
  - 一致性：warp 内线程的访问模式一致，无分支导致的分散访问。

5. 内存对齐：访问的起始地址应对齐到 128 字节（32 个 float）。input 数组由 `cudaMalloc` 分配，基地址通常是 128 字节对齐的。

6. 规约优化:使用共享内存或 `warp shuffle` 指令可以加速归约。


## 配置 Kernel 配置

1. 确定线程块大小

选择一个合适的 block 大小，既能高效处理 dim_size=256 的归约，又能支持 outer_size=32,768 的并行性。
**dim_size = 256 表示每个 softmax 需要处理 256 个元素**。但一个 warp（32 线程）不足以覆盖 256 个元素，所以可以一个 warp 通过循环8次处理 256 个元素。或者多个线程协作: `dim3 threads(32, 4, 1)` 且 每个 `warp（threadIdx.y）`处理一个 softmax，网格大小：(32,768 + 4 - 1) / 4 = 8,192。

选择 256 个线程，与 dim_size 一致，每个线程对应一个元素。【如何做256 个元素的reduce？】并且 256 线程（**8 个 warp**）可以高效利用 SM，提供足够的并行性。所以得到了： `threads_per_block = 256`。


2. 确定线程块内的线程组织

将 256 个线程组织为 `dim3 threads(x, y, z)`，以高效映射到数据。故配置 `dim3 threads(256, 1, 1)`：线程id和元素一一对应，适合逐元素操作。归约操作（max, sum）可以通过线程块内的协作完成（例如**共享内存**或 **warp shuffle**）

直接用一维线程索引更简单，避免复杂索引计算。所以得到：`dim3 threads(256, 1, 1)`。


3. 确定网格大小

目标：分配足够的线程块以覆盖 `outer_size = 32,768` 个 softmax 计算。

假设每个线程块处理 1 个 softmax（256 线程正好覆盖 w=256），需要 `outer_size = 32,768` 个线程块。所以网格大小为 `dim3 blocks(32,768, 1, 1)`。

或者，每个线程块处理多个 softmax（例如 4 个）：配置 `dim3 threads(256, 4, 1)`，每个 `threadIdx.y` 对应一个 softmax。每个线程块处理 4 个 softmax，网格大小为 `(32,768 + 4 - 1) / 4 = 8,192`。

选择 `dim3 blocks(32,768, 1, 1)` 原因：32768 没有超过硬件限制，`blockIdx.x` 直接映射到外层索引 `i`。

4. 数据到线程的映射


## 为什么这样配置 kernel

- 高效：规约256 线程适合树形归约（每次折半），配合共享内存或 warp shuffle 加速。
- 合并访问：线程按 w 维度连续访问数据，保证内存访问合并。
- 并行性: block 内部并行；block 之间并行


## 替代配置

配置 dim3 threads(32, 4, 1)，

- 优点：更低的寄存器压力，可能提高占用率。
- 缺点：需要更多线程块，增加调度开销。


## 向上取整 ceil

ceil 的计算公式是 `ceil(x) = (x + n - 1) // n` (整数除法) 是标准的向上取整。保证了即使 x 可以被 n 整除时的正确性。

意义是 x 需要多少个 n 才能被完全覆盖。


## 线程索引与数据索引的关系

- `row_id = blockIdx.y * blockDim.y + threadIdx.y` : thread 在 grid 垂直方向的行偏移，加上 block 内线程的行偏移。thread 行全局id。（分别用 x，y 索引全局2D tid）
- `tid_in_block = threadIdx.y * blockDim.x + threadIdx.x` : 2D 线程索引转换为 1D 的情况。是其中 block 内线程的一维 id。（用 x，y 的线性组合表示 block 内1D tid）
- `idx = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x)` : 2D 线程索引转换为 1D 的情况。这个是全局的1D tid。


## 2D block 和 2D grid
###  一维局限：仅用 x 可能导致非合并访问或复杂索引，降低效率。

一维 tid 连续，但在二维数据中可能跨行（例如，tid=7 到 8，从 row=1 到 row=2），地址跳跃（7 到 8），导致非合并访问（7 到 8 数据地址不连续）。一维索引没有分割索引的边界。而二维情况，col（threadIdx.x）连续，Warp 访问同一行内地址（例如，0-31），合并高效。对于地址 32，会从 row=1 开始访问，从第二行开始。另起一行就是这个边界。

如果一个 Warp 的线程访问的是有 stride 的或者分散的地址，那么 Non-coalesced access 就会发生。一维 tid 中不总是有 non-coalesced access。


### 二维 block 布局可减少 bank 冲突。例如，block(64, 2, 1) 的 x 维度访问连续，降低冲突概率。【？】

