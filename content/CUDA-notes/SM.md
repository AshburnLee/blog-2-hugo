+++
date = '2025-01-31T12:45:49+08:00'
draft = false
title = 'SM'
tags = ["CUDA","Streaming Multiprocessor"]
categories = ["CUDA"]
+++


## SM

Streaming Multiprocessor（SM）。SM 是 GPU 上的一个**硬件单元**，负责执行 CUDA 核函数（kernels）中的**线程块**（blocks of threads）。每个 SM 能够同时处理多个线程块，这些线程块中的线程并行执行相同的指令，这种执行模式被称为 SIMD（单指令多数据流）。

SM 是 GPU 上的一个**硬件单元**，程序被组织成网格（grid），每个网格由多个线程块（block）组成，而每个线程块又由多个线程（thread）组成。这些线程以 warp为单位被分配到 SM 上执行。


## SM & SP

- SM 是核心计算单元，能够并行执行多个线程，它包含多个 SP、指令单元、共享内存、L1缓存、寄存器和其他资源。
- SM 使用 Warp 调度器来管理线程的执行。
- SP（也称为 CUDA 核心）: SP 是 SM 中最基本的处理单元，负责执行单个指令。
- 协作关系： SM 负责调度和管理线程的执行，而 SP 负责实际的计算。


## SM 中 register 资源分配

硬件条件：一个 SM 中有 768 个 threads，含有 8000 个registers。假设我配置 block 为256个 threads。如何最大化资源占用率？

当每个 thread 占用 10 个 registers，那么一个 SM 共占用 （768*10=）7680 个register，没有超过 registers 总个数 8000。SM 驻扎 （768/256=）3 个block，(768/32=) 24 个 warp。这是最理想的情况，最大化资源占用率。

当每个 thread 占用 11 个 registers, 那么SM 要占用 11x768=8448 个 register 超过了 register的总个数。此时，算数上SM最多使用 (8000/11=727.27) 727 个。但实际上，超出限制后，threads数的减少是以block为粒度的减少。若一个block 有 256 个threads，那么可用threads数就不是从 768 减少到 727 ，而是从 768 减少到 512 （第一个比 727 小的 256 的整数倍，0，256，512，768，。。。 ），所以此时，block 个数是2个（512=2x256）

有多少个 warp 呢？2个 block 有512 个 threads，这个 SM 驻扎 (512/32=) 16个warp。

**只要活动 threads 数减少，就以 block 为单位（粒度）地减少。**


## 好的 CUDA 程序

- 设备最大化并行程度
- 最大化资源利用率
- 计算吞吐量接近GPU的理论峰值
- 内存带宽接近GPU的理论峰值
- 较低的功耗下实现了较高的性能


## SM 中 Shared Memory 分配

同样的，一个 SM 中的 Shared Memory 的大小也是有限的。在同一个 block 的 threads 共享同一块 Shared Memory 。一个 SM 中实际使用的 block 数量也是与每个 block 被分配的 Shared Memory 的大小有关。

一个 SM 有8个 blocks，可使用 Shared Memory 为 16kB。

- 当 每个block被分配 Shared Memory 最多为（16kB/8=）2kB，此时资源利用率最大
- 当 每个block使用了 4kB，则实际只是用了（16kB/4kB=）4 个 blocks，并行程度只有原来的一半。
- 当 每个block使用 5kB，则实际只用了（16kB/5kB=）3 个 blocks，并行程度就更小了。

Registers数量和每个block分配到的Shared Memory的大小，共同约束了整个系统中的block数量个threads数量。实际应用中也要尽量少的使用存储资源，从而最大化并行程度。内存是竞争资源。


## 驻扎/驻留 (Resident)

表示线程或 warp 存在于 SM 中，已经分配了资源（例如，寄存器、共享内存等），但并不一定正在执行。所以：

- 每个 SM 最多驻扎 768 个 threads。意思是，768个线程已经分配到了资源，等待快速切换执行。
- 每个 SM 最多同时驻留 64 个 warp，这 64 个 warp 并不是同时执行的，而是通过快速上下文切换来提高 GPU 的利用率和性能。  SM 内部有 warp 调度器，按照某种策略在不同的 Warp 间切换，达到隐藏延迟的目的。

所以 最大驻扎线程数，即SM 能够同时容纳的线程数量是有限的，这个限制是由 SM 的硬件资源决定的。因为驻扎是已经分配好了资源，资源优先，所以最大驻扎数也是有限的。
