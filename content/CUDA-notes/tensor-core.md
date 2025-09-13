+++
date = '2025-01-31T12:45:17+08:00'
draft = false
title = 'Tensor Core'
tags = ["CUDA","Tensor Core"]
categories = ["CUDA"]
+++


## Tensor Core 在硬件中是如何组织的？


在 SM 中，CUDA Core 和 Tensor Core 协同工作，共同完成计算任务。CUDA Core 负责执行通用计算任务，而 Tensor Core 负责加速矩阵乘法运算。

协同工作通常是**异步**的通过将矩阵乘法运算卸载到 Tensor Core，CUDA Core 可以专注于执行其他任务，从而提高整体性能。

即一旦 Tensor Core 操作启动，CUDA Core 不需要等待 Tensor Core 完成计算。CUDA Core 可以继续执行其他任务，例如数据预处理、激活函数计算或其他类型的计算。

但是 当 CUDA Core 需要 Tensor Core 的计算结果时，它会从 Tensor Core 的输出缓冲区中读取数据。

如果许需要同步，CUDA 程序可以创建一个事件（Event），并在 Tensor Core 操作完成后记录该事件。然后，CUDA Core 可以等待该事件被记录，以确保 Tensor Core 的计算已经完成。事件比 `cudaDeviceSynchronize()` 更细粒度，可以减少阻塞的时间。


## Tensor Core 的输出缓冲区

输出缓冲区位于 register，也可以更具实际需求放在在 shared memory 中。


## Tensor Core 编程模型

Tensor Core 的编程模型与传统的 CUDA 编程模型有所不同。在 CUDA 中，每个线程的数据通常是私有的，但在 Tensor Core 编程中，需要考虑 warp 级别的操作，因为 Tensor Core 是在 warp 内部进行操作的（所以是 `Warp MMA`）。

- Warp 级别编程：Tensor Core 的操作通常在 warp 级别进行。一个 warp 中的多个线程协同工作，共同完成一个矩阵乘法操作 
- 混合精度：Tensor Core 支持**混合精度计算**，通常输入数据是半精度（FP16），而**累加器**使用单精度（FP32）或半精度
- 显式数据加载和存储：需要显式地将数据加载到 Tensor Core 的寄存器中，进行计算，然后将结果写回内存。这通常涉及到使用特定的 WMMA API
- 数据分块：由于 Tensor Core 处理的是小块矩阵，因此需要将大矩阵分成小块，然后分批加载到 Tensor Core 中进行计算


## Tensor Core 上的矩阵乘法 都是小尺寸的？大矩阵如何处理的？

虽然 Tensor Core 本身处理的是小尺寸的矩阵，但通过合理的分块策略和优化手段，可以在实际场景中高效地处理非常大的矩阵。关键在于将大矩阵分解成小块，并充分利用共享内存和数据预取等技术，以减少全局内存的访问次数，提高计算效率。

具体实战，见上述链接中的 1024x1024 的实例。


## 实例中是否体现如下优化策略？

优化策略

1. 共享内存：使用共享内存来存储小块矩阵，减少全局内存的访问次数 
2. 数据预取：在计算当前小块矩阵的同时，预取下一个小块矩阵的数据，以隐藏全局内存的访问延迟。
3. 循环展开：展开内层循环，以减少循环开销，提高计算效率。
4. 对齐访问：确保内存访问是对齐的，以提高内存访问效率。
5. 选择合适的块大小：根据 GPU 的架构和资源，选择合适的块大小，以最大化 Tensor Core 的利用率。


## Tensor Core 上的矩阵乘法为什么比传统 CUDA 更高效,实践对比
## Tensor Core 会与 shared memory 交互吗
## Tensor Core 如何处理 bank confilict


## Tensor Core 期望的数据布局

建议使用 column-major 布局，因为 Tensor Core 内部的计算是基于 column-major 布局优化的。Ampere 架构上也是。

Tensor Cores 的内部架构可能针对 column-major 布局进行了优化。 例如，Tensor Cores 可能包含专门的硬件单元，用于以 column-major 顺序加载和处理数据。


## 内存中物理存储和逻辑存储

矩阵 A 大小 `5x4`，访问 `A[3][1]`, 

row-major 存储时，物理索引是： 3x4+1=13

col-major 存储时，物理索引是： 1x5+3=8


矩阵 A 是 `M x N`, 访问 `A[row][col]`,

row-major 存储时，物理索引是：row * N + col

col-major 存储时，物理索引是：col * M + row

物理上的索引与如何 interpret 这个内存有关。


## 使用 Tensor Core 之前我会在全局内存上初始化两个矩阵，这两个矩阵在内存中是如何存储的？行优先还是列优先？

张量在内存中的存储方式取决于所使用的库。C/C++ 和 Python (NumPy): 默认使用行优先 (row-major order) 存储 。这表示矩阵的每一行在内存中是连续存储的，一行接一行。

所以 Host 端初始化的矩阵都是行优先存储的。

拷贝到 Device 后，是如何存储的？

`cudaMemcpy` 本身不直接支持内存布局转换。`cudaMemcpy` 只是简单地将一块内存从一个位置拷贝到另一个位置，而不改变数据的组织方式。数据在设备端的布局将保持与主机端一致。

