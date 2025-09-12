+++
date = '2025-01-31T12:45:50+08:00'
draft = false
title = 'From Hopper'
tags = ["CUDA"]
categories = ["CUDA"]
+++


https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html

NVIDIA Hopper GPU 架构保留了并扩展了之前 NVIDIA GPU 架构（如 NVIDIA Ampere GPU 架构和 NVIDIA Turing）提供的相同的 CUDA 编程模型，遵循这些架构最佳实践的应程序通常在 NVIDIA H100 GPU 上看到速度提升，无需任何代码更改。


## 高优先级建议如下

- 寻找并行化顺序代码的方法。
- 最小化主机与设备之间的数据传输。
- 调整内核启动配置以最大化设备利用率。
- 确保全局内存访问是 coalesced 的。
- 尽可能减少对全局内存的冗余访问。
- 尽量量避免同一warp中线程执行路径出现长时间的分歧(sequences of diverged execution)。

TODO：这个文档中还有更多的关于 Hopper 的参数。


## NVIDIA Hopper 流式多处理器（SM）在 Turing 和 NVIDIA Ampere GPU 架构的基础上提供了以下改进。

每个 SM 的最大并发 warp 数量与 NVIDIA Ampere GPU 架构相同（即 **64**），每个 warp 需要占用一定的寄存器和共享内存等资源。SM 的资源是有限的，因此能够同时支持的 warp 数量也是有限的。如果一个 SM 上只有少数几个 warp，那么 SM 的资源可能无法充分利用，导致性能下降。资源占用率要高，并行程度也要高。

影响 warp 占用的其他因素包括：

- 寄存器文件大小为每个 SM 64K 个 32 位寄存器。（register个数 64x1024=**65536**个，每个register大小 32bit/8=4Byte, register总大小是 256 KB）

- 每线程的最大寄存器数量为 255。

- 对于计算能力为 9.0（H100 GPU）的设备，**每个 SM 的最大 block 数为 32**，。

- 对于计算能力为 9.0（H100 GPU）的设备，**每个 SM 的 Shared mem 容量为 228KB**，比 A100 的 164 KB 容量增加了 39%。

- 对于计算能力为 9.0（H100 GPU）的设备，**每个 Block 的最大共享内存为 227 KB**。

- 对于使用线程块集群的应用，始终建议使用 cudaOccupancyMaxActiveClusters 来计算占用率，并相应地启动基于集群的内核。


## TMA 

张量内存加速器（TMA）允许应用程序在全局内存和共享内存**之间**以及同一集群（Thread Block Clusters）中不同 SM 的共享内存区域**之间**双向传输 1D 和最多 5D 张量（请参阅线程块集群）。此外，对于从共享内存到全局内存的写入，它允许指定逐元素累加/最小/最大等操作以及对于大多数常见数据类型的位与/或操作。


## Thread Block Clusters

NVIDIA Hopper 架构增加了一个新的可选层次级别，线程块集群，这为并行化应用程序提供了更多可能性。线程块可以读取、写入并对其集群内其他线程块的共享内存执行原子操作。这被称为分布式共享内存。如 CUDA C++ 编程指南所示，有些应用程序无法将所需数据放入共享内存，而必须使用全局内存。分布式共享内存可以作为这两种选项之间的中间步骤。


## 内存系统

NVIDIA H100 GPU 支持 HBM3 和 HBM2e 内存，容量高达 **80 GB**。HBM3 内存系统支持高达 **3 TB/s** 的内存带宽，比 **A100-40GB** 上的 **1.55 TB/s** 提高了 93%。


## 增加 L2 容量

NVIDIA Hopper 架构将 A100 GPU 中的 L2 缓存容量从 **40 MB** 增加到 H100 GPU 中的 **50 MB**。随着容量的增加，**L2 缓存到 SM 的带宽也增加了**。

持久化 L2 缓存 (**Persistent L2 Cache**): 在 Compute Capability 8.0 及以上的设备上，CUDA 支持持久化 L2 缓存。 这允许开发者将频繁访问的数据保留在 L2 缓存中，从而减少对全局内存的访问，提高性能。 这并非直接编程缓存，而是通过 CUDA API 控制数据在 L2 缓存中的驻留时间。 你需要使用 `cudaMemPrefetchAsync()` 或类似函数来提示 CUDA 运行时将数据预先加载到 L2 缓存中。 


