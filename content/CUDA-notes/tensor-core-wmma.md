+++
date = '2025-01-31T12:45:53+08:00'
draft = false
title = 'Tensor Core Wmma'
tags = ["CUDA","Tensor Core"]
categories = ["CUDA"]
+++

## 阅读以下链接，在 Orin 上执行其中的实例

https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/

## WMMA API 会与 shared memory 交互

wmma api 主要与寄存器交互，但在实际使用中也可能与共享内存交互，以优化数据访问和计算效率。

## wmma api 在与 shared memory 交互时，如何避免 bank conflict？

## Tensor Core 编程模型

Tensor Core 的编程模型与传统的 CUDA 编程模型有所不同。在 CUDA 中，每个线程的数据通常是私有的，但在 Tensor Core 编程中，需要考虑 warp 级别的操作，因为 Tensor Core 是在 warp 内部进行操作的（所以是 `Warp MMA`）。

- Warp 级别编程：Tensor Core 的操作通常在 warp 级别进行。一个 warp 中的多个线程协同工作，共同完成一个矩阵乘法操作。
- 混合精度：Tensor Core 支持混合精度计算，通常输入数据是半精度（FP16），而累加器使用单精度（FP32）或半精度。
- 显式数据加载和存储：需要显式地将数据加载到 Tensor Core 的寄存器中，进行计算，然后将结果写回内存。这通常涉及到使用特定的 WMMA API。
- 数据分块：由于 Tensor Core 处理的是小块矩阵，因此需要将大矩阵分成小块，然后分批加载到 Tensor Core 中进行计算。


## 编程模型

1. 数据划分：
    将输入矩阵 A 和 B 划分为小的 `wmma::fragment` (子矩阵)。
    每个 warp 中的线程负责处理一个或多个 `wmma::fragment`。

2. 数据加载：
    使用 `wmma::load_matrix_sync` 函数将 wmma::fragment 从全局内存或共享内存加载到 warp 中每个线程的寄存器中。
    load_matrix_sync 确保 warp 中的所有线程都同步加载数据。

3. 矩阵乘法累加：
    使用 `wmma::mma_sync` 函数执行矩阵乘法和累加操作。
    mma_sync 函数利用 Tensor Cores 在 warp 内并行执行计算。
    计算公式：`D = A * B + C`，其中 A、B、C 和 D 都是 `wmma::fragment`。

4. 数据存储：
    使用 `wmma::store_matrix_sync` 函数将计算结果 `wmma::fragment` 从寄存器存储到全局内存或共享内存中。
    `store_matrix_sync` 确保 warp 中的所有线程都同步存储数据。

## tensor core 计算 16x16 的tile

那么 warp 中 thread 编号 0 到 31，如何对应上 16x16 数据tile ？

答：在WMMA API中，一个 warp 的32个线程共同完成 16x16 矩阵的计算。具体来说，每个线程负责处理矩阵的一部分数据。WMMA API通过内部的数据结构和指令来管理线程如何访问和处理数据，而不是显式地将线程 ID 映射到特定的数据元素。

然而，在实现细节上，WMMA API 通常将 warp 内的线程分为更小的组，每个组负责处理矩阵乘法的一个部分。

## nvcuda::wmma

`mma.h` 命名空间下的 `nvcuda::wmma` 头文件中的内容是 NVIDIA 提供的用于在 CUDA 中使用 Tensor Core 的 API。它包括了执行矩阵乘法（MMA，Matrix Multiply-Accumulate）操作的函数和类型定义。


## wmma::fragment 如何与硬件对应

~~~cpp
wmma::fragment(typename MatrixT,   // matrix_a 或 matrix_b 或 accumulator
               int BlockM,         // 对应于输出矩阵 C 的行数
               int BlockN,         // 对应于输出矩阵 C 的列数
               int BlockK,         // 对应 A 矩阵的列数和 B 矩阵的行数
               typename T,         // 这个分片的数据类型
               typename Layout = wmma::default_layout>)  // 指定分片的数据排布
~~~

`wmma::fragment` 与 Tensor Core 硬件的对应关系如下：

  - 尺寸匹配：`wmma::fragment` 的尺寸（例如，M, N, K）必须与 Tensor Core 硬件所支持的矩阵尺寸匹配。不同的 GPU 架构支持不同的矩阵尺寸。例如，Volta 架构的 Tensor Core 支持 `4x4x4` 的矩阵乘法，而 Ampere 架构支持更大的尺寸（16x16x16）。
  - 数据类型匹配：`wmma::fragment` 的数据类型必须与 Tensor Core 硬件所支持的数据类型匹配。通常，输入矩阵 A 和 B 使用半精度（FP16），而累加器矩阵 C 和 D 可以使用半精度或单精度（FP32）。
  - 布局匹配：`wmma::fragment` 的矩阵布局（`row_major` 或 `col_major`）必须与 Tensor Core 硬件所期望的布局匹配。这会影响数据加载和存储的方式。比如 你为一个 wmma::fragment 指定 row_major 布局时，Tensor Core 会按照行优先的顺序**来解释**和处理该分片中的数据。
  - 硬件指令：WMMA API 提供了一组硬件加速指令，用于加载矩阵片段（`wmma::load_matrix_sync`），执行矩阵乘法（`wmma::mma_sync`），和存储矩阵片段（`wmma::store_matrix_sync`）。这些指令直接在 Tensor Core 硬件上执行，从而实现高性能的矩阵乘法。
  - fragment 被加载到 warp 的线程寄存器中。

这个函数相当于是分配内存


## nvcuda::wmma::load_matrix_sync(a_frag, A_global_row_major, lda); 

  - a_frag: 从全局加载的子矩阵
  - 这是一个指向全局内存中矩阵 A 的指针。load_matrix_sync 函数会从这个指针指向的内存地址开始读取数据。
  - 所有线程在一个 warp 中协同工作，将需要由张量核心计算的 tile 加载。我们只需要将 warp 指向 tile 的**左上角元素**。***
  - 第三个参数 lda 表示的是 leading dimension，也就是 主维度。


## Leading Dimension 的含义

Leading dimension 指的是在内存中，矩阵的行（或列）之间实际间隔的元素数量。更具体地说：

  - 对于行优先 (row-major) 矩阵：lda 表示矩阵中相邻两行起始地址之间的元素数量。如果矩阵是紧密排列的，那么 lda 等于矩阵的列数。但如果矩阵是子矩阵或经过填充，lda 可能大于实际的列数。
  - 对于列优先 (col-major) 矩阵：lda 表示矩阵中相邻两列起始地址之间的元素数量。如果矩阵是紧密排列的，那么 lda 等于矩阵的行数。但如果矩阵是子矩阵或经过填充，lda 可能大于实际的行数。


## 为什么需要 Leading Dimension

  - 处理子矩阵：当您需要从一个大矩阵中提取一个子矩阵进行计算时，子矩阵在内存中可能不是连续存储的。lda 可以帮助 `load_matrix_sync` 函数正确地定位子矩阵的元素。
  - 处理填充矩阵：在某些情况下，为了提高内存对齐或避免 bank conflict，矩阵可能会被填充一些额外的元素。lda 可以帮助 `load_matrix_sync` 函数跳过这些填充元素。
  - 通用性：通过使用 `lda`, `load_matrix_sync` 函数可以处理各种不同存储方式的矩阵，而不仅仅是紧密排列的矩阵。


## nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag) 行为是什么

得到 `a_frag` `b_frag`，并且 用0 初始化 `acc_frag`。`mma_sync` 函数在一个 warp 中的所有线程上**同步执行**。同步意味着在 `mma_sync` 函数返回之前，warp 中的所有线程都必须完成矩阵乘法和累加操作。

输出是 `acc_frag` ，其中存储着更新后的累加结果。这个`acc_frag` 其实是thread registers



## Tensor core 上的矩阵乘法 都是小尺寸的？大矩阵如何处理的？

虽然 Tensor Core 本身处理的是小尺寸的矩阵，但通过合理的分块策略和优化手段，可以在实际场景中高效地处理非常大的矩阵。关键在于将大矩阵分解成小块，并充分利用共享内存和数据预取等技术，以减少全局内存的访问次数，提高计算效率。

具体实战，见上述链接中的 1024x1024 的实例。

~~~sh
FP16 HMMA
C = A^T * B^T  HMMA Latency: 7.489 ms
C = A^T * B    HMMA Latency: 7.066 ms
C = A * B^T    HMMA Latency: 7.927 ms
C = A * B      HMMA Latency: 7.486 ms
~~~

`C = A^T * B` 是最快的， `C = A * B^T` 是最慢的。


## 编译命令

`nvcc -I/home/junhui/workspace/operators_plain/include ./src/mma.cu -o ./outs/mma --gpu-architecture=compute_87`

最后的选项作用是，编译器会根据指定的计算能力，生成相应的机器码。


## 性能指标
### 吞吐

尽管时间是一个分析时非常合适的指标，但更好的选择是查看函数每秒执行的操作数，即**每秒执行浮点运算次数（GFLOPS）**。

当乘以两个 M x K 和 K x N 矩阵时，每个输出矩阵元素大约需要 K 次乘法和 K 次加法，即 2K 次操作。由于总共有 M x N 个输出元素，所以总操作数是 2 x M x N x K。**将这个数字除以执行矩阵乘法所需的时间**，可以得到实现算法的 FLOPS（可以转换为 GFLOPS）。

### 并行程度
### 资源利用率
