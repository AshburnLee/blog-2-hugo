+++
date = '2025-08-31T12:45:50+08:00'
draft = false
title = 'Cublas Mma'
tags = ["CUDA","cuBLAS"]
categories = ["CUDA"]
+++



cuBLAS 也会利用 tensor core，其性能比简单的手工写的 wmma 更优。

https://0mean1sigma.com/tgemm/  这篇文章中的 benchmark 展示了不同方式的性能。可以把最高性能的作为标杆，自己动手优化，这里的性能指标是吞吐，还可以有其他指标 *** 

## 什么是 leading demension

内存中连续存储元素的个数。

给定一个形状为 MxN 的矩阵 ，如果它按 Row-major 存储，其 leading dimension 是 N，如果它按 Col-major 存储，其 leading dimention 是 M。


## 什么是 row-major 和 col-major

row-major 指的是按行存储，即每一行中元素是连续存储的。
col-major 指的是按列存储，即每一列中元素是连续存储的。

~~~txt
A = 1 2 3
    4 5 6
~~~

在 row-major order 中，内存中的存储顺序是：1 2 3 4 5 6, leading dimension 是 3.
在 col-major order 中，内存中的存储顺序是：1 4 2 5 3 6, leading dimension 是 2.

~~~txt
A^T = 1 4
      2 5
      3 6
~~~

可以看出在内存中， A `row-major` 存储和 A^T `col-major` 存储是一样的。对于 row-major 存储的矩阵，**读取 A 行和读取 A^T 的列是快速且缓存友好**的。读取 A 列和读取 A^T 的行是慢的且缓存不友好的。


## 矩阵

MMA 有以下4种形式，并且根据内存中存储矩阵的方式，不同的形式有不同的最优访存方式（访存偏好）：

1. `C = A * B`            A row-major，B col-major 的存储方式最高效
2. `C = A^T * B`          A col-major，B col-major 的存储方式最高效
3. `C = A * B^T`          A row-major，B row-major 的存储方式最高效
4. `C = A^T * B^T`        A col-major，B row-major 的存储方式最高效


通常一个软件框架中的所有矩阵都会使用相同的存储顺序，这意味着，根据软件存储矩阵的方式，**只有 2、3 种形式是最优的**。比如 `cublasSgemm` 库中，A 和 B 都必须是 col-major 的存储方式。

此外，软件中，在物理上进行一次转置操作，不是一个好主意，**矩阵转置本就有很大的开销**。

`cublasSgemm` 函数提供了 `transa` 和 `transb` 参数，用于指定是否对输入矩阵 A 和 B 进行转置。通过合理设置这两个参数，你可以在**不实际转换矩阵存储顺序**的情况下，使用 `cublasSgemm` 函数进行计算。


## `cublasSgemm` 接口

~~~cpp
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, lda,
            d_B, ldb,
            &beta,
            d_C, ldc);
~~~

- `CUBLAS_OP_T`: 表示对 A 进行转置
- `CUBLAS_OP_N`: 表示不对 B 进行转置
- `CUBLAS_OP_T` 的实现方式通常是通过**交换索引的顺序**来实现逻辑转置，而不是通过实际的内存拷贝。


## Benchmark

在 cpu （ARM cortex）上的 MMA 测试结果：【./outs/cpu_mm】

~~~sh
C = A * B      Latency: 99.600 ms
C = A^T * B    Latency: 87.900 ms  # 最优 (A row-major, B row-major)
C = A * B^T    Latency: 117.700 ms # 最差
C = A^T * B^T  Latency: 102.400 ms
~~~

## 使用 cuBLAS 库进行4中矩阵乘法

四种方法性能没有差别：【./outs/cublas_mm】

~~~sh
C = A * B     Latency: 0.137 ms
C = A^T * B   Latency: 0.152 ms  # 最差 (A col-major, B col-major)
C = A * B^T   Latency: 0.129 ms  # 最优
C = A^T * B^T Latency: 0.148 ms
~~~

TODO：这里的测试结果和预期相反，


## cuBLAS 编程模型

