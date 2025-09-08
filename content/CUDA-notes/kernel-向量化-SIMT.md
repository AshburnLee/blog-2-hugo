+++
date = '2025-08-31T12:45:51+08:00'
draft = false
title = 'Kernel 向量化 SIMT'
tags = ["CUDA","SIMD","SIMT"]
categories = ["CUDA"]
+++


NVIDIA GPU 架构是 SIMT，而编译器又会利用 SIMD 指令来优化 int4 类型的操作，这看起来似乎有些矛盾，但实际上它们并不冲突。

## SIMT 和 SIMD 的关系

- SIMT 是**架构层面**：SIMT 描述的是 NVIDIA GPU 的整体架构和执行模型。它指的是多个线程（以 warp 为单位）执行相同的指令，但处理不同的数据。

- SIMD 是**指令层面**：SIMD 是一种指令集，它允许一条指令同时操作多个数据。编译器可以使用 SIMD 指令来优化代码，提高程序的性能。

所以 NV GPU 是 SIMT 架构的：决定了它的基本执行方式：多个线程（warp）执行相同的指令。同时又有 SIMD 优化的：在 SIMT 架构下，编译器仍然可以利用 SIMD 指令来优化代码。例如，对于 int4 类型的操作，编译器可以使用 SIMD 指令一次性加载、存储和计算 4 个整数。

SIMT 是 NVIDIA GPU 的专有模型，依赖 SM 和 warp 调度。Multiple Thread 体现在Warp内线程异步执行相同指令：

~~~cpp
__global__ void add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程独立索引
    if (idx < n) c[idx] = a[idx] + b[idx];  // SIMT：warp 内线程并行执行加法
}
~~~

