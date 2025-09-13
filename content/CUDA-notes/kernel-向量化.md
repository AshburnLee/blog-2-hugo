+++
date = '2025-01-31T12:45:08+08:00'
draft = false
title = 'Kernel 向量化'
tags = ["CUDA","Vectorization"]
categories = ["CUDA"]
+++


## 向量化数据类型  [cuda 内置类型]

CUDA 提供的向量类型，用于表示多个相同类型的值的集合。 它们允许你一次性操作多个数据元素，从而提高性能，特别是对于并行计算。

- `int2`: 包含两个 int 类型的值。
- `int4`: 包含四个 int 类型的值。
- `float2`: 包含两个 float 类型的值。

这些类型并非标准 C++ 的一部分，而是 CUDA 为了优化 GPU 计算而定义的。它们允许编译器生成更有效的指令，例如向量化加载和存储指令 (`LD.E.64`, `ST.E.64`, `LD.E.128`, `ST.E.128`)，从而提高内存带宽利用率并减少指令数量。


## 向量化 kernel

更充分利用带宽，减少指令数。向量化的实现需要数据对齐。

本质上，它将两个 int 元素的加载和存储操作合并为一个操作，从而减少了指令的数量并提高了带宽利用率。

非向量化：

~~~cpp
__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  for (int i = idx; i < N; i += blockDim.x * gridDim.x){ 
    d_out[i] = d_in[i]; 
  } 
} 

void device_copy_scalar(int* d_in, int* d_out, int N) { 
  int threads = 128; 
  int blocks = min((N + threads-1) / threads, MAX_BLOCKS);  
  device_copy_scalar_kernel<<<blocks, threads>>>(d_in, d_out, N); 
}
~~~

使用 `int2`：

~~~cpp
__global__ void device_copy_vector2_kernel(int* d_in, int* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 线程循环次数减少了一半,指令执行也减少到1/2
  for (int i = idx; i < N/2; i += blockDim.x * gridDim.x) {  
    reinterpret_cast<int2*>(d_out)[i] = reinterpret_cast<int2*>(d_in)[i];
  }

  // in only one thread, process final element (if there is one)
  if (idx==N/2 && N%2==1)
    d_out[N-1] = d_in[N-1];
}

void device_copy_vector2(int* d_in, int* d_out, int n) {
  threads = 128; 
  blocks = min((N/2 + threads-1) / threads, MAX_BLOCKS); 
  device_copy_vector2_kernel<<<blocks, threads>>>(d_in, d_out, N);
}
~~~

使用 `int4`：

~~~cpp
__global__ void device_copy_vector4_kernel(int* d_in, int* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 线程循环次数减少到原来的1/4，指令执行也减少到1/4
  for(int i = idx; i < N/4; i += blockDim.x * gridDim.x) {  
    reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
  }

  // process final elements (if there are any)
  int remainder = N%4;
  if (idx==N/4 && remainder!=0) {
    while(remainder) {
      int idx = N - remainder--;
      d_out[idx] = d_in[idx];
    }
  }
}

void device_copy_vector4(int* d_in, int* d_out, int N) {
  int threads = 128;
  int blocks = min((N/4 + threads-1) / threads, MAX_BLOCKS);
  device_copy_vector4_kernel<<<blocks, threads>>>(d_in, d_out, N);
}
~~~


### 向量化会增加寄存器使用量，为什么？

int 占用 4 个字节，而 int2 占用 8 个字节（两个 int）。 每个线程需要存储从内存加载的数据。当使用 int 时，线程只需要一个寄存器来存储一个整数。但当使用 int2 时，线程需要一个能够容纳 8 个字节数据的寄存器（或者两个 4 字节的寄存器，取决于编译器和硬件的具体实现），来存储两个整数。所以每个线程每一次的执行使用的寄存器数量增加了。

向量化加载提高了吞吐量，但需要更多的寄存器来存储加载的更多数据。这是一种**空间换时间的权衡**。


### 编译器会自动每次处理 4 个数

int4 类型明确告诉编译器，每次操作的是 4 个 int 元素。**编译器会尝试利用 SIMD** (Single Instruction, Multiple Data) 指令，将 4 个元素的加载、存储和计算并行化。 背后是 GPU 硬件通常对向量化操作有很好的支持。 ***

CPU 在处理对齐的数据时，也会是每次处理多个，因为CPU中也有SIMD指令集。
