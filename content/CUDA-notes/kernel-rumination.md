+++
date = '2025-09-17T23:13:13+08:00'
draft = false
title = 'Kernel Rumination'
tags = ["CUDA"]
categories = ["CUDA"]
+++


## 一个block中的threadIdx 只在这个block中有效

元素个数 size=4096，kernel launch 配置 blocksPerGrid=(32,1,1)，threadsPerBlock=(128,1,1)，总线程数 32 × 128 = 4096。

~~~cpp
__global__ void reduce(float* d_input, float* d_output, int size) {
    // 每个线程处理一个元素计算全局 thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shared_data[];  // 动态分配共享内存，大小会是 thread per block * sizeof(float)

    // 1. 计算每个 block 中的局部最大值，所以与blockIdx 无关，这里的scope是一个block
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (i < size && d_input[i] > max_val) max_val = d_input[i];
    }
    shared_data[threadIdx.x] = max_val;
    __syncthreads();    
    ...
}

~~~

~~上述 code 的目的是将 block 对应的 128 个元素写入这个 block 的 shared memory。而且 `i += blockDim.x` 永远不会发生，因为 这样做thread的ID 就超过了 0~127 ，所以这个for循环并不会循环，他是多于的。~~

~~循环让线程访问超出 block 范围的数据。threadIdx.x 在这个case下是 0~127，i 不会大于127。~~

上述表达错误！threadIdx.x 不会超过127，但是 i 可以！所以这里的逻辑是正确的（但是冗余）。每个 block 的 128 个线程通过 stride （blockDim.x）访问了所有 4096 个元素，这个 block 内的归约结果实际上是全局总和。


code应该改为：

~~~cpp
__global__ void reduce(float* d_input, float* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shared_data[];  // 动态分配共享内存，大小会是 thread per block * sizeof(float)
    // 每个 block 将对应元素写入自己的 shared memory
    float max_val = (idx < size) ? d_input[idx] : -INFINITY;
    shared_data[threadIdx.x] = max_val;
    __syncthreads(); 
    ...
}
~~~

诶，改成这样就错了，计算得到的 max_val 是block内的值，而非全局max。


