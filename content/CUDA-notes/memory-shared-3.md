+++
date = '2025-01-31T12:45:14+08:00'
draft = false
title = 'Memory Shared 3'
tags = ["CUDA","Shared Memory","Bank"]
categories = ["CUDA"]
+++


## 读代码

~~~cpp
// 有 Bank Conflict 的 Kernel
__global__ void kernelWithBankConflict(float *input, float *output) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    tile[threadIdx.x][threadIdx.y] = input[x * TILE_DIM + y];
    __syncthreads();
    output[x * TILE_DIM + y] = tile[threadIdx.x][threadIdx.y];
}

// 避免 Bank Conflict 的 Kernel
__global__ void kernelWithoutBankConflict(float *input, float *output) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    tile[threadIdx.x][threadIdx.y] = input[x * TILE_DIM + y];
    __syncthreads();
    output[x * TILE_DIM + y] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    ...
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(1, 1);
    // 执行 Kernel (有 Bank Conflict)
    kernelWithBankConflict<<<gridDim, blockDim>>>(d_input, d_output);

    // 执行 Kernel (避免 Bank Conflict)
    kernelWithoutBankConflict<<<gridDim, blockDim>>>(d_input, d_output);
    ...
}
~~~


`input[x * TILE_DIM + y];` : x，y 表示每个线程自己的全局索引，`x * TILE_DIM` 表示目标位置所在的行，`+ y` 表示目标位置所在的列，所以行偏移后，列偏移，就得到了目标位置index。

## 既然在同一个 GPU 上共享内存的 Bank 宽度是 4 字节，那么处理 float (4 字节) 和 double (8 字节) 数据类型时，需要考虑如何有效地访问共享内存，以减少 Bank conflict

通过padding 构建新的数据类型：

~~~cpp
struct DoubleData {
    double value;
    char padding[4]; // 添加 4 字节 padding，确保对齐
};
~~~

这里的 padding 并不是为了让 double 类型本身对齐（因为 double 本身通常会按照 8 字节对齐），而是**为了避免不同线程访问共享内存时发生 Bank conflict。**

上述，Bank 宽 4 bytes的组织方式，如果double数据不 padding，会导致 Bank conflict 如下：

~~~sh
线程 0 访问 shared_double[0].value (地址 0) -> Bank 0
线程 1 访问 shared_double[1].value (地址 8) -> Bank 2
线程 2 访问 shared_double[2].value (地址 16) -> Bank 4
...
线程 7 访问 shared_double[7].value (地址 56) -> Bank 14
线程 8 访问 shared_double[8].value (地址 64) -> Bank 16
...
线程 15 访问 shared_double[15].value (地址 120) -> Bank 30
线程 16 访问 shared_double[16].value (地址 128) -> Bank 0 <-- 与线程 0 冲突  ***
~~~


padding padding 的主要作用是改变线程访问共享内存时，数据到 Bank 的映射关系。

padding之后，？？存疑，实际上会怎么做？？
