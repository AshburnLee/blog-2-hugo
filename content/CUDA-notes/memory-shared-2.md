+++
date = '2025-08-31T12:45:52+08:00'
draft = false
title = 'Memory Shared 2'
tags = ["CUDA","Shared Memory","Bank"]
categories = ["CUDA"]
+++


## 为什么会冲突


假设:

- TILE_DIM = 32
- Shared Memory 有 32 个 Banks
- 每个 Bank 宽度为 4 字节 (float)
- 线程块维度为 (32, 32)，即每个线程块有 1024 个线程

我们声明一个共享内存数组：

~~~cpp
__shared__ float tile[TILE_DIM][TILE_DIM]; // __shared__ float tile[32][32];
~~~

现在，

### 如果线程配置是 `dim3 blockDim(1, 32)` 

这些线程的 `threadIdx.x` 都为 0，而 `threadIdx.y` 从 0 到 31，
通过 `int index = threadIdx.y + blockIdx.x * blockDim.y = threadIdx.y;` 得到线程id和其对应的tile元素索引: 

~~~cpp
// tile[threadIdx.x][threadsIdx.y]
Thread 0: tile[0][0]
Thread 1: tile[0][1]
Thread 2: tile[0][2]
...
Thread 31: tile[0][31]
~~~

由于 tile 数组是按行存储的，并且每个 float 占用 4 bytes，根据行存储公式： `Address = base_address + (row * num_cols + col) * element_size` 得到这些线程访问的地址如下：

~~~sh
Thread id=0: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (0 * 32 + 0) * 4B = 0
Thread id=1: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (0 * 32 + 1) * 4B = 4
Thread id=2: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (0 * 32 + 2) * 4B = 8
...
Thread id=31: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (0 * 32 + 31) * 4B = 124
~~~

因为每个 Bank 的宽度是 4 字节，所以：

Thread id=0 访问 Bank 0
Thread id=1 访问 Bank 1
Thread id=2 访问 Bank 2 
...
Thread id=31 访问 Bank 31

在这种情况下，**没有 Bank Conflict**，因为每个线程访问不同的 Bank。   DONE


但是，

### 如果线程配置是 `dim3 blockDim(32, 1)`

这些线程的 `threadIdx.y` 都为 0，而 `threadIdx.x` 从 0 到 31,
通过：`或者 int index = threadIdx.x + blockIdx.x * blockDim.x = threadIdx.x;` 得到线程id和其对应的tile元素索引：

~~~cpp
// tile[threadIdx.x][threadsIdx.y]
Thread 0: tile[0][0]
Thread 1: tile[1][0]
Thread 2: tile[2][0]
...
Thread 31: tile[31][0]
~~~

同理，得到，这些线程访问的地址如下：

~~~s
Thread id=0: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (0 * 32 + 0) * 4B = 0
Thread id=1: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (1 * 32 + 0) * 4B = 128
Thread id=2: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (2 * 32 + 0) * 4B = 256
...
Thread id=31: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (31 * 32 + 0) * 4B = 3968
~~~

这些地址值**都是 Bank0 的地址**，所以会有严重的 Bank conflict。


### 通过 Padding 避免 Bank Conflict 的情况

~~~cpp
__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // __shared__ float tile[32][33];
~~~

既然上面 `threadIdx.y` 都为 0 时，严重的conflict。一个 warp (32 个线程) 中的线程访问 tile 数组，假设这些线程的 threadIdx.y 都为 0，而 threadIdx.x 从 0 到 31：

~~~cpp
// tile[threadIdx.x][threadsIdx.y]
Thread 0: tile[0][0]
Thread 1: tile[1][0]
Thread 2: tile[2][0]
...
Thread 31: tile[31][0]
~~~

同理，得到，这些线程访问的地址如下：注意此时 `num_cols = 33`，  ***

~~~s
Thread id=0: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (0 * 33 + 0) * 4B = 0
Thread id=1: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (1 * 33 + 0) * 4B = 132
Thread id=2: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (2 * 33 + 0) * 4B = 264
...
Thread id=31: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (31 * 33 + 0) * 4B = 4092
~~~

现在，计算每个线程访问的 Bank：

`Bank = Address % (Number of Banks * Bank Width) / Bank Width`

即：

`Bank = Address % (32 * 4) / 4 = Address % 128 / 4`

~~~s
Thread 0: Bank = 0 % 128 / 4 = 0
Thread 1: Bank = 132 % 128 / 4 = 4 / 4 = 1
Thread 2: Bank = 264 % 128 / 4 = 8 / 4 = 2
...
Thread 31: Bank = 4092 % 128 / 4 = 124 / 4 = 31
~~~

现在，每个线程访问不同的 Bank，因此避免了 Bank Conflict！


### extra 如果线程配置是 `dim3 blockDim(32, 32)`

**在一个 Warp 中，线程的 threadIdx.x 是连续的，而 threadIdx.y 相同**。例如，第一个 Warp 的线程的 `threadIdx.x` 从 0 到 31，`threadIdx.y` 都为 0。

对于第一个 Warp 中的线程，其内存（内存访问模式）的地址和 上述 `dim3 blockDim(32, 1)` 情况一模一样。

这才是实际的情况，一定会有 Bank conflict。

