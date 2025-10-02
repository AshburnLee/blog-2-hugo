+++
date = '2025-01-31T12:45:13+08:00'
draft = false
title = 'Memory Shared：分析Bank conflict 与 Padding'
tags = ["CUDA","Shared Memory","Bank"]
categories = ["CUDA"]
+++


## 为什么会冲突

假设:

- TILE_DIM = 32
- （事实）Shared Memory 有 32 个 Banks
- （事实）每个 Bank 宽度为 4 字节 (float)
- 线程块维度为 (32, 32)，即每个线程块有 1024 个线程

声明一个共享内存数组：

~~~cpp
__shared__ float tile[TILE_DIM][TILE_DIM]; // __shared__ float tile[32][32];
~~~


## 假设线程配置是 `dim3 blockDim(32, 1)`

block x 方向 thread 有32个，**如果我的内存访问模式是** `float val = tile[threadIdx.x][threadsIdx.y]`，即 `float val = tile[threadIdx.x][0]`，那么所有 thread 访问第 0 列的元素。澄清一个点：

`tile[i][j]`,在 C++/CUDA 上下文中，i是行索引表示第 i 行，j 是列索引表示第 j 列，所以：

  - `tile[0][0]` 指访问 第 0 行，第 0 列元素
  - `tile[1][0]` 指访问 第 1 行，第 0 列元素
  - `tile[2][0]` 指访问 第 2 行，第 0 列元素
  - ...
  - `tile[31][0]` 指访问 第 31 行，第 0 列元素

这些元素都在第0列。

【计算二维数组在一维内存中的地址】
【是否 Bank 冲突】
【通过 padding 处理 bank 冲突】

这些线程的 `threadIdx.y` 都为 0，而 `threadIdx.x` 从 0 到 31。通过 `int index = threadIdx.x + blockIdx.x * blockDim.x = threadIdx.x;` 得到线程 id 和其对应的 tile 元素索引：

~~~cpp
// tile[threadIdx.x][threadsIdx.y]
Thread 0: tile[0][0]
Thread 1: tile[1][0]
Thread 2: tile[2][0]
...
Thread 31: tile[31][0]
~~~

得到这些线程访问的**字节地址**如下：

~~~s
Thread id=0: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (0 * 32 + 0) * 4B = 0
Thread id=1: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (1 * 32 + 0) * 4B = 128
Thread id=2: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (2 * 32 + 0) * 4B = 256
...
Thread id=31: Address = 0 + (threadIdx.x * 32 + threadIdx.y) * 4B => 0 + (31 * 32 + 0) * 4B = 3968
~~~

对应的 Bank ID是：

~~~
Bank ID: 0//4%32 = 0
Bank ID: 128//4%32 = 0
Bank ID: 256//4%32 = 0
Bank ID: 3968//4%32 = 0
~~~

**这些地址值都是 Bank0 的地址**，所以会有严重的 32-way Bank conflict。


## 通过 Padding 避免 Bank Conflict 的情况

~~~cpp
__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // __shared__ float tile[32][33];
~~~

既然上面 `threadIdx.y` 都为 0 时，严重的 conflict。一个 warp (32 个线程) 中的线程访问 tile 数组，假设这些线程的 threadIdx.y 都为 0，而 threadIdx.x 从 0 到 31：

~~~cpp
// tile[threadIdx.x][threadsIdx.y]
Thread 0: tile[0][0]
Thread 1: tile[1][0]
Thread 2: tile[2][0]
...
Thread 31: tile[31][0]
~~~

同理，得到，这些线程访问的字节地址如下：注意此时 `num_cols = 33`，  ***

~~~s
Thread id=0: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (0 * 33 + 0) * 4B = 0
Thread id=1: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (1 * 33 + 0) * 4B = 132
Thread id=2: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (2 * 33 + 0) * 4B = 264
...
Thread id=31: Address = 0 + (threadIdx.x * 33 + threadIdx.y) * 4B => 0 + (31 * 33 + 0) * 4B = 4092
~~~

根据 `bank_id = (字节地址 // 4​) mod 32` 得到对应的 Bank ID 是：

~~~
Bank ID: 0//4%32 = 0
Bank ID: 132//4%32 = 1
Bank ID: 264//4%32 = 2
...
Bank ID: 4092//4%32 = 31
~~~

现在，这个 case 中 32 个 threads 每个线程访问不同的 Bank，因此避免了 Bank Conflict


## extra 如果线程配置是 `dim3 blockDim(32, 32)`

**在一个 Warp 中，线程的 threadIdx.x 是连续的，而 threadIdx.y 相同**。例如，第一个 Warp 的线程的 `threadIdx.x` 从 0 到 31，`threadIdx.y` 都为 0。

对于第一个 Warp 中的线程，其内存（内存访问模式）的地址和 上述 `dim3 blockDim(32, 1)` 情况一模一样。

这才是实际的情况，一定会有 Bank conflict。


## 对于 `__shared__ float tile[31][31]` 是否有 Bank Conflict？

线程配置仍为 blockDim(32, 1)，只使用到31个threads。还是访问模式：`tile[threadIdx.x][0]`。

首先计算字节地址：`tile[i][j] = 0 + (i * 31 + j) * 4Byte`

~~~s
tile[0][0] = 0 + (0 * 31 + 0) * 4B = 0
tile[1][0] = 0 + (1 * 31 + 0) * 4B = 124
tile[2][0] = 0 + (2 * 31 + 0) * 4B = 248
tile[3][0] = 0 + (3 * 31 + 0) * 4B = 372
...
tile[30][0] = 0 + (30 * 31 + 0) * 4B = 3720
tile[31][0] = 0 + (31 * 31 + 0) * 4B = 3964
~~~

然后计算 Bank ID （`(字节地址 // 4) mod 32`）：

~~~s
Bank ID = 0//4%32 = 0
Bank ID = 124//4%32 = 31
Bank ID = 248//4%32 = 30
Bank ID = 372//4%32 = 29
...
Bank ID = 3720//4%32 = 2
Bank ID = 3964//4%32 = 1
~~~

结论是 31 个线程访问 31 个不同 bank，所以无冲突。（31 与 32 **互质**）

~~~s
0 + (0 * 31 + 0) % 32 = 0
0 + (1 * 31 + 0) % 32 = 31
0 + (2 * 31 + 0) % 32 = 30
...
0 + (30 * 31 + 0) % 32 = 2
0 + (31 * 31 + 0) % 32 = 1
~~~


## 上述两步合并后可以将 位宽 4 Bytes 抵消 

即线程全局 `Bank ID = Global-ID % 32` 

