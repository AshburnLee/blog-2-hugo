+++
date = '2025-08-31T12:45:52+08:00'
draft = false
title = 'Memory Shared 1'
tags = ["CUDA","Shared Memory","Bank"]
categories = ["CUDA"]
+++


## Shared Memory

- 片上内存，极低延迟
- 是每个线程块 (Thread Block) 独有的；
- 生命周期与创建它的线程块相同；当线程块执行完毕后，共享内存中的数据也会被释放。
- 用于实现高性能的协作并行算法，例如并行归约。
- 用于手动管理的数据缓存，减少对全局内存的访问。比如通过 Shared Mem 实现 reverse 一个数组。
- 共享内存可以静态分配（在编译时指定大小）或动态分配（在运行时指定大小）；
- 每个 SM 都有，且是有限的共享内存容量；
- 注意 Bank conflict
- 在共享内存中进行读写操作时，通常需要使用 `__syncthreads()` 函数进行线程同步。
- 每个线程块可用的共享内存量是有限的。可以使用 `cudaGetDeviceProperties` 函数来查询设备的共享内存大小。


## 分配 Shared Memory

静态分配：适用于共享内存大小在运行时保持不变的情况，在 kernel 中固定大小 `vectorAddStatic<<<blocksPerGrid, threadsPerBlock>>>()` 。它更简单，并且编译器可以进行更好的优化。

动态分配：适用于共享内存大小可能在运行时变化的情况，在启动 kernel 时 给出 `vectorAddDynamic<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>()`。

更多使用 Shared Memory 的实例：...


## Bank

CUDA 共享内存被划分为多个内存 Bank，**每个 Bank 在一时钟周期内只能处理一个内存请求**。如果多个线程试图同时访问同一个 Bank 中的不同地址，就会发生 Bank 冲突 (Bank conflict)。这会导致内存访问串行化，降低性能。

避免 Bank 冲突的策略:

- 内存对齐: 确保线程访问的内存地址在不同的 Bank 中。
- 访问模式: 避免多个线程同时访问同一个 Bank 。 例如，如果共享内存是一个二维数组，则应避免所有线程同时访问同一列（或同一行，取决于内存布局）。
- 数据布局: 选择合适的数据布局，例如将数据按 Bank 进行排列，以最大限度地减少冲突。


- 错位访问：通过在共享内存的声明中添加一个小的偏移量，可以避免 Bank conflict。
- 交错访问：通过交错访问共享内存，可以避免 Bank conflict。
- 数据对齐：确保线程访问的数据在共享内存中对齐到 Bank 宽度。例如，如果每个 Bank 宽度为 4bytes，则确保每次访问的地址是4的倍数。



## Block 中使用 Shared Memory

一个 block 中使用的 Shared Memory 不一定是单维的。虽然 Shared Memory 的物理布局是线性的，但你可以将其逻辑地组织成多维数组。

1. 静态分配：

    ~~~cpp
    __global__ void myKernel() {
      __shared__ float matrix[16][16]; // 16x16 的二维数组

      // ... 使用 matrix ...
    }
    ~~~

2. 动态分配：

    使用 `extern __shared__` 声明 Shared Memory 时，你可以动态分配内存:

    ~~~cpp
    __global__ void myKernel(int size) {
      extern __shared__ float shared_mem[];

      float *matrix = shared_mem; // 将 shared_mem 视为一个指针

      // ... 使用 matrix 作为二维数组 ...  需要手动计算索引
    }
    ~~~

重要考虑:

- 性能: 无论你如何逻辑地组织 Shared Memory，其底层**物理布局始终是线性**的。 因此，访问模式对性能至关重要。 为了最大限度地提高性能，应尽量避免 Bank conflict（银行冲突）。 这通常意味着访问元素时应尽量保持内存访问的连续性。 多维数组的访问模式需要仔细设计，以避免 Bank conflict。
- 索引计算: 对于动态分配的 Shared Memory，你需要自己负责索引计算，这可能会增加代码的复杂性。
- 大小限制: 每个 block 可用的 Shared Memory 总量是有限的。 你需要根据你的 GPU 架构和 block 大小来选择合适的多维数组大小。


## Bank 的组织方式

- Bank 的宽度是硬件决定的，无法通过软件配置进行更改。
- 同一个 Bank 中的相邻地址是**不连续的**。
- 同一个 Bank 中的相邻地址之间相差 32 * 宽（x Byte）
- 0, 32, 64, 96, ... 这些数字表示的是**共享内存地址的编号**。在共享内存中，每个字节都有一个唯一的地址编号。这些地址编号从 0 开始，依次递增。
- Shared Memory 的地址编号是从 0 开始，**每个字节都有一个唯一的编号**。***

### 如下是 32 个 Bank，每个 Bank 宽 1Bytes（8位）

~~~sh
Bank 0: 0, 32, 64, 96, 128, 160, 192, 224, ...
Bank 1: 1, 33, 65, 97, 129, 161, 193, 225, ...
Bank 2: 2, 34, 66, 98, 130, 162, 194, 226, ...
...
Bank 31: 31, 63, 95, 127, 159, 191, 223, 255, ...
~~~


### 如下是 32 个 Bank，每个 Bank 宽 4Bytes（32位）这是一般情况

~~~sh
Bank 0: 0, 128, 256, ...
Bank 1: 4, 132, 260, ...
Bank 2: 8, 136, 264, ...
...
Bank 31: 124, 252, 380, ...
~~~

## Bank 冲突

同一个 warp 中多个线程同时访问同一个 Bank 中的不同地址，会导致 Bank conflict。

不同 warp 中的线程可以独立访问共享内存，而不会导致 Bank conflict。这是因为 warp 调度器会分别处理每个 warp 的内存访问请求。

### case1: 一个 Warp 中的 32 个线程尝试访问 Shared Memory 中的以下地址：

~~~sh
Thread 0: addr 0
Thread 1: addr 4
Thread 2: addr 8
...
Thread 31: addr 124
~~~

在这个例子中，每个线程访问的**地址都位于不同的 Bank 中**，因此**不会发生** Bank conflict。


### case2: 如果一个 Warp 中的 32 个线程尝试访问 Shared Memory 中的以下地址：

~~~sh
Thread 0: Address 0
Thread 1: Address 0
Thread 2: Address 0
...
Thread 31: Address 0
~~~

所有线程都访问同一个地址，这**不会导致** Bank conflict，因为硬件会将数据**广播**给所有线程.


### case3: 如果一个 Warp 中的 32 个线程尝试访问 Shared Memory 中的以下地址：

~~~sh
Thread 0: Address 0
Thread 1: Address 32
Thread 2: Address 64
...
Thread 31: Address 992
~~~

所有线程都尝试访问 Bank 0，这会**导致严重的** Bank conflict，因为这些访问必须串行化。

## 二维数组中元素在内存中的地址计算，行优先

在行优先存储中，数组的元素按照以下顺序存储：

~~~sh
a[0][0], a[0][1], a[0][2], ..., a[0][num_cols-1],
a[1][0], a[1][1], a[1][2], ..., a[1][num_cols-1],
a[2][0], a[2][1], a[2][2], ..., a[2][num_cols-1],
...
a[num_rows-1][0], a[num_rows-1][1], a[num_rows-1][2], ..., a[num_rows-1][num_cols-1]
~~~

假设有一个二维数组 `a[num_rows][num_cols]`，我们要计算元素 `a[row][col]` 在内存中的地址。推导如下：

1. 行偏移量：要找到 `a[row][col]` 的地址，我们首先需要跳过 `row` 行。每一行有 `num_cols` 个元素，所以我们需要跳过 `row * num_cols` 个元素。
2. 列偏移量：在跳过 `row` 行后，我们需要在当前行中跳过 `col` 个元素。
3. 总偏移量：总共需要跳过 `row * num_cols + col` 个元素。
4. 地址计算：由于每个元素占用 `element_size` 个字节，所以总偏移量需要乘以 `element_size`。最后，加上数组的起始地址 `base_address`，就可以得到元素 `a[row][col]` 在内存中的地址：

`Address = base_address + (row * num_cols + col) * element_size`

`base_address`：表示数组在内存中的起始位置。

