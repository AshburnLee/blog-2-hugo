+++
date = '2025-01-31T12:45:12+08:00'
draft = false
title = 'Memory Shared：字节地址与 Bank ID'
tags = ["CUDA","Shared Memory","Bank"]
categories = ["CUDA"]
+++


- Bank x-way conflict
- 先计算字节地址
- 后字节地址与 Bank id 的对应关系（用于判断thread 访问的Bank id）
- （上述两步骤可以合并为一个公式，消除位宽）
- 二维数组在一维内存中的地址计算


## Shared Memory

- 片上内存，极低延迟。与L1在一起，
- 是每个 Thread Block 独有的；
- 生命周期与创建它的线程块相同；当线程块执行完毕后，共享内存中的数据也会被释放。
- 用于实现高性能的协作并行算法，例如并行归约。
- 用于手动管理的数据缓存，减少对全局内存的访问。比如通过 Shared Mem 实现 reverse 一个数组。
- 共享内存可以静态分配（在编译时指定大小）或动态分配（在运行时指定大小）；
- 每个 SM 都有，且是有限的共享内存容量；
- 注意 Bank conflict，当同一 warp 中多个线程访问**同一个 Bank 的不同地址**时。
- 在共享内存中进行读写操作时，通常需要使用 `__syncthreads()` 函数进block内线程程同步。
- 每个线程块可用的共享内存量是有限的。可以使用 `cudaGetDeviceProperties` 函数来查询设备的共享内存大小。
- 物理上时一维的。

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
- 数据对齐：确保线程访问的数据在共享内存中对齐到 Bank 宽度。（Bank 宽度为 4bytes）则确保每次访问的地址是 4 的倍数。


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

- 性能: 无论你如何逻辑地组织 Shared Memory，其底层**物理布局始终是线性**的。 因此，访问模式对性能至关重要。应尽量避免 Bank conflict。 这通常意味着访问元素时应尽量保持内存访问的连续性。 多维数组的访问模式需要仔细设计，以避免 Bank conflict。
- 索引计算: 对于动态分配的 Shared Memory，你需要自己负责索引计算，这可能会增加代码的复杂性。
- 大小限制: 每个 block 可用的 Shared Memory 总量是有限的。 你需要根据你的 GPU 架构和 block 大小来选择合适的多维数组大小。


## Bank 的组织方式  【记住】

- Bank 的**宽度（Bank Width）**（每一个 Bank 内访问的数据位宽）是硬件决定的（4 Bytes），这表示每个 Bank 每次只能服务 4Byte 的访问请求（即一个 float）。
- 同一个 Bank 中的相邻地址是**不连续的**。
- Shared memory 被划分为 **32 个 Bank**。Bank 编号是0~31，在0~31之间的循环。
- 同一个 Bank 中的相邻地址之间相差 32（个）*宽（4 Byte）= 128 Byte
- **共享内存地址的编号**：在共享内存中，每个字节都有一个唯一的地址编号。这些地址编号从 0 开始，依次递增。
- **字节编号**：Shared Memory 的地址编号是从 0 开始的，每个字节都有一个唯一的编号（**字节编号**）。比如，ID是 0 的 Bank 包含的字节编号是 0,1,2,3, 128,129,130,131，256,257,258,259, 384,385,386,387，。。。表示当前字节是“第几个字节”
- **字节地址**（byte address）：shared memory 中该字节相对于 Base address 的偏移地址。
- Base address：表示数组在内存中的起始位置。
- 字节编号和地接地址是一个意思


### 如下是 32 个 Bank，每个 Bank 宽 4 Bytes

|Bank id|字节编号|字节编号|字节编号|字节编号|
|---|---|---|---|---|
|Bank 0|  0~3|    128~131|   256~259| ... |
|Bank 1|  4~7|    132~135|   260~263| ... |
|Bank 2|  8~11|   136~139|   264~267| ... |
|...|...|...|...|...|
|Bank 31| 124~127| 252~255|  380~383| ... |


Bank id 和 **byte 地址**之间的关系是：

`bank_id = (字节地址 // 4​) mod 32`

mod 32 表示除以32后的 余数，值在 0~31。**通过这个公式可以判断一个 byte 地址是属于哪个 Bank**。 这是判断是否bank conflict 的核心。

Thread 访问的地址属于哪个 Bank，这只取决于 **字节地址**，与该地址上存放的数据类型（如 float、int、double）无关。


## Bank 冲突

同一个 Warp 中多个线程同时访问同一个 Bank 中的不同地址，会导致 Bank conflict。

**不同 warp 中的线程可以独立访问共享内存，而不会导致 Bank conflict**【？？？】。这是因为 warp 调度器会分别处理每个 warp 的内存访问请求。


### case1: 一个 Warp 中的 32 个线程尝试访问 Shared Memory 中的以下地址（字节地址）：

~~~sh
Thread 0: Address 0     属于 Bank 0   0//4%32=0
Thread 1: Address 4     属于 Bank 1   4//4%32=1
Thread 2: Address 8     属于 Bank 2   8//4%32=2
...
Thread 31: Address 124  属于 Bank 31  124//4%32=31
~~~

在这个例子中，每个线程访问的**地址都位于不同的 Bank 中**，因此**不会发生** Bank conflict。


### case2: 如果一个 Warp 中的 32 个线程尝试访问 Shared Memory 中的以下地址（字节地址）：

~~~sh
Thread 0: Address 0    属于 Bank 0   0//4%32=0
Thread 1: Address 0    属于 Bank 0   0//4%32=0
Thread 2: Address 0    属于 Bank 0   0//4%32=0
...
Thread 31: Address 0   属于 Bank 0   0//4%32=0
~~~

所有线程都访问同一个地址，这**不会导致** Bank conflict，因为硬件会将数据**广播**给所有线程.


### case3: 如果一个 Warp 中的 32 个线程尝试访问 Shared Memory 中的以下地址（字节地址）：

~~~sh
Thread 0: Address 0     属于 Bank 0//4%32=0
Thread 1: Address 32    属于 Bank 32//4%32=8
Thread 2: Address 64    属于 Bank 64//4%32=16
Thread 3: Address 96    属于 Bank 96//4%32=24
Thread 4: Address 128   属于 Bank 128//4%32=0
Thread 5: Address 160   属于 Bank 160//4%32=8
Thread 6: Address 192   属于 Bank 192//4%32=16
Thread 7: Address 224   属于 Bank 224//4%32=24
Thread 8: Address 256   属于 Bank 256//4%32=0
Thread 9: Address 288   属于 Bank 288//4%32=8
Thread 10: Address 320  属于 Bank 320//4%32=16
...
Thread 31: Address 992  属于 Bank 992//4%32=24
~~~

32 个 thread 属于同一个 warp，可以看出：

- conflict 发生在 4 个Bank上（Bank id：0,8,16,24）
- 32（个thread）/ 4 = 8，每个 Bank 有 8 个 thread 同时访问不同的地址，故这是 8-way 的 conflict
- **x-way conflict** 指的是，有 x 个线程在同一个 Bank 中同时访问不同的地址。
 

## KAQ：Double 类型的数据会跨Bank访问，如何？

Double 占 8 字节, 必然跨越 两个相邻 banks

地址 0~7 属于 Bank 0（地址编号 0~3）和 Bank 1（地址编号 4~7）。

避免在Shared memory中使用 Double，如果必须使用，确保地址对齐到 8 字节边界：0、8、16。。。


## KAQ：二维数组 `tile[32][32]` 在内存中是 行优先存储意味着什么

二维数组先行后列是 C/C++ 数组的标准语义。

行优先意味着数组 `tile[i][j]` 的元素按行排列，即先存储第 0 行所有元素（`tile[0][0]`, `tile[0][1]`, ..., `tile[0][31]`），然后第 1 行（`tile[1][0]`, ..., `tile[1][31]`），依此类推。一共有 32 行和 32 列。

`tile[i][j]` 的**字节地址**为 `(i * 32 + j) * sizeof(float)`，其中的 32 表示每一行的元素个数。


## 计算二维数组中元素在行优先内存中的一维的字节地址

在行优先存储中，数组的元素按照以下顺序一维存储：

~~~sh
a[0][0], a[0][1], a[0][2], ..., a[0][num_cols-1],
a[1][0], a[1][1], a[1][2], ..., a[1][num_cols-1],
a[2][0], a[2][1], a[2][2], ..., a[2][num_cols-1],
...
a[num_rows-1][0], a[num_rows-1][1], a[num_rows-1][2], ..., a[num_rows-1][num_cols-1]
~~~

假设有一个二维数组 `a[num_rows][num_cols]`，要计算元素 `a[i][j]` 在内存中的地址。过程如下：

1. 行偏移量：要找到 `a[i][j]` 的地址，我们首先需要跳过 `i` 行。每一行有 `num_cols` 个元素，所以我们需要跳过 `i * num_cols` 个元素。
2. 列偏移量：在跳过 `i` 行后，我们需要在当前行中跳过 `j` 个元素。
3. 总偏移量：总共需要跳过 `i * num_cols + j` 个元素。
4. * 起始地址计算：由于每个元素占用 `element_size` 个字节，所以总偏移量需要乘以 `element_size`。最后，加上数组的起始地址 `base_address`，就可以得到元素 `a[i][j]` 在行主序内存中一维内存偏移的**起始地址**：`Address = base_address + (i * num_cols + j) * element_size`


## 注意 Bank Conflict 只发生在同一个 Warp 中的32个线程中

Shared Memory 的 Bank 是 每个 Warp 独立争用 的资源，~~不同的 Warp 之间访问 Shared memory 不会 Bank Conflict~~。如果不同的 Warp 尝试访问共享内存，并且它们的目标 Bank 相同，SM 调度器会**串行化**这些 Warp 的执行，而不是发生 Warp 内部的 Bank Conflict。

多个warp 访问同一个 bank时的冲突是 inter-warp conflict。

- `intra-warp` bank conflict：1 warp 内多线程争同一 bank
- `inter-warp` conflict：不同 warp 争 bank, SM 串行化整个 warp


### KAQ：这两种Conflict 那种更慢？

