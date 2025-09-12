+++
date = '2025-01-31T12:45:53+08:00'
draft = false
title = 'Tensor Core Wmma Detail'
tags = ["CUDA","Tensor Core"]
categories = ["CUDA"]
+++

关键的是索引的计算

## 矩阵分块，需要减少对全局内存的访问

每个矩阵的子矩阵被访问多少次？

给出结论：AxB=C，

  - 对于 A(2块x3块)，B(3块x4块)，得C(2块x4块)，A的每个子矩阵被访问 4 次，B的每个子矩阵被访问 2 次。
  - 对于 A(3块x3块)，B(3块x3块)，得C(3块x3块)，A的每个子矩阵被访问 3 次，B的每个子矩阵被访问 3 次。
  - 对于 A,B 矩阵大小都是 1024x1024, 每个子矩阵大小 16x16，那么 A(64块x64块)，B(64块x64块)，得C(64块x64块)，A的每个子矩阵被访问 64 次，B的每个子矩阵被访问 64 次。

这就是为什么，分块矩阵一定许需要对访存进行优化，减少对全局内存的访问次数。


## tensor core 计算 16x16 的 tile

那么 warp 中 thread 编号 0 到 31，如何对应上 16x16 数据 tile ？

答：在WMMA API中，一个 warp 的32个线程共同完成 16x16 矩阵的计算。具体来说，每个线程负责处理矩阵的一部分数据。WMMA API通过内部的数据结构和指令来管理线程如何访问和处理数据，而不是显式地将线程 ID 映射到特定的数据元素。

然而，在实现细节上，WMMA API 通常将 warp 内的线程分为更小的组，每个组负责处理矩阵乘法的一个部分。


## warp 全局 idx 计算

我的线程配置是 `grid(16, 16), block(128, 4), ` 其中 block 每行 128 个线程，被分成 4 个 warp （4x32=128），y 方向有4组。计算 warp 全局 id：

~~~cpp
uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};
~~~

- `blockIdx.x * blockDim.x` 计算当前块在 x 方向的起始位置。即在当前哪个block
- `threadIdx.x`是当前线程在块内的 x 坐标。block 的哪个 thread
- 将这两个值相加，然后除以 `warpSize`，就可以得到当前线程所在的 warp 在 x 方向的索引。**也就是确定当前线程属于哪个warp**。***

假设我们有一个线程块，其大小为 256 个线程（即 blockDim.x = 256），那么这个块包含 8 个 warp（因为256 / 32 = 8）。

如果当前线程的 `threadIdx.x` 为 0 到 31 （这些`id / warpSize = 0`），则它属于第一个 warp。
如果当前线程的 `threadIdx.x` 为 32 到 63 （这些`id / warpSize = 1`），则它属于第二个 warp，以此类推。

这样做，是将一个**warp 做为一个基本单位了**，即一个warp x 方向有自己的全局 id。 *** 
实例：

    假设 blockIdx.x = 0, blockDim.x = 128, threadIdx.x = 0，则 warpM = (0 * 128 + 0) / 32 = 0
    假设 blockIdx.x = 0, blockDim.x = 128, threadIdx.x = 31，则 warpM = (0 * 128 + 31) / 32 = 0
    假设 blockIdx.x = 0, blockDim.x = 128, threadIdx.x = 32，则 warpM = (0 * 128 + 32) / 32 = 1
    假设 blockIdx.y = 0, blockDim.y = 4, threadIdx.y = 0，则 warpN = 0 * 4 + 0 = 0
    假设 blockIdx.y = 0, blockDim.y = 4, threadIdx.y = 3，则 warpN = 0 * 4 + 3 = 3
    假设 blockIdx.y = 1, blockDim.y = 4, threadIdx.y = 0，则 warpN = 1 * 4 + 0 = 4


## 然后根据 warp 的全局 ID 计算这个 warp 如何访问矩阵数据

gridDim.x: 1

gridDim.y: 1

blockDim.x: 64

blockDim.y: 2

已知 block(64,2) grid(1,1)，矩阵 A 大小：32x32 矩阵 B 大小：32x32

已知 tensor core 是 col-major 的访问，且 `C=A^T * B`, 所以 A 应该是 row-major 的。

~~~cpp
// A 矩阵只与 warpM 相关
uint32_t const matrix_mma_a_row_idx{ki};  
uint32_t const matrix_mma_a_col_idx{warpM * WMMA_M};
// B 只与 warpN 相关
uint32_t const matrix_mma_b_row_idx{ki};  
uint32_t const matrix_mma_b_col_idx{warpN * WMMA_N};
// 计算每个 fragment 的首地址,作为 load_sync 的第二个参数
// 这里的首地址，对于每一个 warp 都有不同的，对应A,B的首地址, 保证读取子矩阵的正确性！ 
T1 const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx + matrix_mma_a_col_idx * lda};
T1 const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx + matrix_mma_b_col_idx * ldb};
// 2. Load the mma matrix inputs.
nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);
~~~

~~~sh
gridDim.x: 1
gridDim.y: 1
blockDim.x: 64
blockDim.y: 2

m: 32
n: 32
k: 32
lda: 32
ldb: 32
ldc: 32
~~~

通过 log 得：

  - 每个 warp 读取A和B的子矩阵，而且每个子矩阵被读 **2** 次。（这与上述结论一致）
  - 而且 A 矩阵是行优先地读，B 矩阵是列优先地读。
  - 哪个 warp 读哪个子矩阵，并没有关系，它总是连续的 thread 读连续的地址。
  - 硬件读时，A 和 B 都是行列读，这不影响数学上的矩阵乘法。
  - 对于 `C=A^T * B`，A是横着读？B是列着读？

~~~sh
(warpM,warpN):0,0   Aidx: 0, 0,   A: A+0+0*32,     Bidx: 0, 0,    B: B+0+0*32
(warpM,warpN):1,0   Aidx: 0, 16,  A: A+0+16*32,    Bidx: 0, 0,    B: B+0+0*32
(warpM,warpN):0,1   Aidx: 0, 0,   A: A+0+0*32,     Bidx: 0, 16,   B: B+0+16*32
(warpM,warpN):1,1   Aidx: 0, 16,  A: A+0+16*32,    Bidx: 0, 16,   B: B+0+16*32

(warpM,warpN):0,0   Aidx: 16, 0,  A: A+16+0*32,    Bidx: 16, 0,   B: B+16+0*32
(warpM,warpN):1,1   Aidx: 16, 16, A: A+16+16*32,   Bidx: 16, 16,  B: B+16+16*32
(warpM,warpN):0,1   Aidx: 16, 0,  A: A+16+0*32,    Bidx: 16, 16,  B: B+16+16*32
(warpM,warpN):1,0   Aidx: 16, 16, A: A+16+16*32,   Bidx: 16, 0,   B: B+16+0*32

# 加上 c_frag 的子结果首地址：
(warpM,warpN):0,0   Aidx: 0, 0,  A: A+0+0*32,     Bidx: 0, 0,  B: B+0+0*32   C: C+0+0*32
(warpM,warpN):1,0   Aidx: 0, 16, A: A+0+16*32,    Bidx: 0, 0,  B: B+0+0*32   C: C+16+0*32
(warpM,warpN):0,1   Aidx: 0, 0,  A: A+0+0*32,     Bidx: 0, 16, B: B+0+16*32  C: C+0+16*32
(warpM,warpN):1,1   Aidx: 0, 16, A: A+0+16*32,    Bidx: 0, 16, B: B+0+16*32  C: C+16+16*32

(warpM,warpN):1,1   Aidx: 16, 16, A: A+16+16*32,   Bidx: 16, 16, B: B+16+16*32  C: C+16+16*32
(warpM,warpN):0,0   Aidx: 16, 0,  A: A+16+0*32,    Bidx: 16, 0,  B: B+16+0*32   C: C+0+0*32
(warpM,warpN):1,0   Aidx: 16, 16, A: A+16+16*32,   Bidx: 16, 0,  B: B+16+0*32   C: C+16+0*32
(warpM,warpN):0,1   Aidx: 16, 0,  A: A+16+0*32,    Bidx: 16, 16, B: B+16+16*32  C: C+0+16*32
~~~

根据上述索引，每个 warp 加载了A 和 B 的子矩阵，并执行计算之得到 acc_frag。
~~~cpp
// 3. Perform the matrix multiplication
nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
~~~



~~~cpp
uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

T2* matrix_mma_c_mptr{C + matrix_mma_c_row_idx + matrix_mma_c_col_idx * ldc};
nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc, nvcuda::wmma::mem_col_major);
~~~

最后有一步是缩放和累加，适用于情况是 C=A*B+C。


  - `alpha = 1.0f, beta = 0.0f`： 这表示只对 `acc_frag` 中的结果进行缩放，并将结果存储到 `c_frag` 中，`c_frag` 的初始值被忽略。适用于`+C`不存在时。
  - `alpha = 0.5f, beta = 0.5f`： 这表示对 `acc_frag` 和 `c_frag` 中的结果都进行缩放，并将结果累加到 `c_frag` 中，`acc_frag` 和 `c_frag` 的初始值都有影响。
  - `alpha = 1.0f, beta = 1.0f`： 这表示将 `acc_frag` 中的结果直接累加到 `c_frag` 中，`c_frag` 的初始值会被保留。

~~~cpp
// Scale and accumulate (optional)
for (uint32_t i = 0; i < acc_frag.num_elements; i++) {
    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
}
~~~


## 矩阵分块是，每个子矩阵被访问多少次？

见上述结论

## warp 如何访问 A 和 B ？ [通过小规模线程配置和输入数据]

对于 `C=A^T * B`，A是横着读？B是列着读？

2x2，2x2 例子不好，改用2x3，3x1 的例。 【TODO】

## 所有 warp 计算的子结果，如何组合变为最终的结果？ [没有理解wmma的acc部分]

mma_sync() 函数帮你做了这部分工作，

## warp 需要循环处理时

~~~cpp
// 循环 K 次，累加结果
for (int k = 0; k < K; k += WMMA_K) {
    // 加载输入矩阵 A 和 B 的子块到 fragment 中
    wmma::load_matrix_sync(a_frag, a + row_start * K + k, K);
    wmma::load_matrix_sync(b_frag, b + k * N + col_start, N);

    // 执行矩阵乘法和累加
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
}
~~~

每次循环 `mma_sync` 执行的是：`acc_frag = acc_frag + a_frag * b_frag`
当循环结束时，`acc_frag` 中存储的是所有局部结果的累加和，即完成了整个矩阵乘法中对应子块的计算。

### warp 循环处理时，如何保证 acc_frag 累加的是正确的 位置上的局部结果？

。。。
