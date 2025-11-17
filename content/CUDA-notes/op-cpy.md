
+++
date = '2025-11-16T12:45:35+08:00'
draft = true
title = 'op cpy_transpose'
tags = ["CUDA"]
categories = ["CUDA"]
+++


## cpy transpose

非 transpose 的cpy kernel 同 op-continue 。

## 当前多维 Tensor 的 transpose 语义？

只交换两个维度，具体看上下文语义。从以下 code 看，是哪两个维度的 transpose。
假设 shared memory 按 float 对齐（4字节）。向量化 T = half thread 访问 2 个，T = int8_t thread 访问 4 个

~~~cpp
const int64_t nmat = ne / (ne00 * ne01);
const int64_t n = ne00 * ne01;
~~~


## KAQ：这里是什么意思

从上下文看，nb00 nb01 nb02 nb03 默认是递增的。当nb00 > nb02 时，有以下额外操作
~~~cpp
else if (nb00 > nb02) {  //
            ne00n = ne00;
            ne01n = ne01*ne02;
            ne02n = 1;
        }
~~~

- 为什么有这个操作？
- 为什么是 nb00 与 nb02 比较？


## KAQ: Shared memory 是多少字节对齐？ 

Shared Memory 以bank为单位组织的，每个bank 一次处理4字节（这都是已知的事实）。所以Shared memory 是以4字节对齐的。
~~~cpp
const int col = threadIdx.x * sizeof(float)/sizeof(T);
~~~


## KAQ：为什么只有 col 向量化？

~~~cpp
const int row = threadIdx.y + j;
const int col = threadIdx.x * sizeof(float)/sizeof(T); 
...
const int row = threadIdx.x;
const int col = (threadIdx.y + j) * sizeof(float)/sizeof(T);
~~~

因为 col 索引变化是在 shared memory 中是连续的 `tile[row][col]`, 即 col 的变化的最快的，故只对 col 向量化。表示连续的线程访问多个连续的数据。在transpose 中一个线程连续访问多个相邻位置，是符合逻辑的。


## KAQ：这里表明是如何分块，如何并行的？

~~~cpp
// 分多个子矩阵 nmat, z 循环遍历 batch
// 每个 block 处理 CUDA_CPY_BLOCK_NM=8 个矩阵
for (int i = 0; i < CUDA_CPY_BLOCK_NM; ++i) {
    const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM/*=8*/ + i;
    for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {...}

    __syncthreads();

    // 这里的转置操作才是重点，体现了 block 如何从一个地方读到另一个地方，但是这两个地方的形状是转置的
    //读 tile 32 行 7 列 ， 写入 dst 7 行 32 列
    for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {...}
}
~~~
最外层是 mat 并行（mat是另一个维度，不影响做 transpose 的两个维度），mat 对应的维度不参与转置。内层第一个循环将 src 数据读入 tile，内层第二个循环将tile中数据写回 dst。


### KAQ：双重循环中的内循环的第一个 for 循环 src 如何访问？

~~~cpp
src[imat * ne00 * ne01 + (y + j) * ne01 + x]
~~~

为什么不是

~~~cpp
src[imat * ne00 * ne01 + (y + j) * ne00 + x]
~~~

~~因为（我的合理推测，需要在实际 case 根据case的输入ne和nb验证） src（col-major） shape = [ne01, ne00, ne02] 转置后得到 shape=[ne00, ne01, ne02]。与 code 中的 `dst[imat * ne00 * ne01 + (ty + j) * ne00 + tx]`  变化最快的的维度是 ne00。~~

进过分析：
- 首先，从 src 到 tile 是一行 src 写入一行 tile，
- 然后，tile 是 row major 的，写入 tile，计算其 flatten id 时，就应该是 x + y * num_col。

故有理由推测 ne01 表示 num_col，所以 转之前的 shape 是 `[ne00, ne01, ne02]`, 转置后 shape `[ne01, ne00, ne02]`


## KAQ：以下写法正确吗？

~~~cpp
const int row = threadIdx.y + j;
const int col = threadIdx.x * sizeof(float)/sizeof(T); 
T *tile2 = reinterpret_cast<T*>(tile);
tile2[raw][col] = src[imat * ne00 * ne01 + (y + j) * ne01 + x];
~~~

错，因为 `T*` 不能 `[][]`，只能是一维索引。


## KAQ: 转之后的 索引对应的是什么意义？对角索引？ 即 tx ty 为什么这样写？

一图胜千言，画出**读写示意图**就明了了

转之后的 tx ty 表示 dst 的写入索引

什么是对角索引？**读写示意图**给出答案


## KAQ: x,y,tx,ty 的含义是什么?

- x 表示 src 列索引，y 表示 src 行索引
- tx 是 dst 列索引，ty 是 dst 的行索引。
- row & col 是 block 各自的行列索引。多个block 时，每个 block 有自己的 tile，只与 threadidx.x/y 有关，因为tile是block wise的。


## KAQ：使用 Shared memory 的优势并没有体现?
## KAQ：`__shared__ float tile[32][32+1]` 写入数据时，位置不会出错吗？

不会。事实上，最后一列不会写入元素。写入元素的只有32行和32列。这完全是由你的访问索引控制的。

假如 src[32][32]，block(32,1), 我定义

- 访问src的索引分别是 `x=threadIdx.x `（0~31）, `y=threadIdx.y` （0）
- 定义block纵向循环 `j=0~31`
- 访问tile的索引分别是 `col=threadIdx.x` (0~31) `row=threadIdx.y + j` (0 + j)

那么 `tile[row][col] = src[x+(y+j)*32] ` 32 表示src列个数。32个数写入 tile **一行中的32个列，而非33个列**。

故tile中最后一列是不会被访问到了。加一列只是 bank 中的 word 错位了。


## code


~~~cpp
template <typename T>
static __global__ void cpy_flt_transpose(const char * cx, char * cdst, const int ne,
                               const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                               const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                               const int nb12, const int nb13) {

    const T* src = reinterpret_cast<const T*>(cx);
    T* dst = reinterpret_cast<T*>(cdst);

    const int64_t nmat = ne / (ne00 * ne01);

    // 转置前索引
    const int sx = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int sy = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    // 转置之后的索引 从读写示意图推导出
    const int tx = blockIdx.y/*画出示意图就明了了*/ * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int ty = blockIdx.x/*画出示意图就明了了*/ * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

    // block wise 每一个block都有一个 [32][32+1] 的 tile
    __shared__ float tile[CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D+1];

#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM/*=8*/; ++i) {
        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM/*=8*/ + i;
        if (imat >= nmat)
            break;

#pragma unroll
        // j = 0 ~ 31，将 src[] 内容读到 tile[row][col]，读转之前，故与 x，y 相关
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D/*=32*/; j += CUDA_CPY_BLOCK_ROWS/*=8*/) {
            if(sx < ne01 && sy + j < ne00) {
                const int row = threadIdx.y + j; // (0+j)~(7+j)
                const int col = threadIdx.x * sizeof(float)/sizeof(T); 
                // 转之前 [ne00][ne01][ne02] 
                T *tile2 = reinterpret_cast<T*>(tile[row]);
                const int idx = sx + (sy + j) * /*num_col=*/ne01 + imat * ne00 * ne01;
                // 第 0 row 的 32 个 col 读入
                // 第 2 row 的 32 个 col 读入
                // ...
                // 第 7 row 的 32 个 col 读入
                tile2[col] = src[idx];
            }
        }

        __syncthreads();

#pragma unroll
        // j = 0 ~ 31，将 tile[row][col] 内容写入 dst[]，读转之后，故与 tx,ty 相关
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D/*=32*/; j += CUDA_CPY_BLOCK_ROWS/*=8*/) {
            if (ty + j < ne01 && tx < ne00) {
                const int row = threadIdx.x; // 0~31
                const int col = (threadIdx.y + j) * sizeof(float)/sizeof(T);
                // 转之后 [ne01][ne00][ne02]
                const int idx = tx + (ty + j) * /*转之后num_col=*/ne00 + imat * ne00 * ne01;
                // 32 行，对于每一行，读 7个col
                const T *tile2 = reinterpret_cast<const T*>(tile[row]);
                // 第 0 row 的 7 个 col 写入
                // 第 1 row 的 7 个 col 写入
                // ...
                // 第 31 row 的 7 个 col 写入
                dst[idx] = tile2[col];
            }
        }
    }
}


# define CUDA_CPY_BLOCK_SIZE 64
const int CUDA_CPY_TILE_DIM_2D = 32; // 2D tile dimension for transposed blocks
const int CUDA_CPY_BLOCK_NM = 8;     // block size of 3rd dimension if available
const int CUDA_CPY_BLOCK_ROWS = 8;   // block dimension for marching through rows

template<typename src_t, typename dst_t, bool transposed = false>
static void ggml_cpy_flt_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    if (transposed) {
        GGML_ASSERT(ne == ne00*ne01*ne02);  // ne[3] is 1 assumed
        int ne00n, ne01n, ne02n;
        if (nb00 <= nb02) { // most likely safe to handle nb00 = nb02 case here
            ne00n = ne00;
            ne01n = ne01;
            ne02n = ne02;
        } 
        // else if (nb00 > nb02) {  // 暂不考虑这种情况
        //     ne00n = ne00;
        //     ne01n = ne01*ne02;
        //     ne02n = 1;
        // }
        // num_blocks: ((ne01 + 32-1)/32, (ne00 + 32-1)/32, (ne/(ne01*ne00) + 8-1)/8)
        dim3 dimGrid( (ne01n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                      (ne00n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                      (ne/(ne01n*ne00n) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM);
        // num_threads: (32,8,1)
        dim3 dimBlock(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
        cpy_flt_transpose<dst_t><<<dimGrid, dimBlock, 0, stream>>>
            (cx, cdst, ne, ne00n, ne01n, ne02n, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
    } else {
        const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
        cpy_flt<cpy_1_flt<src_t, dst_t>><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
            (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
    }
}
~~~


## 拆解
- 有个具体case（最好的，若没有，对不不复杂的kernel 可以从 code 中推导出）
- 无 mat 循环
- 无 j 循环读入 tile
- 无 j 循环写入 dst
- 手写 kernel
