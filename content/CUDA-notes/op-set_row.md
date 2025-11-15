+++
date = '2025-11-14T12:45:35+08:00'
draft = true
title = 'op set_rows'
tags = ["CUDA"]
categories = ["CUDA"]
+++


## Why

在 llama.cpp 中，这个 kernel 用于 embedding lookup、MoE 专家路由、量化权重选择等场景，PyTorch 版本可用于验证逻辑或调试。

`set_rows`，是“按索引写入行”，不是“按索引读取行”。
这正是 KV cache 更新、RoPE 位置注入等场景所需的 **Scatter 语义**。

`dst[src1[i]] = src[i]`

set_rows 是“原地写”（in-place scatter），不是“构造新张量”。如何理解？

在 KV Cache 中，dst 是预分配的完整 tensor，set_rows 只负责写入指定行，不会创建和剪裁，因为KV-Cache 的语义是**更新**，而不是过滤，**未更新的旧行是要保留的**。

另外 dst 是预分配大小的，避免内存重新分配。



## 逻辑

根据 src1 的索引，从 src0 中选择行，复制到 dst 的对应位置。实例:

~~~
row-major 的存储
src0 (4×3):
[ 0.0,  0.1,  0.2 ]
[ 1.0,  1.1,  1.2 ]
[ 2.0,  2.1,  2.2 ]
[ 3.0,  3.1,  3.2 ]

src1 (2):
[ 1, 3 ]

for i in [0, 1]:
    dst[src1[i]] = src0[i]

即：
dst[1] = src0[0] 是 [ 0.0, 0.1, 0.2 ] 
dst[3] = src0[1] 是 [ 3.0, 3.1, 3.2 ]

dst (6×3):
[ old, old, old]
[ 0.0, 0.1, 0.2]
[ old, old, old]
[ 3.0, 3.1, 3.2]
[ old, old, old]
[ old, old, old]
~~~

实际 case（col-major）：
~~~
src0: [1024, 13]  <= 13 行新数据（e.g., 当前 13 tokens 的 K/V）
src1: [13]        <= 写入位置（e.g., [51,52,...,63]）
dst:  [1024, 64]
~~~

1024 是索引连续变化的列的长度，dst 中有 64 个这样的列。所以对于col-major，`set_rows` 是物理上的 set column。


## code


~~~cpp
template<typename src_t, typename dst_t>
static __global__ void k_set_rows(
        const src_t * __restrict__ src0, const int64_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3) {

    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;

    if (i >= ne_total) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    (void)(ne10);
    (void)(ne13);
}


template<typename src_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const int64_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(int64_t);
    const int64_t s11 = nb11/sizeof(int64_t);
    const int64_t s12 = nb12/sizeof(int64_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0) {
        k_set_rows<<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            s01, s02, s03,
            s10, s11, s12,
            s1, s2, s3);
    }
}

void caller() {
set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
}
~~~