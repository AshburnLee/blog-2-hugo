+++
date = '2025-11-10T12:45:35+08:00'
draft = true
title = 'op get_rows'
tags = ["CUDA"]
categories = ["CUDA"]
+++



【hold】llama.cpp 中 get_rows 的输入输出么有理解，故code不能分析

## Why

在 llama.cpp 中，这个 kernel 用于 embedding lookup、MoE 专家路由、量化权重选择等场景，PyTorch 版本可用于验证逻辑或调试。

`get_rows` 是与 PyTorch 的 index_select对齐的，都是 **Gather 语义**。

`dst[i] = src[src1[i]]`


## 逻辑

根据 src1 的索引，从 src0 中选择行，复制到 dst 的对应位置。实例:

~~~
src0 (4×3):
[ 0.0,  0.1,  0.2 ]   // 行 0
[ 1.0,  1.1,  1.2 ]   // 行 1
[ 2.0,  2.1,  2.2 ]   // 行 2
[ 3.0,  3.1,  3.2 ]   // 行 3

src1 (2):
[ 1, 3 ]  // 索引：第 1 行 和 第 3 行

dst (2×3):
[ 1.0,  1.1,  1.2 ]   // 第 1 行
[ 3.0,  3.1,  3.2 ]   // 第 3 行
~~~

Pytorch 中有对齐的reference：`index_select`

~~~py
import torch

# ======================
# 1. 定义张量（与 CUDA kernel 对应）
# ======================

# src0: [ne00, ne01] = [4, 3]
src0 = torch.tensor([
    [0.0, 0.1, 0.2],  # 行 0
    [1.0, 1.1, 1.2],  # 行 1
    [2.0, 2.1, 2.2],  # 行 2
    [3.0, 3.1, 3.2],  # 行 3
], dtype=torch.float32)  # shape: [4, 3]

# src1: [ne10] = [2], 索引值
src1 = torch.tensor([1, 3], dtype=torch.long)  # 选择 src0 的第 1 和第 3 行

# dst: [ne10, ne01] = [2, 3]
dst_expected = torch.tensor([
    [1.0, 1.1, 1.2],  # 来自 src0[1]
    [3.0, 3.1, 3.2],  # 来自 src0[3]
], dtype=torch.float32)

print("src0:\n", src0)
print("src1:", src1)


# 直接从 src0 中选择 src1 指定的行
dst = torch.index_select(src0, dim=0, index=src1)

print("dst (index_select):\n", dst)
print("check:", torch.allclose(dst, dst_expected))
~~~

~~~py
import torch
src0 = torch.tensor([[0, 1, 2],     # 行0
                     [3, 4, 5],     # 行1
                     [6, 7, 8]])    # 行2
# shape: [3, 3] → llama.cpp 的 [ne00=3, ne01=3]

src1 = torch.tensor([0, 2])        # 取第0行和第2行
# shape: [2]

dst = torch.index_select(src0, dim=1, index=src1)
print(dst)
~~~


## code

如果 nb（num_bytes） 表示不是连续的，那么 s（元素stride）当然也不是连续的。
~~~cpp
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
~~~

input:
- src0: float （2048,13,1,1）  , llm 中的意义是什么，各个维度在llm中的意义是什么？
- src1: int32  (1,1,1,1)
- dst:  float  (2048,1,1,1)

src0:
- ne00=2048, ne01=13, 
- nb01=2048*4(bytes)=8192, nb02(=nb03)=2048*13*4(bytes)=106496

src1:
- ne10(=ne11=ne12=ne13)=1
- s10(=s11=s12)=1

dst:
- s0=1,s1(=s2=s3)=2048

~~~cpp
template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
    const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
    const int64_t ne00,
    const int64_t ne12, 
    const size_t s1,      // 2048 dst 第一维 元素stride
    const size_t s2,      // 2048 dst 第二维 元素stride
    const size_t s3,      // 2048 dst 第三维 元素stride
    const size_t nb01,    // 2048*4(bytes)   =8192  , src0 第一维 字节stride
    const size_t nb02,    // 2048*13*4(bytes)=106496, src0 第二维 字节stride
    const size_t nb03,    // 2048*13*4(bytes)=106496, src0 第三维 字节stride
    const size_t s10,     // 1 src1 第零维 元素stride
    const size_t s11,     // 1 src1 第一维 元素stride
    const size_t s12      // 1 src1 第二维 元素stride
) { 

    // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
    const int i00 = blockIdx.y * blockDim.x + threadIdx.x;
    const int i10 = blockIdx.x;
    const int i11 = blockIdx.z / ne12;
    const int i12 = blockIdx.z % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const src0_t * src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
}
~~~

config: 
block (256,1,1)
grid  (1,8,1)

~~~cpp
#define CUDA_GET_ROWS_BLOCK_SIZE 256

const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
const dim3 block_nums(ne10, block_num_y, ne11*ne12);
~~~


# learn from code

## KAQ：为甚要转换为 char？

`const src0_t * src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);`表示字节级指针运算， 精确跳转到 src0 的第 i01 行，支持 非连续内存 + 任意 src0_t 类型。

“指针算术用 char*，索引用类型指针” 是 **处理非连续张量**的金科玉律。字节指针偏移（Byte-addressed pointer arithmetic）用 (const char *) + **字节步长** nbXX 实现 通用、非连续张量行定位。

`const char *` 是为了用 nb01（字节步长）做指针偏移，绕过 src0_t* 的元素级算术限制。 这是处理非连续张量的标准 CUDA 技巧。


## KAQ：为什么说是非连续张量？

我的 case 是连续的，但是“当前连续 ≠ 永远连续”，从大 KV Cache 中切片 → 自动生成非连续 src0，这是 llama.cpp 中 get_rows 的日常场景。如：

~~~cpp
ggml_tensor view = ggml_view_2d(ctx, &src0, 2048, 13, 
                                0, 50 * nb02);  // 从第 50 行开始
~~~

`ggml_view_2d` 并不复制数据，也不重新排列内存 —— 它只“重新解释”步长（strides）。所以kernel中会传入 字节步长。


## KAQ：为什么 dst 不用 字节级跳转？

因为dst存储是连续的，不存在非连续访问。并且dst类型确定，**指针算术天然按元素跳，因为你在定义一个指针时实惠声明类型的**。


## KAQ：从实际输入输出看，该op的功能是get columns

dst 的shape是什么样的？为什么，Qwen 中的case 是个特例。无法确定输出shape


## KAQ：如何并行？

与block/grid config 有关，其中就表示了如何并行！

上述case中输入2048*13 个元素，但是src1 只有一个元素，表示dst中只有src0中的一行，也就是只有2048个元素。所以线程2048 正好覆盖所有数据。13 表示 src1 的可选组范围。

`const dim3 block_nums(ne10, block_num_y, ne11*ne12)` 这里应该隐藏了并行规划

好的，请给我很小的实例，比如 src0 float [8,3,1,1]， src1 float [1,1,1,1], ~~但是 src0 是从x float [4,10,1,1] 中切片的，[:,2:5,:,:] 得到的。请给出code 中的计算过程~~

