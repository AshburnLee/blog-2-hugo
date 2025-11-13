## Why

首先根据全局索引，计算得到src和dst的各个维度索引，即得到 i03/i13, i02/i12, i01/i11, i00/i10。然后与各自的字节 stride 组合成最终的内存里的字节偏移。

这个逻辑是完美处理任意 padding、permute、view 的根本原因

torch中的 `.contiguous()` 函数作用是 把非连续的 tensor 变成内存连续排布的tensor。


## case & 逻辑 

ne 26624
src ne00=128 ne01=16 ne02=13 ne03=1
    nb00=4 nb01=6656 nb02=512 nb03=106496

dst ne10=2048 ne11=13 ne12=1 ne13=1
dst nb10=4 nb11=8192 nb12=106496 nb13=106496

src:  ne[128, 16, 13, 1]    nb[4, 6656, 512,    106496]
dst:  ne[2048, 13, 1, 1]    nb[4, 8192, 106496, 106496]

~~~cpp
# define CUDA_CPY_BLOCK_SIZE 64
const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
const int num_threads = CUDA_CPY_BLOCK_SIZE;


typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

template <cpy_kernel_t cpy_1>
static __global__ void cpy_flt(const char * cx, char * cdst, const int ne,
                               const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                               const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                               const int nb12, const int nb13, char ** cdst_indirect, int graph_cpynode_index) {
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // char * cdst = (cdst_indirect != nullptr) ? cdst_indirect[graph_cpynode_index]: cdst_direct;

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets


    /*
    const int64_t i03 = i / s03;
    const int64_t i02 = (i - i03*s03 ) / s02;
    const int64_t i01 = (i - i03*s03  -  i02*s02) / s01;
    const int64_t i00 = i - i03*s03 - i02*s02 - i01*s01;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;
    */
    const int64_t i03 = (i / ne00*ne01*ne02) % ne03
    const int64_t i02 = (i / ne00*ne01) % ne02
    const int64_t i01 = (i / ne00) % ne01
    const int64_t i00 = (i / 1) % ne00
    const int64_t dst_offset = i00*nb10 + i01*nb11 + i02*nb12 + i03 * nb13;

    const int64_t i13 = (i / ne10*ne11*ne12) % ne13
    const int64_t i12 = (i / ne10*ne11) % ne12
    const int64_t i11 = (i / ne10) % ne11
    const int64_t i10 = (i / 1) % ne10
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

// caller 子啊实际调用者那里，看出 cpy_1 在编译时是实际的 cpy_1_flt 函数
{...} else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_flt<cpy_1_flt<float, float>>;
} ...

template<typename src_t, typename dst_t>
static __device__ void cpy_1_flt(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}

~~~


## KAQ：比核心公式3 更一般的公式

核心公式3 适用于连续存储的多维 tensor，但是对于不连续存储的多维 tensor。但是对于这里的case，由于shape 不会应为padding而改变，公式中只有元素stride，不存在字节stride，故两个公式是等价的（除法取余 & 减法除法）。


## KAQ：Padding 后的 tensor，只有 nb 变化 shape 没有变化，这是什么样的情形？

src ne00=128 ne01=16 ne02=13 ne03=1
    nb00=4 nb01=6656 nb02=512 nb03=106496

推测：如果连续存储，那么：
- nb00 = 4
- nb01 = nb00 × ne00 = 4 × 128 = 512
- nb02 = nb01 × ne01 = 512 × 16 = 8192
- nb03 = nb02 × ne02 = 8192 × 13 = 106496

但实际上：
- nb00 = 4
- nb01 = 6656      6656/4=1664
- nb02 = 512       512/4=128
- nb03 = 106496

可能存在 Permute 和 Padding，导致 nb 变化。但 shape 不会变化，是因为实际数值还是那些，只是 padding 了许多没有的数据。128×16×13 = 26624 个元素的值其实全都在，一个都不少，只是被故意“以某种方式” 摆在了一块更大的内存里，中间插满了 6144 字节的 padding。


## KAQ：对于不连续存储的 tensor，offset 只要套用核心公式1 就可以了？为什么 

公式：`offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;`

无论有没有 padding，无论张量是不是 contiguous，无论有没有被 permute/view 过，只要你知道当前的 ne[] 和 nb[]，这个公式永远能给你算出逻辑坐标 (i03, i02, i01, i00) 对应的元素的真实字节偏移。

padding 已经被 nb 吃进去了, 完全不用手动处理。

必须先用 `char*` 完成字节级偏移，再强转成目标类型。

这是非连续的tensor，需要提供正确的字节stride。然后使用 char* 完成字节偏移的计算，追后转换成目标数据类型。


