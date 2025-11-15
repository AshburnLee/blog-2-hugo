+++
date = '2025-11-12T12:45:35+08:00'
draft = true
title = 'op roll'
tags = ["CUDA"]
categories = ["CUDA"]
+++


## Why

Qwen （Qwen3-1.7B-Q4_K_M.gguf）中没有这个 kernel 的调用。


## 逻辑

llama.cpp 中 roll op 的参数

~~~cpp
#define CUDA_ROLL_BLOCK_SIZE 256
void ggml_cuda_op_roll(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int s0 = dst->op_params[0];
    int s1 = dst->op_params[1];
    int s2 = dst->op_params[2];
    int s3 = dst->op_params[3];

    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) dst->src[0]->data;
    float *             dst_d  = (float *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst));

    cudaStream_t stream = ctx.stream();

    int64_t sz         = (ne00 * ne01 * ne02 * ne03);
    int64_t num_blocks = (sz + CUDA_ROLL_BLOCK_SIZE - 1) / CUDA_ROLL_BLOCK_SIZE;

    roll_f32_cuda<<<num_blocks, CUDA_ROLL_BLOCK_SIZE, 0, stream>>>(
        src0_d, dst_d, ne00, ne01, ne02, ne03, s0, s1, s2, s3);
}
~~~

kernel 的参数什么意义？根据：

~~~cpp
struct ggml_tensor * ggml_roll(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   shift0,
        int                   shift1,
        int                   shift2,
        int                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    // 这里保证了每个维度可shift的位数，不超过这个维度的长度
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    return result;
}
~~~

可以合理推测 kernel 调用时时传入的参数 s0 s1 s2 s3 分别是（从左向右，从fast变化的维度开始） `shift0, shift1, shift2, shift3`。并且保证了每个维度可以 shift 的位数，**不超过**这个维度的长度这与 `numpy.roll`/`torch.roll` 中的 roll 语义一致。

- 正数 `shift`：向索引值增加的方向循环移动（右 或 下）
- 负数 `shift`：向索引值减小的方向循环移懂（左 或 上）


##  Ground Truth

Torch & numpy 是对齐的，都可作为 Ground Truth

~~~py
src=np.array([[1,2,3],[4,5,6]])

# axis=0, +1 = 向下, -1 = 向上
# axis=1, +1 = 向右, -1 = 向左
dst = np.roll(src, shift=(-1, -2), axis=(0,1))
print(dst)

src1 = torch.tensor(src)
dst1 = torch.roll(src1, shifts=(-1, -2), dims=(0,1))

>>> dst1
tensor([[6, 4, 5],
        [3, 1, 2]])
>>> dst
array([[6, 4, 5],
       [3, 1, 2]])
~~~


## 如何并行

每个thread 负责一个元素，计算每一元素的 4 维索引（i0,i1,i2,i3）， 然后需要一个映射（函数），它用来执行从原始索引（4个）到roll之后索引（四个）的变化。


## code

block (256), grid ((元素个数总数 + 256-1) / 256)

~~~cpp
// 从实际case中总结得到的
static __forceinline__ __device__ int64_t idx_after_roll(const int64_t idx, const int64_t ne) {
    if (idx < 0) {
        return idx + ne;
    }
    if (idx >= ne) {
        return idx - ne;
    }
    return idx;
}

static __global__ void roll_f32_cuda(const float * __restrict__ src,
                                     float * __restrict__ dst,
                                     const int64_t ne00,
                                     const int64_t ne01,
                                     const int64_t ne02,
                                     const int64_t ne03,
                                     const int     s0,
                                     const int     s1,
                                     const int     s2,
                                     const int     s3) {
    // grid 和 block 都是只有x方向有值，故索引计算只有x
    const int64_t idx       = threadIdx.x + int64_t(blockDim.x) * blockIdx.x; 
    const int64_t stride0   = 1;
    const int64_t stride1   = ne00;
    const int64_t stride2   = ne00 * ne01;
    const int64_t stride3   = ne00 * ne01 * ne02;
    const int64_t n_elements = ne00 * ne01 * ne02 * ne03;

    if (idx >= n_elements) {
        return;
    }

    // [x,y,z,w] 
    // '%x'：表示x维度的局部索引
    // '/y': （y一定是一个stride）表示
    // i0 i1 i2 i3: i0 变化最快
    // 核心公式3。表达4个维度中元素索引
    const int64_t i0 = (idx / stride0) % ne00;
    const int64_t i1 = (idx / stride1) % ne01;
    const int64_t i2 = (idx / stride2) % ne02;
    const int64_t i3 = (idx / stride3) % ne03;

    // roll 之后的 4个维度中元素索引，从 ix 转换为 dx
    const int64_t d0 = idx_after_roll(i0 - s0, ne00);
    const int64_t d1 = idx_after_roll(i1 - s1, ne01);
    const int64_t d2 = idx_after_roll(i2 - s2, ne02);
    const int64_t d3 = idx_after_roll(i3 - s3, ne03);

    // src 的读取变化了
    // 反过来通过每一个维度的索引和stride 组合成 flat 的数据索引
    int64_t src_id =  1 * d0 + stride1 * d1 + stride2 * d2 + stride3 * d3;
    int64_t sdt_id =  1 * i0 + stride1 * i1 + stride2 * i2 + stride3 * i3;

    dst[sdt_id] = src[src_id];
}
~~~


