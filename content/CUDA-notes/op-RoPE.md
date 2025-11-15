+++
date = '2025-11-13T12:45:35+08:00'
draft = true
title = 'op RoPE'
tags = ["CUDA"]
categories = ["CUDA"]
+++


## Why RoPE

RoPE 的意义：给 Transformer 注入“相对位置信息”，让 Attention 天然感知 token 与 token 之间的距离。在实现中，这个“位置”通过“旋转角度”编码体现。

RoPE 就像给每个词一个时钟指针，位置越靠后，指针转得越多。没有它，模型会把 "猫在垫子上" 和 "垫子上在猫" 视为一样的。

RoPE 旋转的对象是计算 Attention 时的 Q 和 K 矩阵（Query 和 Key）。其形状是 `[seq_len, n_heads, head_dim]`

## 公式

旋转核心公式：

$$ \begin{bmatrix}
x'_i \\
x'_{i+d/2}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta &  \cos\theta
\end{bmatrix}
*
\begin{bmatrix}
x_i \\
x_{i+d/2}
\end{bmatrix}$$



## 锁定一个 case

输入 x 的维度是 [128,16,13,1]（col-major），

  - 第0维 128 `head_dim`：每个 head 的 embedding 维度。旋转对：(0,64),(1,65),...,(63,127)
  - 第1维 16 `n_heads`：多头数量（每个 head 独立 RoPE）
  - 第2维 13 `seq_len`：当前 prompt 的 token 数量（共 13 个位置）

`block_nums (208,1,1)`, 其中 208 是 `16*13`，即 `ne1*ne2`，表示输入有208个128。

关于 pos 参数， `const int32_t * pos = (const int32_t *) src1_d; 并且 src1` （第二个src）是的维度是 [13,1,1,1], src1_d 是src1 的数据部分。

  - `pos[t]` 表示 第 t 个 token 的绝对位置 ID。
  - 如何得到这个位置信息：由 tokenizer + 上下文历史决定：
    - 如：[0,1,2,...,11,12]（从头开始）
    - 或 [100,101,...,112]（续写时）
  - 表示 RoPE 旋转角度的起点。 $\theta_i = pos[t] \times 10000^{-2i/128}$ 

这个 13 是 `n_tokens`，就是 `n_prompt`，src1 维度 [13,1,1,1] 表示每个 token 提供一个 pos 值，4 维度因为 `ggml_tensor` 有4个维度。


## 小规模、直观、可手算的实例

x = np.arange(24).reshape(4, 2, 3, 1).astype(np.float32)：

~~~
head 0:                  head 1:
token0  token1  token2 | token0  token1  token2
[[ 0     4       8]      [[12    16      20]
 [ 1     5       9]       [13    17      21]
 [ 2     6      10]       [14    18      22]
 [ 3     7      11]]      [15    19      23]]
~~~


pos = np.array([0, 10, 20], dtype=np.int32)

旋转对？Neox 风格的旋转对：
~~~
x = [x[0], x[1], ..., x[63], x[64], x[65], ..., x[127]]
    └───────┴──────────┘      └───────┴────────────┘
         前半段                     后半段
~~~
旋转对：(x[0], x[64]), (x[1], x[65]), ...（x[63], x[127]）

x 添加位置旋转信息后，写入 y：

~~~
i = 0:
  y[0] = x[0] * cosθ - x[64] * sinθ
  y[64] = x[0] * sinθ + x[64] * cosθ

i = 1:
  y[1] = x[1] * cosθ' - x[65] * sinθ'
  y[65] = x[1] * sinθ' + x[65] * cosθ'
~~~

其中，每个 `i` 控制一个 频率（$\theta$ 随 i 指数衰减）, 每对 (`x[i]`,` x[i+64]`) 被旋转 不同角度。角度由 `pos × 10000^(-2i/128)` 决定。


## LLM 中关键超参数

- `n_heads`： 头数（如 32）
- `head_dim`： 头的长度，每个头的维度
- `seq_len`： 序列长度，即token的数量，不参与 hidden_size 计算
- `n_heads × head_dim = hidden_size`
- `hidden_size`：模型每个 token 的总嵌入维度（如 4096）
- `[seq_len, n_heads, head_dim]` 是 Q/K/V 的形状，


## KAQ：为什么 block 配置是（1,256,1），它覆盖的只有 128 个数？

block 配置是通过常量 CUDA_ROPE_BLOCK_SIZE=256 得到了，所以这是一个折中的通用block 配置。并不是为了性能。实际上只有前64个thread工作，其他空转.


## KAQ：block（1,256,1）这是连续的4个warp吗，与（256,1,1）比较呢？

col-major 下 `x[d][h][t]`，要 coalesced，**必须**让 warp 线程遍历 d 维，而不是 h 或 t，即**访问连续变化**的那个维度。

两种 block 配置的 thread 都是连续的。前者是 threadIdx.x 连续， 后者是 threadIdx.y 连续，硬件的连续性有warp调度器决定的。但是访问的连续性需要你来决定（通过设计各个 id 的计算方式）。

所以 不同的 block 配置，必须搭配不同的索引计算方式，才能发挥最佳性能。


## KAQ：code中有冗余

const int id_fast = 2 * (blockDim.y * blockIdx.y + threadIdx.y);
改为
const int id_fast = blockDim.y * blockIdx.y + threadIdx.y;


## KAQ：任务并行划分

~~~
grid(208,1,1)      =>     208 个 blocks
block(1,256,1)     =>    每个 block:
   ├─ x 维：1 线程 => 固定处理 1 个 (h,t)
   └─ y 维：256 线程 => 超配处理 128 维（前 64 个有效）

=> blockIdx.x  => 选 (h,t)
=> threadIdx.y => 选 维度对 (i, i+64)

blockIdx.x 表达“我处理第几个（num_head, token）”
threadIdx.y 表达“我处理第几个维度对儿”

两者配合覆盖所有元素
~~~


给出线程 id 与 idst，ix，channel_x，row_dst，i0 4个索引映射的 table，数据大小 [128,16,13], 以便我直观理解


## 线程id与数据索引映射
。。。
【这是关键的一步，这是写code 的依据】



## code

~~~cpp
static __device__ float rope_yarn_ramp(const float low, const float high, const int id_fast) {
    const float y = (id_fast / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}


// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
// 输入 theta_base，输出 cos_theta 和 sin_theta
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t id_fast, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    // 当 mscale=1 时，这是标准的 RoPE
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], id_fast) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

struct rope_corr_dims {
    float v[2];
};

// Standard RoPE
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    cos_theta = cosf(theta);
    sin_theta = sinf(theta);
    if (!forward) {
        sin_theta *= -1.0f;
    }
}



{
    // block grid 配置如下：
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // 旋转角度
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
}


// 输入 x 有3个维度 
// const int64_t ne0 = src0->ne[0]; // head dims
// const int64_t ne1 = src0->ne[1]; // num heads
// const int64_t ne2 = src0->ne[2]; // num heads

// 其中 const int32_t * pos = (const int32_t *) src1_d;
// 并且 src1 （第二个src）是的维度是 [13,1,1,1], src1_d 是src1 的数据部分
template<bool forward, bool has_ff, typename T>
static __global__ void rope_neox(
        const T * x,                     // 对 x 进行 RoPE 操作
        T * dst,                         // 将结果写入 dst
        const int ne0,                   // 128，变化最快（第一个维度）的维度大小
        const int ne1,                   // 16，变化其次块（第二个维度）的维度大小
        const int s1,                    // stride Of dim1：128
        const int s2,                    // stride of dim2：128*16=2048
        const int n_dims,                // 每个 head 的前 n_dims 维参与旋转，后半部分直接拷贝
        const int32_t * pos,             // 
        const float freq_scale,          // 1，
        const float ext_factor,          // 0，
        const float attn_factor,         // 1，
        const rope_corr_dims corr_dims,  // [24,41],
        const float theta_scale,         // 0.805842161,
        const float * freq_factors       //
) {
    // y 方向，即变化最快的实际值索引 0~127
    // block 负责覆盖变化最快维度的数据，故与grid 有关
    const int id_fast = blockDim.y * blockIdx.y + threadIdx.y;

    if (id_fast >= ne0) {
        return;
    }
    // x 方向，表示（num_head, token）索引 0~207
    const int id_flat_ht = blockDim.x * blockIdx.x + threadIdx.x;
    // 值 0,1,2，...，15，故是 head 索引
    // why ne1? 因为是 col-major (先排 head 后排 token)
    const int id_head       = id_flat_ht % ne1;  
    const int id_token      = id_flat_ht / ne1;  

    // 扁平化后的输输入/输出数据索引，与 核心公式2 一致
    // 直接体现了 col-major
    const int id_dest = id_fast + ne0 * id_flat_ht;
    const int ix      = id_fast + s1 * id_head + s2 * id_token;

    // 超出 n_dims的位置元素直接copy
    if (id_fast*2 >= n_dims) {
        dst[id_dest + id_fast + 0] = x[ix + id_fast + 0];
        dst[id_dest + id_fast + 1] = x[ix + id_fast + 1];
        return;
    }

    // 不同的token 
    const float theta_base = pos[id_token] * powf(theta_scale, (float)id_fast);
    const float freq_factor = has_ff ? freq_factors[id_fast] : 1.0f;
    float cos_theta;
    float sin_theta;
    // 计算 cos_theta & sin_theta 带 $\theta$
    // 当前 case 没有 YaRN 不生效
    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, id_fast*2, ext_factor, attn_factor, cos_theta, sin_theta);

    // Rotation 核心计算
    // 1. 获取旋转对
    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[id_dest + 0]        = x0*cos_theta - x1*sin_theta;
    dst[id_dest + n_dims/2] = x0*sin_theta + x1*cos_theta;
}
~~~


