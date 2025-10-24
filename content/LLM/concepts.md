+++
date = '2025-08-31T12:49:41+08:00'
draft = false
title = 'Concepts'
tags = ['General']
categories = ["LLM"]
+++


## JSON Lines vs JSON

JSON Lines（也称为 JSONL）是一种数据格式，其中每个记录都是一个单独的 JSON 对象，并以换行符分隔。如：

~~~jsonl
{"name": "Alice", "age": 30}
{"name": "Bob", "age": 25}
...
~~~

相比之下，传统的 JSON 数据通常包含一个数组或对象，如下所示：

~~~json
{
  "people": [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
  ]
}
~~~

- JSON Lines 更适合处理**大量独立的数据条目**，因为每条记录都是独立的实体，易于逐个读取和处理。
- 传统 JSON 更适用于**嵌套结构和复杂的多层关系**，更适合一次性加载整个数据集到内存中进行分析或操作。


## 查看内存使用

在 Python 中测量内存使用的一种简单方法是使用 psutil 库。数据集在磁盘上的大小，使用 dataset_size 属性

~~~py
import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

print(f"Dataset size in bytes: {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")
~~~


##  HTTP 请求的头部信息 header

请求中非常重要的组成部分，用来传递额外的**元信息**，提升请求的准确性和安全性。内容包括：

- 指定内容类型（Content-Type）：告诉服务器发送的数据类型，比如 JSON、表单等。

- **身份验证（Authorization）**：提供令牌或凭证，用于访问需要权限的接口。

- 浏览器标识（User-Agent）：标明请求来自哪个客户端（浏览器、应用等），有助于服务器识别和响应适配请求。

- 语言偏好（Accept-Language）：告知服务器客户端期望回复的语言。

- 缓存控制（Cache-Control）：指定请求的缓存策略。

- Cookies：传递存储的Cookies，维持会话状态。


## Pooling in NLP VS Pooling in the rest of DL

- NLP中 的 pooling 是 将 Transformer 模型输出的多个 token 嵌入向量（每个 token 对应一个向量）通过某种**聚合操作**（如平均池化、最大池化或取特定 token 的嵌入）组合成一个**单一的句子级别向量表示**。方法包括：Mean Pooling、Max Pooling、`CLS pooling`

- 在深度学习中其他模型，如 CNN 中，pooling 是一种降维操作，用于从输入特征图中提取关键信息，减少计算量并增强模型的泛化能力。即 **缩小特征图的尺寸，同时保留重要特征，降低过拟合**风险。方法包括：Max Pooling、Average Pooling、Global Pooling

都是从高维信息中提取紧凑信息，用于后续任务。


## 内存映射

是 **RAM** 与**文件系统**存储之间的映射。RAM是物理内存，内存映射将文件内容映射到**虚拟内存**，**操作系统OS**决定**哪些部分加载到 RAM，或在需要时将其换出（paging）RAM**。内存映射是映射到 RAM ，因为虚拟内存提供隔离、灵活性、按需加载和安全性，进程通过虚拟地址操作文件数据，操作系统管理实际的物理内存分配。


## Asymmetric Semantic Search

从长文档（如维基百科、知识库）中检索与**简短问题相关**的段落。而**对称语义搜索**是查询和文档长度、复杂性相似（如句子对句子），使用相同的嵌入模型。


## 分词器不是模型通用的

原因是：
1. **语料差异**：分词器的词汇表和规则基于训练语料生成。不同模型的语料（如英文、中文、代码）差异导致分词器不同。
2. **分词算法选**择影响 token 的拆分方式和词汇表结构。
3. **特殊token**：BERT 使用 `[CLS]` 用于分类任务，而 GPT 没有 `[CLS]`，因为它是自回归模型。
4. 词汇表大小和内容。
5. 任务要求：目标语言不通，有的是代码，分词的方法必然不同。


## 模型能力的蒸馏

**Distillation**（知识蒸馏） 是一种模型压缩技术，通过将一个复杂、大型模型（教师模型）的知识转移到一个更小、更高效的模型（学生模型），使其在保持类似性能的同时减少计算和存储需求。

通过知识蒸馏过程，将教师模型的核心能力（如语言理解、生成、任务特定技能）转移到学生模型，使学生模型能够以更低的资源开销复现这些能力。


## 冷启动

冷启动：刚启动时，由于缺少足够的历史数据、用户反馈或预训练信息，导致无法有效运行或提供高质量输出。

相对的 有热启动。在启动时利用已有的数据、缓存、预训练参数或运行时状态，快速达到高效性能，减少初始化开销。fine-tuning 和 迁移学习 是热启动的一种。热启动涵盖的更广，任何利用初始状态的场景，不仅限于模型训练，还包括缓存、运行时优化等


## KL 离散度

KL 散度（Kullback-Leibler Divergence）是一种衡量两个概率分布差异的指标，在强化学习（如 PPO、GRPO）中常用于。

- 正则化：限制新策略（优化后的模型输出分布）与旧策略（初始模型分布）的偏离，防止过大更新导致不稳定。

- 稳定训练：确保模型在优化奖励（如 reward_function 的输出）时，不会完全偏离预训练行为。


## LoRA

LoRA（Low-Rank Adaptation）微调技术，用于在大型语言模型（LLM）上进行**参数高效微调**（Parameter-Efficient Fine-Tuning, PEFT）。LoRA 通过在**预训练模型的权重**矩阵上**添加低秩分解的更新矩阵，减少需要训练的参数数量，同时保持模型性能**。LoRA 配置是指在微调过程中设置的参数，用于定义 LoRA 的行为和特性。

比如：
~~~py
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1
)
~~~

### r 

r 是 LoRA 适配器的低秩矩阵的秩（rank）。LoRA 将权重更新分解为两个低秩矩阵 $ A $ 和 $ B $，使得权重更新 $ \Delta W = A \cdot B $，其中 $ A \in \mathbb{R}^{d \times r} $，$ B \in \mathbb{R}^{r \times k} $，$ r $ 远小于原始权重矩阵的维度 $ d $ 和 $ k $。

控制 LoRA 适配器的参数量。较低的 r 意味着更少的参数（更高效），但可能限制模型的表达能力；较高的 r 增加参数量，增强表达能力但计算成本更高。r=8 通常足够捕捉任务特定的模式，同时保持高效。

### lora_alpha

是 LoRA 适配器输出缩放因子，用于调整低秩更新的幅度。它的作用是控制 LoRA 适配器对模型输出的影响强度。较大的 lora_alpha 使适配器更新的影响更显著，较小的值则使更新更保守。

### target_modules

指定应用 LoRA 适配器的 Transformer 模型模块。这些模块是**模型权重矩阵的名称**，LoRA 只更新这些模块的权重，而其他部分保持冻结。

- `q_proj`：查询（Query）投影矩阵，计算 $ W_Q $。
- `k_proj`：键（Key）投影矩阵，计算 $ W_K $。
- `v_proj`：值（Value）投影矩阵，计算 $ W_V $。
- `o_proj`：注意力输出投影矩阵，计算 $ W_O $。

- `gate_proj`：FFN 中 gating 层的权重，通常用于控制信息流（如在 LLaMA 或 Qwen 模型中）。
- `up_proj`：FFN 中上投影层（扩展维度）的权重。
- `down_proj`：FFN 中下投影层（恢复维度）的权重。

上述覆盖了 Transformer 核心组件。


### lora_dropout

以 lora_dropout=0.1 的概率随机将 LoRA 适配器的部分输出置为 0，增强模型的泛化能力。



## KV 缓存（Key-Value Cache）存储注意力机制的中间结果

存储注意力机制的 Key (K) 和 Value (V) 向量。在生成序列时，每次只计算新 token 的 KV，并缓存先前所有 token 的 KV，避免重复计算。

作用是加速推理：传统注意力计算复杂度为 O(n^2)，KV 缓存将增量生成复杂度降为 O(n)，显著提高速度。缓存 KV 减少重新计算开销，防止内存爆炸。

如果没有它，每次生成 token 需重新计算所有先前 token 的 KV。导致OOM。推理了性能严重下降。


## @ 注意力头数量？

注意力头数 是指 Multi-Head Self-Attention 中**并行计算**的注意力单元数量。每个“头”独立处理输入序列的查询（Query）、键（Key）和值（Value）向量（**QKV**），计算注意力得分，功能是**捕获 token 之间的不同关系**。

每个头学习不同模式（如语法、语义），**增强模型表达力**。多头并行处理子空间，提高效率。更多头数捕获复杂 token 关系。

小模型 10 个 head 以内，中模型 20 个 head 以内，大型模型 30~100 个 head。


## @ 一个模型后缀 Q4_K_M 表示什么？

1. Q4：表示模型权重采用 4-bit 整数量化（Quantized to 4 bits）。每个权重从浮点数（如 FP16，16-bit）压缩到 4-bit 表示，显著减少内存占用。

2. K：表示使用 `K-quants`（K 量化），一种高级量化技术，结合不同位精度（如 4-bit 和 6-bit）优化精度和效率。`K-quants` 通过对权重分组（block-wise quantization）并分配不同量化级别，平衡性能和压缩率。

3. M：表示 Medium（中等）量化级别，是 `K-quants` 中的一种配置。`K-quants` 有多个级别（如 `Q4_K_S`、`Q4_K_M`、`Q4_K_L`），M 表示中等精度，相比 S（Small）更高，相比 L（Large）更节省内存。

`Q4_K_M` 在精度和内存占用间取得平衡，适合大多数推理任务。


## @ mmap ？

mmap 是 Linux/Unix 系统调用（memory map），用于将文件（或设备）的部分或全部内容**映射到**进程的 虚拟地址空间，使文件数据可以像内存一样通过指针访问。

什么叫 “映射到”？

- 内存映射（mmap）：llama.cpp 使用操作系统提供的 mmap 系统调用，将磁盘上的 GGUF 文件映射到进程的虚拟地址空间。
- 虚拟地址空间：**每个进程有自己的虚拟内存地址范围**（由操作系统动态管理，可能映射到 RAM、磁盘（如交换空间或文件）或其他存储区域），**CPU 通过虚拟地址访问数据**。mmap 将 GGUF 文件的权重数据映射到这些地址，使程序可以像访问内存一样访问文件内容。
- 按需加载：映射后，权重数据**仍主要存储在磁盘（或其他物理区域）上**，操作系统只在 CPU 访问特定数据时，将对应的文件部分（页面）加载到物理 RAM（页面缓存）。未访问的部分不会占用 RAM。
- 直接访问：**CPU 通过虚拟地址**直接读取权重数据，无需显式 I/O 操作，减少内存拷贝开销。

如何访问？权重数据映射到虚拟地址空间后，程序**需要一个 handle 来访问这些数据**。这个 handle 实际上是 mmap **返回的内存指针**，指向虚拟地址空间中映射的区域。


## @ 本地跑模型时，回答似乎总是来来回回回答同一个问题，重复相同的内容，不会停止了，为什么？

`llama-cli -m ../models/Qwen3-1.7B-Q4_K_M.gguf -no-cnv --prompt "Hello, tell me something about llama.cpp"`

**文本退化 text degeneration**

user: what is the result of 1 + 1? 1 + 1 = 2. So, the result is 2.

output: Okay, so the question is 1 = 2. So, the result is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the answer is 2. So, the 。。。

LLM（如基于 Transformer 的模型）是 autoregressive 自回归：给定输入 prompt，模型计算下一个 token 的概率分布（logits），然后通过**采样策略选择一个 token**，追加到序列中，并重复这个过程直到达到停止条件（如最大 token 数或 EOS token）。

如果 prompt 或初始生成引导模型进入“高概率陷阱”，模型会强化这个模式，因为后续预测基于已生成的文本。原因：

- 正反馈循环：一旦模型生成一个重复 token，后续上下文**会强化它**。模型计算下一个 token 时，这个短语的概率会上升，导致无限循环。
- 采样机制缺陷：贪婪或低温度采样忽略低概率选项，导致模型偏好常见模式。
- 模型大小和训练数据影响：较小的模型（如 <7B 参数）更容易重复，因为它们泛化能力弱。


## @ 运行大模型推理的硬件资源

1x24GB card：指24GB 显卡容量


## 量化

通过减少模型参数的精度来压缩模型大小，从而降低计算成本并提高推理速度。量化修改权重和激活的精度。

里程碑

- 2017: BinaryConnect
- 2018: Quantization-Aware Training (QAT)
- 2019: Post-Training Quantization (PTQ)
- 2020: GPTQ (Accurate Post-Training Quantization for Generative Pre-trained Transformers)
- 2021: SmoothQuant
- 2022: AWQ (Activation-Aware Weight Quantization)
- 2023: LLM-QAT (Data-Free Quantization-Aware Training)
- 2023: QLoRA (Quantized LoRA)
- 2024: ZeroQuant-FP
- 2025: SST (Self-training with Self-adaptive Thresholding)

PTQ 常用技术 GPTQ、AWQ、SmoothQuant
QAT 常用技术 LLM-QAT、QLoRA
混合精度量化 常用技术 Outlier Suppression、ZeroQuant-FP
量化注意力 KV-cache

均匀量化（Uniform Quantization）：将浮点数均匀映射到固定范围的整数（如 `INT8 [-128, 127]` 或 `INT4 [-8, 7]`），使用线性缩放因子
`K-quant`。许多量化技术都是基于均匀量化的。

`llama.cpp` 中使用到的量化技术


## KL divergence

在信息论中是衡量两个分布差的方法。

假设两个二元概率分布 $P$ 和 $Q$（事件 A 和 B 的概率），公式：
$$D_{\text{KL}}(P \| Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$$
使用自然对数 $\log$（ln）计算。

### Case 1: $P$ 和 $Q$ 的概率相同

$P = [0.5, 0.5]$，$Q = [0.5, 0.5]$

计算：

$$D_{\text{KL}}(P \| Q) = 0.5 \log \left( \frac{0.5}{0.5} \right) + 0.5 \log \left( \frac{0.5}{0.5} \right) = 0.5 \log 1 + 0.5 \log 1 = 0 + 0 = 0$$

解释：分布相同，KL 散度为 0，表示两个分布无差异。

### Case 2: $P$ 和 $Q$ 的概率相差很大

$P = [0.9, 0.1]$，$Q = [0.1, 0.9]$

计算：

$$D_{\text{KL}}(P \| Q) = 0.9 \log \left( \frac{0.9}{0.1} \right) + 0.1 \log \left( \frac{0.1}{0.9} \right) = 0.9 \log 9 + 0.1 \log \left( \frac{1}{9} \right)$$

$$\log 9 \approx 2.197, \log \left( \frac{1}{9} \right) = -\log 9 \approx -2.197$$
$$= 0.9 \times 2.197 + 0.1 \times (-2.197) \approx 1.977 - 0.220 = 1.757$$

解释：**分布差异大，KL 散度较大**，表示 $Q$ 近似 $P$ 时损失信息量多。

