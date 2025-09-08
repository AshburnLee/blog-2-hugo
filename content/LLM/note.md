+++
date = '2025-08-31T12:49:42+08:00'
draft = true
title = 'Note'
tags = ["LLM"]
categories = ["LLM"]
+++


实例：

~~~py
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

~~~


tansformers 是 Huggingface 开发的开源库，提供了大量流行的预训练模型。

预训练模型是指模型结构（架构）和**训练好的参数权重（weights** 的集合。

比如：

~~~py
from transformers import BertForSequenceClassification

# 加载预训练模型和权重
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
~~~

`bert-base-uncased` 就包含了 BERT 模型的**架构定义**和**预训练后的参数权重**，你不需要自己训练即可直接调用。可以直接用于**推理**或**继续训练**。
