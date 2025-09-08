+++
date = '2025-08-31T12:13:33+08:00'
draft = false
title = 'Concepts'
tags = ["Agent","General"]
categories = ["Agent"]
+++


# Serverless API & Server-based API

Serverless API:

  - 基础设施管理：无需管理服务器。云服务提供商负责所有底层基础设施的管理。
  - 扩展性：自动扩展，根据流量需求动态调整资源。
  - 成本：按需付费，只需为实际使用的计算资源付费。
  - 部署：通常通过函数即服务 (FaaS) 平台部署，如 AWS Lambda、Azure Functions、Google Cloud Functions。
  - 运维：运维工作量较少，主要关注业务逻辑。

Server-based API:

  - 基础设施管理：需要自行管理服务器，包括配置、维护和扩展。
  - 扩展性：需要手动配置和扩展服务器，以应对流量高峰。
  - 成本：需要为服务器的运行时间付费，即使在低流量时段也需要支付费用。
  - 部署：通常部署在传统的服务器或虚拟机上，如 Apache、Nginx 等。


## 文本检索 Text retieval

其核心作用是根据用户输入的查询（query），快速高效地在庞大的文档集合中筛选出与查询最相关的文档或文本片段。

BM25Retriever，是 BM25 的python library，基于词频概率统计估计相关性，是目前经典且广泛使用的排名算法。

SentenceTransformers 是基于embedding 的文本检索方法 Python library。


## RAG 检索增强生成 用途

~~~py
# doc is a list of documents
bm25_retriever = BM25Retriever.from_documents(docs)
# query is a string
results = bm25_retriever.invoke(query)
~~~

RAG 模式的 AI Agent 适用于需要**结合外部知识**生成准确回答的场景，如知识库问答、搜索整合、客服、法律分析、医疗支持和教育。



## SGLang （Structured Generation Language） 与 vLLM（Vectorized Large Language Model Inference）

- SGLang：是复杂多轮交互及结构化生成的语言模型服务框架。适用于需要**多步骤**任务、**多GPU协作**、**大规模**模型复杂应用。需要**多轮复杂任务**支持场景。

- vLLM：是高吞吐量、高效内存管理的单轮推理框架。适用于高并发请求、多用户低延迟响应场景。需要**极致吞吐**和**易用性**的场景。
极致吞吐和易用性（vLLM）。

- Ollama

- LMStudio： 桌面化多模型UI与API服务器。适用于“开箱即用”的 LLM **桌面客户端与推理引擎**场景，内置聊天界面，也可作为本地API服务器使用。

- MLX-LM： 专为 Apple Silicon

- llama.cpp： 是 Llama 及同类大模型的高性能本地 C/C++ 实现，被广泛用于框架底层和自定义推理。有**极高的性能和扩展性**。需要极致性能/资源优化的本地部署、高度自定义下的嵌入式系统、移动端推理。

- KTransformers: 是**高性能、低成本**本地大型模型推理优化的 Python 框架。适用于高性能推理、科研工程优化、自定义部署、对响应速度和内存利用要求极高的应用。

## GAIA

A benchmark for General AI Assistants
