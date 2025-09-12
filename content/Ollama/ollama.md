+++
date = '2025-03-01T12:50:42+08:00'
draft = false
title = 'Ollama'
tags = ["Ollama","llama.cpp"]
categories = ["Ollama"]
+++



Ollama 是一个开源的本地大型语言模型（LLM）运行框架，旨在降低使用大型语言模型的门槛，同时确保数据隐私。

硬件加速：虽然主要针对 CPU 优化，但 Ollama 也可能支持 GPU 或 TPU 加速，以进一步提升推理速度

## 使用

适用场景：CPU/GPU 显存不足或计算速度慢。
~~~sh
# 查看当前模型量化类型
ollama show deepseek-r1 --modelfile
# 替换为更低精度的量化版本（如 2-bit）
ollama pull deepseek-r1:2b
~~~

适用场景：使用 AMD/NVIDIA GPU 时显存未被充分利用。
~~~sh
# 尝试增加卸载层数（如 -ngl 40）
ollama run deepseek-r1 --gpu-layers 40
# 调整分块模式（如 row 或 layer）
--split-mode row # 修改 GPU 核函数的分块大小（-split-mode 参数）以匹配硬件特性
~~~

适用场景：CPU 利用率低或多卡并行效率不足。
~~~sh
# 设置 CPU 线程数为物理核心数（如 16）
ollama run deepseek-r1 --threads 16
# 启用多 GPU 并行
CUDA_VISIBLE_DEVICES=0,1 ollama run deepseek-r1
~~~

适用场景：内存/显存不足导致频繁交换。
~~~sh
# 减小分块大小以减少峰值内存占用
ollama run deepseek-r1 --chunk-size 512
~~~


# llama.cpp

llama.cpp 是一个轻量级的 C++ 实现，旨在高效地运行 LLaMA（Large Language Model Meta AI）模型。它通过简化的 API 与 Ollama 集成，使得用户能够在本地环境中快速部署和使用语言模型。
最佳实践

  - 使用命令行工具：通过命令行启动模型，使用参数来指定模型路径和生成设置。
  - 优化性能：根据硬件配置调整参数，例如线程数和批处理大小，以提高响应速度和效率

# whisper.cpp

whisper.cpp 是 OpenAI 的 Whisper 语音识别模型的 C++ 实现，专注于快速和高效的语音转录。此实现旨在减少延迟并降低计算资源消耗，适合需要实时语音识别的应用。
最佳实践

  - 本地运行：可以在本地或远程服务器上运行 whisper.cpp，确保网络连接稳定以减少延迟。
  - 音频格式：确保输入音频文件符合要求（如WAV格式），并使用 ffmpeg 进行必要的格式转换

# stable-diffusion.cpp

stable-diffusion.cpp 是用于生成图像的 Stable Diffusion 模型的 C++ 实现。它相较于其他实现（如Automatic1111）更易于安装，但每次请求时都需要加载模型，这可能导致延迟。
最佳实践

  - 保持模型加载：尝试使用持久化服务或其他工具（如Koboldcpp）来避免每次请求都重新加载模型，从而提高生成速度
  - 优化GPU使用：根据GPU资源配置进行优化，以提高图像生成效率。

# GGUF 模型

GGUF（Generalized Unified Format）是一种新型模型格式，旨在支持多种机器学习框架和工具。Ollama允许用户轻松运行来自Hugging Face等平台的GGUF格式模型。

最佳实践

  - 简单命令运行：使用 `ollama run <model_name>` 命令快速启动并使用GGUF模型，无需复杂设置
  - 配置文件管理：创建 Modelfile 以便于管理和导入GGUF模型，使得模型的使用更加灵活和高效

本地成功运行，见 Jetson-ollama.md


# llama.cpp

开源，大语言模型轻量级推理框架，高效，占用资源低，适合在本地运行语言模型。 详见 [LLM](../LLM/) 中与llama.cpp 相关。

可以运行任何语言模型都吗？

非也，llama.cpp 设计是用来运行 基于 Transformer 架构的模型，尤其是 LLaMa 及其变种。 确认模型是否是基于 LLaMA 或类似的 decoder-only Transformer。
