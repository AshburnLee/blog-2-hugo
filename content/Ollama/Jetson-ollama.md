+++
date = '2025-08-31T12:50:42+08:00'
draft = false
title = 'Jetson Ollama'
tags = ["Ollama","Jetson"]
categories = ["Ollama"]
+++


NVIDIA Orin 是一款专为**自动驾驶汽车**和**机器人**设计的高性能系统级芯片，包含新一代 Ampere GPU 架构和 Arm Hercules CPU 内核，以及深度学习加速器和计算机视觉加速器。

应用领域：Orin 芯片不仅适用于自动驾驶汽车，还广泛应用于机器人、工业边缘计算等领域。

支持C/C++, python, cuda, pytorch, ROS (Robot Operating System), JetPack SDK, DeepStream, VScode 

TensorRT 是 NVIDIA 开发的一个高性能深度学习推理 SDK。它不是完全开源的.

`Linux for Tegra (L4T) `是 NVIDIA 为其 Tegra 系列系统芯片 (SoC) 开发的嵌入式 Linux 发行版。它主要用于 NVIDIA Jetson 系列开发套件等嵌入式系统。L4T 提供了运行在 Tegra SoC 上的内核、驱动程序、库和工具，支持各种应用，包括机器人、人工智能、自动驾驶和媒体处理等。 它包含了 NVIDIA 专有的驱动程序，以充分利用 Tegra SoC 的硬件加速功能。不同的 L4T 版本支持不同的 Tegra 系列芯片和功能。 例如，较新的版本可能支持 Vulkan 和更新的 CUDA 版本。 开发者可以使用 L4T 来构建和部署各种嵌入式应用。


# 部署 Ollama

Ollama 中的大模型（如 Llama、Qwen、DeepSeek、Gemma 等）本身与 Ollama 框架的本地使用是完全免费的。你可以免费下载、部署这些开源大模型在自己的设备上，运行、推理和本地开发时都免费，唯一成本是自己的硬件资源，如 Jetson 设备。


## 1. 安装 ollama && 下载 Ollama 模型

~~~sh
# 下载安装或者是 update ollama
curl -fsSL https://ollama.com/install.sh | sh
# 如果网络有问题，那么下载文件后解压
# 找到指定版本和平台的 压缩包：https://github.com/ollama/ollama/releases
sudo tar -C /usr -xzf ollama-linux-arm64.tgz  # 下载或更新
~~~

docker 什么时候使用：

~~~sh
# 第一次先下载 image, 代理配置好后，就能正确下载了
sudo docker pull dustynv/ollama:r36.4.3
# models cached under your user's home directory
docker run --runtime nvidia --rm --network=host -v ~/ollama:/ollama -e OLLAMA_MODELS=/ollama dustynv/ollama:r36.4.3
~~~

## 2. 下载 image 后，运行 ollama

在一个 terminal 中开启 ollama microserve 服务：
`ollama serve`

检查服务是否开启：
`ollama ps`

在另一个terminal中 执行模型：
`ollama run deepseek-r1:1.5b`

常用命令：
~~~sh
ollama list # 列出已有的模型及其大小
ollama pull deepseek-r1:1.5b  # 更新已有的模型
ollama rm model_name # 删除已安装的
~~~

模型命令祥见：https://ollama.com/library/deepseek-r1:1.5b


## 3. 在 Agent 脚本中使用本地 LLM 的片段

~~~py
import os
os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"

infer_server_url = "http://localhost:11434/v1"
model_name = "deepseek-r1:1.5b"  #

# 保留了 openai 相关的参数名，保持接口一致和兼容
vision_llm = ChatOllama(
    model=model_name,
    openai_api_base=infer_server_url,
    openai_api_key="none",  # Ollama 本地模型不需要云端 Key 授权
    temperature=0,
)
~~~

其中 `/v1` 表示 Ollama 兼容的 OpenAI API 第1版（version 1）接口路径，是一个标准的API路径版本控制与规范。


# 探索 ollama 的 LLM （所有可得到的 1GB 左右的 LLM）！

1. qwen3:1.7b          1.4GB 输入是TEXT only（所有大小）【可】中英日翻译、有实用工具，验证确实可以 tool calling！

3. llama3.2:1b         1.3GB 输入是TEXT only，有 tool calling，此外它的能力有些弱。有 llama3.2-vision, 但最小是 7.8GB。
4. moondream:1.8b      1.7GB 输入是TEXT、Image。轻量级的 vision language model。【可】图片文字提取能力。你需要在使用时提供图片作为输入。`ollama run moondream:1.8b "这个图中有什么内容？ ./btm.png"`。功能太单一，只能描述图片的内容，不支持提取文字。
5. ~~Phi-3 Vision 4.2B  。。。。。 试试有OCR功能否~~

1. deepseek-r1:1.5b    没有调用 tools 的能力，不能用
2. gemma3:1b           0.8GB 输入是TEXT（其他大Size可输入Image）可以中日问翻译、当前版本只有 TEXT 输入。不支持tool calling。vision能力（Describe images、Recognize visual features、Understand visual concepts） 
2. deepseek-r1:8b      5.2GB，（model requires more system memory (4.3 GiB) than is available (2.7 GiB)）
3. gemma3:4b           3.3GB 输入 TEXT、Image。。（model requires more system memory (4.3 GiB) than is available (2.9 GiB)）

各个模型的 ollama 主页中有 **Huggingface 页面**，进入就有这个模型的使用方法。


# 调用 LLM 的方法 ***

两个端点：

1. `http://localhost:11434/v1` 是用于兼容 openAI version 1 的接口，[来源](https://github.com/ollama/ollama/blob/main/docs/openai.md)
2. `http://localhost:11434/api` 是 Ollama 原生的，字Ollama 开始，这个端点就是其核心功能的接口。[来源](https://github.com/ollama/ollama/blob/main/docs/api.md)

Ollama API 端点 `http://localhost:11434/api/` 路径下**包含多个子端点**，用于**与模型交互、管理模型和执行其他具体**操作。

- `curl http://localhost:11434/api/generate` 用于生成文本。
- `curl http://localhost:11434/api/chat`  用于对话模式，生成基于聊天历史的响应
- `curl http://localhost:11434/api/tags` 列出所有可用的模型标签（tags）。
- `curl http://localhost:11434/api/show -d '{"name": "llama3.2:1b"}'` 显示指定模型的详细信息
- `curl http://localhost:11434/api/pull -d '{"name": "llama3.2:1b"}'` 从远程仓库拉取模型到本地
- `curl http://localhost:11434/api/embed -d '{...}'`  生成文本的嵌入向量
- 其他。。。


## 1. Ollama API 的 cml  【端点api】

命令行中与模型交互：

~~~sh
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
~~~


## 2. REST API  【端点api】

这是目前运行和集成大量语言模型推理的**通用标准接口**方式。

requests.post 是 Python 中 requests 库的函数，用于向指定 URL（这里是 infer_server_url 本地 Ollama 服务）发送 HTTP POST 请求，请求体是 JSON 格式的 payload。
你提交给服务器的是一段文本提示（prompt）和生成参数，服务器运行 LLM 模型生成文本回应。

这里的 payload 对应具体模型的。。。，具体见模型的定义。

~~~py
import requests

infer_server_url = "http://localhost:11434/api/generate"  # 本地 Ollama 模型服务地址
model_name = "deepseek-r1:1.5b"

payload = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": "明天天气如何？"}
    ]
}

response = requests.post(infer_server_url, json=payload)
result = response.json()
~~~

其中的`payload`如何提供？它们是 Ollama `/api/generate` 端点的核心参数，用于定义输入、输出格式和响应方式。[所有参数见这里](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion)

其中有个参数是 `options`: 当前模型参数，存在于 **Modelfile** 中，比如 `temperature`。


## 3. 通过兼容 OpenAI 的 API 即 Python 库【端点v1】

当你使用支持 openAI api 的 Python 库，如 LangChain、LlamaIndex 时，可以使用兼容的 OpenAI API: 比如 ChatOllama 使用 OpenAI 的 API 结构（例如 `/v1/chat/completions` 端点）。优点是可以从ChatGPT 等 openAI的模型切换到Ollama 本地模型。如下：

~~~py
from langchain_ollama import ChatOllama

infer_server_url = "http://localhost:11434/v1"  # openai api 的端点
model_name = "qwen3:1.7b"
qw_llm = ChatOllama(
    model=model_name,
    openai_api_base=infer_server_url,
    openai_api_key="none",  # Ollama 本地模型不需要云端 Key 授权
    temperature=0,
)
# 以 message 形式发送对话请求
response = qw_llm.invoke([
    {"role": "user", "content": "Why is the sky blue?"}
])
print(response.content)
~~~

看，你需要提供 `openai_api_base` 参数值，所以这种方法是原本支持 openAI API 的，你需要提供设置 API 的基础 URL: `http://localhost:11434/v1`。同时 `openai_api_key` 的对，于 Ollama 的本地部署，通常不需要，故设置 none。

Under the hood，**ChatOllama 的调用方式实际上是对 Ollama 的 `/v1/chat/completions` 端点的封装**。

调用方式: 通过 ChatOllama 实例调用模型，发送对话请求，请求通常是 `messages` 格式，包含 `role` 和 `content`，然后并接收模型的响应。

然后 ChatOllama 会将请求转换为 OpenAI 兼容的 JSON 格式，发送到 `http://localhost:11434/v1/chat/completions` 端点。Ollama 服务处理请求并返回响应，ChatOllama 再将响应解析为 Python 对象。


## 4. 端点 api 和端点 v1 的区别

端点 v1 原生支持 tools 和 tool_choice；专为多轮对话设计; 支持 OpenAI 客户端库（如 openai Python SDK）; 适合与 LangChain、LlamaIndex 集成，工具调用。

端点 api 无原生支持 tools 调用，需通过 prompt 和 format: "json" 模拟；`/api/chat` 支持对话，`/api/generate` 适合单次生成; 需要 Ollama 客户端或手动 HTTP 请求; 适合简单生成任务、模型管理或自定义逻辑

总结：端点 v1 更强大。


## 5. JavaScript library 

略


# 更多关于 Ollama

llama4:latest   

Size: 67GB          模型文件本身的磁盘大小

Context: 10M        模型最大的上下文窗口（Context Window）容量约为 10M tokens（Token）。

Input: Text, Image  支持多模态输入，文本和图片

pull 模型到本地之后，模型文件位于 `~/.ollama/models`，包括 manifest 和 blobs（权重文件）


## 关于 ollama serve 模型的信息

这些环境变量是影响 Ollama 本地推理服务行为的配置参数，通过启动 serve 时，设置这些系统环境变量控制

~~~txt
INFO server config env="
map[CUDA_VISIBLE_DEVICES:                          指定可见的 CUDA GPU 设备编号，空代表不限制，通常用于多GPU选择。例如设置0表示只使用第0号GPU
GPU_DEVICE_ORDINAL: 
HIP_VISIBLE_DEVICES:                               针对 AMD GPU 的对应环境变量
HSA_OVERRIDE_GFX_VERSION: 
HTTPS_PROXY: 
HTTP_PROXY: 
NO_PROXY: 
OLLAMA_CONTEXT_LENGTH:2048                         模型的上下文窗口大小，表示模型一次可以处理的最大Token长度，默认是2048。
OLLAMA_DEBUG:false                                 是否开启调试模式，开启后会输出更详细的日志，默认关闭
OLLAMA_FLASH_ATTENTION:false                       是否启用闪存注意力机制，这通常是某些高效推理优化，默认关闭。
OLLAMA_GPU_OVERHEAD:0                              给 GPU 资源预留的额外开销比例，0 表示无预留，方便调度资源。
OLLAMA_HOST:http://127.0.0.1:11434                 Ollama 服务监听的 HTTP 地址和端口，本地默认11434端口。
OLLAMA_INTEL_GPU:false 
OLLAMA_KEEP_ALIVE:5m0s                             模型加载后缓存保持时间，默认为 5 分钟，超过此时间模型将被卸载释放资源。
OLLAMA_KV_CACHE_TYPE: 
OLLAMA_LLM_LIBRARY:                                指定底层LLM推理库（如 llama.cpp、ggml等），空表示默认 
OLLAMA_LOAD_TIMEOUT:5m0s 
OLLAMA_MAX_LOADED_MODELS:0 
OLLAMA_MAX_QUEUE:512 
OLLAMA_MODELS:/home/junhui/.ollama/models          模型文件存放目录路径。
OLLAMA_MULTIUSER_CACHE:false 
OLLAMA_NEW_ENGINE:false 
OLLAMA_NOHISTORY:false                             是否关闭会话历史记录保存，默认保存。 
OLLAMA_NOPRUNE:false                               是否禁止模型自动剪枝（释放内存的机制），默认允许自动剪枝。
OLLAMA_NUM_PARALLEL:0                              支持的最大并发请求数，0 可能表示没有限制或默认串行。
OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:* app://* file://* tauri://* vscode-webview://* vscode-file://*] OLLAMA_SCHED_SPREAD:false ROCR_VISIBLE_DEVICES: http_proxy: https_proxy: no_proxy:]"
~~~

## ollama上的每个模型有 model 、template 、license、params 等信息




## 我有个本地的大模型，如何将它发布到 ollama

1. Ollama 使用 GGUF 格式（由 LLaMA.cpp 开发）作为其模型文件格式。如果你的本地模型不是 GGUF 格式（例如 PyTorch、Hugging Face 格式），需要先将其转换为 GGUF。

2. Ollama 使用 Modelfile 定义模型的元数据、参数和提示模板。这是发布模型的关键步骤。

3. 将本地模型导入 Ollama 注册表，创建自定义模型。

4. 在本地测试模型以确保其正常运行。`ollama run <model_name>`。

5. 如果只是本地使用，导入后即可；若需分发给其他设备，可以打包模型文件。

6. 在本地测试模型以确保其正常运行：
    - 创建 Ollama 账户
    - 推送模型：`ollama push registry.ollama.ai/<your-username>/my-custom-model`


# 问题

- CPU 是满载的，GPU是空闲的。【todo】
- 执行模型也需要 proxy 开启
- 模型回答有些混乱，我问第二个问题时，它还在想第一个问题。
- jetson-containers  【todo】


## Jetson 共享 windows 代理

1. 查看 Windows 局域网IP，`<windows-ip>`
2. Jetson 接入相同的局域网
3. Jetson 设置变量: `export http_proxy="<windows-ip>:7890"`
4. Clash 开启"允许局域网连接"
5. Clash 开启 TUN模式
6. 配置 docker 文件：`sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf`，添加上述 http_proxy。
7. 重启 Daemon & docker

