+++
date = '2025-08-31T12:47:02+08:00'
draft = true
title = 'Jetson Ops'
tags = ["Jetson"]
categories = ["Jetson"]
+++



NVIDIA Orin 是一款专为**自动驾驶汽车**和**机器人**设计的高性能系统级芯片，包含新一代 Ampere GPU 架构和 Arm Hercules CPU 内核，以及深度学习加速器和计算机视觉加速器。

应用领域：Orin 芯片不仅适用于自动驾驶汽车，还广泛应用于机器人、工业边缘计算等领域。

支持C/C++, python, cuda, pytorch, ROS (Robot Operating System), JetPack SDK, DeepStream, VScode 

TensorRT 是 NVIDIA 开发的一个高性能深度学习推理 SDK。它不是完全开源的.

`Linux for Tegra (L4T) `是 NVIDIA 为其 Tegra 系列系统芯片 (SoC) 开发的嵌入式 Linux 发行版。它主要用于 NVIDIA Jetson 系列开发套件等嵌入式系统。L4T 提供了运行在 Tegra SoC 上的内核、驱动程序、库和工具，支持各种应用，包括机器人、人工智能、自动驾驶和媒体处理等。 它包含了 NVIDIA 专有的驱动程序，以充分利用 Tegra SoC 的硬件加速功能。不同的 L4T 版本支持不同的 Tegra 系列芯片和功能。 例如，较新的版本可能支持 Vulkan 和更新的 CUDA 版本。 开发者可以使用 L4T 来构建和部署各种嵌入式应用。

# Profiling
## nsight compute（windows）remote profling Orin

不行：Windows 上的 Nsight Compute 无法直接剖析 Jetson Orin 上的可执行文件。 因为后者是Linux，你需要一个linux系统才能与Orin通信

## Jetson Orin 命令行 Nsight System 工具

Nsight Systems 的命令行工具是 nsys. 基本的命令格式如下：`sudo nsys profile --trace=cuda  ./vectorAdd` 需要sudo权限！

输出的文件名 Default is `report#.nsys-rep`. 生成文件在 Ubuntu 上通过 nsys-ui 打开 或拷贝到 windows，就可以 profiling了。！***


## Jetson Orin 命令行 Nsight Compute 工具

更新jetpack 6.2之后，nsight compute 已经安装。***
或者：ncu 命令行工具。如何使用？



# 算子库

通过CMakeLists.txt 的构建方式，写一个 CUDA 算字库，包含 softmax 算子，interplate算子等。通过 cmake 构建，并执行每个算子的测试用例。给出结果。请写出一个框架，以便我扩展更多的算子。

~~~sh
cuda-operators/
├── src/
│   ├── softmax.cu
│   ├── interpolate.cu
│   └── operators.h  // 算子接口头文件
├── test/
│   ├── softmax_test.cu
│   └── interpolate_test.cu
├── CMakeLists.txt
└── main.cu // 主程序，用于测试
~~~

## 如何使用cmake在Orin中？找已有的解决方法

# 哪里有 nvidia Jetson Orin 上的CUDA 编程实践？

- github 开源库中查找 Jetson 关键字的宏定义。
- 通过 Nsight compute，Profile 一个程序，来间接了解 Orin 芯片的特点

# profile 工具

## 安装 Nsight Compute： [done]

通过重新烧录Jetpack，ncu 成功安装。

## Nsight System：已安装，使用: 

1. `nsys profile -o vectorAdd ./vectorAdd` 生成文件 `vectorAdd2.nsys-rep` 然后可以通过 `nsys-ui vectorAdd.nsys-rep` 图形化分析结果。

2. `nsys-ui` 开启UI，找到可执行文件，Start 开始分析，结束后会生成一个 `.nsys-rep` 文件。此方法显示Qt版本问题。不成功。



# 大模型中出现的op

## DeepSeek-R1 

开源的推理模型，通过大规模的强化学习（RL）训练，并结合监督微调（SFT）来提高可读性和连贯性。包括结构和算子：

MLA，旋转位置编码（RoPE），MatMul，混合专家（MoE），Top-K 专家加权，SwiGLU，多令牌预测（MTP），强化学习（RL）与监督微调（SFT），

## Google Gemma3

google开发的 Vision/Language Models。

Matmul，layernorm, RMSNorm, SWiGLU, SiLU, FullyConnected，卷积池化

## QwQ-32B

阿里云，应用RL，集成Agent 能力部署成本低，可部署在消费级硬件上。包含算子：

SwiGLU，RMSNorm， RoPE， Softmax， 

