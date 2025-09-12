+++
date = '2025-01-31T12:45:53+08:00'
draft = true
title = 'TensorRT'
tags = ["TensorRT"]
categories = ["CUDA"]
+++

【todo】缺少实质的认识，都是些表面的文字。


TensorRT 是一个用于高性能深度学习推理的 SDK。它包括一个深度学习**推理优化器**和**运行时**，可为深度学习推理应用提供低延迟和高吞吐量。它用于优化在 NVIDIA GPU 上运行的深度学习模型的推理性能。

## 优势

提高推理吞吐，减少推理延迟，简化部署

## 编程模型

- 导入模型: 将训练好的模型从 TensorFlow、PyTorch 或 ONNX 等框架导入到 TensorRT。
- 优化模型: 使用 TensorRT 优化器对模型进行优化，例如层融合、精度校准和内核自动调整。
- 构建推理引擎: 创建一个 TensorRT 推理引擎，该引擎针对目标 GPU 平台进行了优化。
- 执行推理: 使用推理引擎对输入数据执行推理并获取输出。

## 说明书

https://docs.nvidia.com/deeplearning/triton-inference-server/archives/Tensorrt_inference_server_180/Tensorrt-inference-server-guide/docs/index.html


## 其中常用的库和 tools

1. `libnvinfer`: 这是 TensorRT 的主要库，包含了用于构建推理引擎、执行推理以及进行其他核心操作的 API。
2. `libnvonnxparser`: 用于解析**ONNX 模型并将其转换导入到 TensorRT 内部，以构建可以在 NVIDIA GPU 上高效运行的推理网络**。这使得 TensorRT 可以支持各种深度学习框架，因为许多框架都可以导出 ONNX 格式的模型。
3. `TensorRT Model Optimizer (trtexec)`: 这是一个命令行工具，可以用来优化模型、生成推理引擎，以及对模型进行基准测试。它可以帮助开发者快速评估不同优化策略的效果


## use TensorRT with PyTorch 的一般步骤

1. 训练并导出 PyTorch 模型： 以 TensorRT 可用的格式训练并导出 PyTorch 模型。您可以通过使用 PyTorch 模型的 `torch.onnx.export()` 方法将 PyTorch 模型转换为 ONNX 格式来实现。

2. 优化 ONNX 模型以适配 TensorRT：一旦你有了 ONNX 模型，你可以使用 TensorRT 的 `trtexec` 工具来优化模型以适配 TensorRT。此工具将 ONNX 模型作为输入，并生成一个 **TensorRT 引擎文件**，该文件可以被加载和使用进行推理。 `trtexec` 工具还允许你指定各种优化参数，如精度模式、批大小和输入/输出形状。

3. 加载优化后的 TensorRT 引擎到 Python：一旦你有了优化后的 TensorRT 引擎文件，你可以使用 `Tensorrt.Tensorrt.Builder` 和 `Tensorrt.Tensorrt.ICudaEngine` 类在 Python 中加载它。 `Builder` 类用于从优化的 ONNX 模型创建 TensorRT 引擎，而 `ICudaEngine` 类用于管理和在引擎上执行推理。

4. 在 TensorRT 引擎上运行推理：最后，您可以使用 `ICudaEngine` 对象在 TensorRT 引擎上运行推理。为此，您需要为输入和输出张量分配内存，将输入数据复制到 GPU 内存，使用 `ICudaEngine` 执行推理，然后将输出数据复制回 CPU 内存。


## 对比 torch 模型推理性能 fps，Tensorrt 有成倍的提升（3x~4x）

注意：

1. 并非所有 PyTorch 操作都由 TensorRT 支持。某些操作可能需要在 TensorRT 中手动实现，或替换为提供类似功能的受支持操作。
2. 内存使用：TensorRT 引擎需要额外的内存来存储中间结果和优化数据。这意味着 TensorRT 引擎的内存需求可能与原始 PyTorch 模型不同，部署模型时应予以考虑。


## 动态 shape

通过使用动态形状，用户可以在输入形状每次变化时避免重新创建 TensorRT 引擎。相反，您可以简单地重用相同的引擎，并根据需要分配输入和输出缓冲区。这可以提高性能并减少内存使用。


## 用户接口

为了获得最佳性能，通常会结合使用 Python 和 C++。对于简单的模型沟通过Python API 即可。但高性能部署的常见做法:

1. 使用 Python API 构建和优化 TensorRT 引擎，并将**引擎序列化到磁盘**。
2. 使用 C++ 加载序列化后的引擎，并使用 C++ API 执行推理。 这是因为 C++ API 可以更好地控制底层硬件和内存管理，从而获得最佳性能。

这种混合使用 Python 和 C++ 的方法结合了 Python 的易用性和 C++ 的高性能，是 TensorRT 在实战中的一种最佳实践。


## ONNX 格式

ONNX（Open Neural Network Exchange）是一种开放的深度学习模型交换格式。它允许开发者在不同的深度学习框架之间转换模型，从而提高互操作性。

**框架互操作性**: ONNX 的主要目标是促进不同深度学习框架之间的互操作性。通过将模型转换为 ONNX 格式，开发者可以在一个框架中训练模型，然后将其部署到另一个框架中，而无需重写模型代码。但是ONNX 本身并不是一个可执行的模型格式。 目标框架需要一个 ONNX 运行时或转换器来加载 ONNX 模型并将其转换为其内部表示，以便进行推理或进一步的训练。

简化模型部署流程: 使用 ONNX 可以简化模型部署流程，因为它提供了一个标准化的格式，可以被不同的推理引擎识别和使用。


## TensorRT 中的优化方法 与AI编译器中的优化方法的区别和联系

TensorRT 主要针对 NVidia GPU 平台的，可移植性相对 AI 编译器差


# 实践

Jetson Orin 上已经安装了 TensorRT

## Torch-TensorRT （https://pytorch.org/TensorRT/）

为 Pytorch 提供的一个推理编译器，通过利用 TensorRT 这个优化器和运行时，给 NVidia GPU 提供高性能的推理。


