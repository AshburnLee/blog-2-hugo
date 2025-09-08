+++
date = '2025-08-31T12:57:46+08:00'
draft = false
title = 'Triton Workflow'
tags = ["triton"]
categories = ["triton"]
+++


~~~s
TorchDynamo #FX Graph。是一个即时编译器，将一系列 PyTorch 操作转换成一个 FX 图 。
# 这个 FX 图是一个中间表示，可以被不同的后端编译器优化和执行。 TorchDynamo 本身并
# 不直接进行优化，而是将优化工作交给其他的后端，例如 Inductor。
    |
    |
    v
Inductor (这是一个编译器后端 , 它接收 TorchDynamo 生成的 FX 图，并将它们编译成优化的 C++/Triton 内核) 
#  Inductor 将 TorchDynamo 提供的 FX 图转换成其自身的 IR，这个 IR 考虑了循环融合、内存访问优化等因素。 
# 然后，Inductor 会根据目标硬件 (GPU 或 CPU) 和其他配置，将这个自身 IR 转换成优化的代码。 
# 对于 GPU，它会使用 Triton 作为代码生成的后端。
(Triton Kernels)  # @triton.jit, python function。 用户使用 Python 和 Triton API 
# 定义 Triton kernel 的计算逻辑，Triton 编译器将这个 Python 定义的 kernel 转换成实际的 CUDA 或 ROCm 代码，
# 作为最终的 GPU kernel。
    |
    |
    v
## Triton 里边
# FrontEnd 是 python 表达的计算-> Triton IR
TritonDialect, Triton IR [upstream]
# Pass
- Comnbine pass
- Braodcast reordering
- Tensor pointer rewriting
- other optimizers ...
    ||
    ||
    vv
# Middle End
TritonGPU Dialect, TrtionGPU IR [upstream]
# Pass
- Coalescing
- Layout conversion removal
- Thread Locality optimization
- ...
    ||
    ||
    vv
# TritonGPU Dialect 中最重要的是通过添加Layout 来改变一个tensor的表示形式，表达一个data在GPU的thread是
# 如何partition的
# .mlir 文件中的变量会通过 Blocked 和 Shared 这两中Layout 类型描述，blocked 表示数据切割方式是有 blocked 定义，
# 如: 
# #blocked0 = #triton_gpu.blocked<{versionMajor = 3, 
#                                  versionMinor = 0, 
#                                  warpsPerCTA = [8, 1], 
#                                  CTAsPerCGA = [1, 1], 
#                                  CTASplitNum = [1, 1], 
#                                  CTAOrder = [1, 0], 
#                                  instrShape = [16, 256, 32]}>

# 它定义了数据的切割方式
# %cts : tensor<256xi1, #blocked0>, 逗号前是 mlir 定义的，后边是 TritonGPUDialect 自己定义的
[Shared Layout, Distributed layout (Block layout), Do operand laytout, MMA layout]
    ||
    ||
    vv
## BackEnd
# Intel (&NV, AMD) 各家vendor自己的Dialect
Intel specified Dialect: TritonGEN, TritonIntelGPU 和 许多优化Pass。
At the same time re-use most of the Triton upstream infrastructure and optimizations
    |||
    |||
    vvv
LLVM Dialect, LLVM IR
    |||
    |||
    vvv
Intel: GenISA/GenX  # IGC 编译器得到 SPIRV 中间表示, Intel 没有与 nvcc 对应的工具来生成asambly，
# 所以只能生成 SPIRV，然后使用官方的工具将 SPIRV 翻译为 LLVM
Nvidia: cubin #  nvcc 得到 PTX 表示（nvcc 会生成 PTX（一种IR，与硬件无关）代码，并将其传递给 ptxas，
# 最终得到cubin，它是针对特定 GPU 架构编译的二进制可执行文件）
# IGC has 2 path to compile Triton kernels: SIMT & SIMD (Triton 走 SIMT，没有应用IMEX)
# SIMD： lowerTritonGPU IR to lowe level IR and maps to XeGPU Dialect (来自IMEX)
    |||
    |||
    vvv
Runtime # IPEX, 目前是 Stock Pytorch
~~~


# Tools SPIRV-LLVM-Translator
## In-tree mode 在 mlir project 中使用

cmake 中通过使用 FetchContent(cmake, in-tree mode) 作为依赖构建它的编译， 详见 open source code。

