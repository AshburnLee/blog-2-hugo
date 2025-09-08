+++
date = '2025-08-31T12:47:01+08:00'
draft = false
title = 'Jetson Orin Chip'
tags = ["Jetson"]
categories = ["Jetson"]
+++


NVIDIA Jetson Orin Nano 4GB版本的芯片使用的是NVIDIA Ampere架构，配备512个CUDA核心和16个Tensor Cores。具体的芯片型号是 GA10B

# Jetson Orin Nano 4GB

GPU name: GA10B

Architechture: NVIDIA Ampere

算力：34 TOPS

GPU：NVIDIA Ampere architecture 

SM 个数：4

Tensor cores/SM：4

CUDA core/SM：128

总 Tensor Cores：16 

总 CUDA core：512

GPU 最大频率：1020MHz

CPU：6-core Arm® Cortex®-A78AE v8.2 64-bit CPU 1.5MB L2 + 4MB L3

CPU 最大频率：1.7 GHz

内存：4GB 64-bit LPDDR5 51 GB/s

功耗：7W - 10W - 25W

吞吐：512 * 611.35 * 1000000 * 10^9 * 2(flops) = 34 TOPS


# 芯片 SM 架构

- L1 instruction cache
- L1 data cache / shared memory
- L0 instruction cache
- Warp scheduler (32 threads /clk) clk表示时钟周期，1~2 纳秒，于每个 warp 包含32个线程，这些线程在同一时钟周期内执行相同的指令。所以是 32.

- Dispatch unit (32 threads /clk)
- Register file (16348个 x 32bit)：片上内存空间。寄存器文件和共享内存通常被认为是片上内存，因为它们与 SM（Streaming Multiprocessor）集成在同一块芯片上。
- LD/ST：单元负责内存访问和数据传输，
- SFU：则专门用于执行复杂的数学函数，如sin，cos，log等。
- INT32/FP32/FP64: CUDA core 被组织为FP32和INT32两类，两类1：1，可以并发。


- `GigaThread` 引擎是 NVIDIA GPU 中的一个硬件调度器，负责管理和调度线程的执行。它在 GPU 上下文切换和数据传输方面起着关键作用。
- `MIG` 是 NVIDIA Ampere 架构引入的一项技术，允许将单个物理 GPU 分割成多个独立的 GPU 实例。每个实例都有其自己的专用资源，例如内存、计算单元和 I/O 带宽。
- `Gigathread engine with MIG Control`：在 MIG 环境中，GigaThread 引擎负责在不同的 GPU 实例之间调度线程和管理资源，从而提高 GPU 利用率和性能。


# 硬件待从Orin上确认

GPU 包含多个 GPC，GPC 之间共享 L2 cache，每个 GPC 包含多个 SM 。每个 GPC 含有多个 TPC ，每个 TPC 包含多个 SM。每个 SM 有自己的 L1 和 shared memory。

# From device query

- CUDA Driver Version / Runtime Version          11.4 / 11.4
- CUDA Capability Major/Minor version number:    **8.7**
- Total amount of global memory:                 3311 MBytes (3471556608 bytes)
- (004) Multiprocessors, (128) CUDA Cores/MP:    512 CUDA Cores
- GPU Max Clock rate:                            624 MHz (0.62 GHz)
- Memory Clock rate:                             624 Mhz
- Memory Bus Width:                              64-bit
- L2 Cache Size:                                 2097152 bytes = **2 MB**
- Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
- Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
- Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
- Total amount of constant memory:               65536 bytes
- Total amount of shared memory per block:       49152 bytes
- Total **shared memory** per multiprocessor:    167936 bytes
- Total number of registers available per block: 65536
- Warp size:                                     32
- Maximum number of threads per multiprocessor:  1536
- Maximum number of threads per block:           1024
- Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
- Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
- Maximum memory pitch:                          2147483647 bytes
- Texture alignment:                             512 bytes



# shared memory bank 宽度

没有找到相关文档，但是 Ampere GPU的共享内存（Shared Memory）架构与其他 NVIDIA GPU类似，共享内存的 bank 宽度通常为32位，每个 bank 在每个时钟周期可以处理32位数据。4bytes


# hardware info From ncu
## fp32 优化得好

该 GPU 的单精度浮点（FP32）峰值性能是双精度浮点（FP64）峰值性能的 64 倍。如果需要 FP64 操作，性能将受限，因为 FP64 硬件单元较少（仅 1/64 的 FP32 吞吐量） 

## SM 个数 = 4 
## 每个SM 可用的shared memory = 16KB

