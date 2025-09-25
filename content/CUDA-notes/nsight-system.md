+++
date = '2025-09-19T00:05:50+08:00'
draft = false
title = 'Nsight System'
tags = ["CUDA", "Nsight Systems"]
categories = ["CUDA"]
+++

nsys/nsys-ui 作用是CPU活动和GPU活动的并行可视化，显示两者在时间轴上的具体事件和占用时长。对于kernel 他只能显示到kernel函数这一层，kernel内部是如何的，nsys 无法显示。进入kernel内部，需要使用ncu。在nsys-ui 中，右击 kernel 可以在 ncu 中查看。

## nsys stats

`nsys stats report1.nsys-rep`

~~~txt
 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)   StdDev (ns)            Name
 --------  ---------------  ---------  ------------  ------------  --------  ----------  ------------  ----------------------
     96.5       68,467,392          2  34,233,696.0  34,233,696.0     6,880  68,460,512  48,404,027.4  cudaMalloc
      2.0        1,396,032          2     698,016.0     698,016.0    63,264   1,332,768     897,674.9  cudaMemcpy
      1.2          865,920          1     865,920.0     865,920.0   865,920     865,920           0.0  cudaLaunchKernel
      0.3          198,656          2      99,328.0      99,328.0    16,064     182,592     117,753.1  cudaFree
      0.0            2,336          1       2,336.0       2,336.0     2,336       2,336           0.0  cuModuleGetLoadingMode
~~~

## 生成report

`nsys export --type text --output report1.txt report1.nsys-rep`

输出无法阅读


## 使用 sqlite
`nsys export --type sqlite --output report1.sqlite report1.nsys-rep`
`sqlite3 report1.sqlite`

不直观


## 使用 ui

`nsys-ui report1.nsys-rep` 是最佳方案，需要开启 X11 forwarding。但是 OpenGL 有问题：

OpenGL version is too low (1). Falling back to Mesa software rendering. Mesa failed

通过一下命令可以解决：

~~~sh
export QT_QUICK_BACKEND=opengl
export QSGR_RENDERER_TYPE=opengl
nsys-ui report1.nsys-rep
~~~

可用。

1. UI中，可以在时间轴中定位到 kernel ，右键可以在 ncu 中得到 basic 的报告。

2. 进一步：在gui中 查看 GPU 硬件指标如 SM 利用率、Tensor Core 活动、线程束（Warp）效率、显存带宽和 PCIe/NVLink 数据吞吐

3. `nsys profile --trace=cuda -o report_name ./softmax`  得到 `report_name.nsys-rep`，用 nsys-ui 打开。在 Event Even 标签中查看细节

4. ui 中会有生成指标的命令行。


