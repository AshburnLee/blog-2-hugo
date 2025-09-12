+++
date = '2025-01-31T12:45:56+08:00'
draft = false
title = '概念 Event'
tags = ["CUDA","Event"]
categories = ["CUDA"]
+++


## CUDA event & stream

默认使用的 stream 0。CUDA 流简单来说是一系列**按顺序**在设备上执行的运算。**不同 Stream 中的运算可以交错进行**，在某些情况下还可以重叠——这一特性可以用来隐藏主机和设备之间的数据传输。

这里是使用 event 的最佳实践：

~~~cpp
  // 创建 CUDA 事件
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 记录开始事件
  cudaEventRecord(start, 0); // 0 代表默认流

  // 启动内核
  myKernel<<<(size + 255) / 256, 256>>>(d_data, size);

  // 记录事件 stop。 这本身是一个异步操作；它不会阻塞 CPU。 
  // 记录事件只是在流中插入一个标记。在事件被记录后，GPU 仍然可能继续执行其他任务。
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); // 阻塞 CPU 直到事件 stop 完成

  // 计算经过的时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // 将数据从设备复制回主机
  cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

  // 打印结果
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  // 销毁事件
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
~~~
