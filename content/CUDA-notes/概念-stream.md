+++
date = '2025-01-31T12:45:57+08:00'
draft = false
title = '概念 Stream'
tags = ["CUDA","Stream"]
categories = ["CUDA"]
+++


## 多 stream 用于 overlap datatransfer

并发，隐藏延时要实现数据传输与其他操作的重叠，需要使用 CUDA 流.

CUDA 中的流是一系列按主机代码发出的顺序在设备上执行的运算。虽然流内的运算保证按预定顺序执行，但不同流中的运算可以交错执行，并且在可能的情况下，它们甚至可以并行运行。***

所有 CUDA 中在 device 中的操作（内核和数据传输）都在流中运行。当未指定流时，使用默认流（也称为“空流”）。默认流与其他流不同，因为它是一个与设备操作同步的流。

~~~cpp
  float *a, *d_a;
  cudaMallocHost((void**)&a, bytes);   // 弃用的   // host pinned 更推荐使用 cudaHostAlloc
  cudaMalloc((void**)&d_a, bytes);    // device

  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventCreate(&dummyEvent);
  for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);
~~~

在默认流中：

~~~cpp
  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  checkCuda( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );
  kernel<<<n/blockSize, blockSize>>>(d_a, 0);
  checkCuda( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));
~~~

版本1：asynchronous version 1: loop over {copy, kernel-exe, copy-back}

~~~cpp
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice, 
                               stream[i]) );
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));
~~~

版本2，loop over copy, loop over kernel-exe, loop over copy-back。

将3个操作分别分配在不同的 Stream 中，实现并行执行。***

~~~cpp
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
  }
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));
~~~

~~~cpp
cudaFreeHost(a);  // 释放pinned memory
~~~

申请 host pinned 更推荐使用 cudaHostAlloc，它更灵活
