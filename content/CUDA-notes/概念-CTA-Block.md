+++
date = '2025-01-31T12:45:56+08:00'
draft = false
title = '概念 CTA Block'
tags = ["CUDA","CTA"]
categories = ["CUDA"]
+++

## CTA （就是Block）

cuda 中的 CTA 是什么概念？什么时候引入的？它的特点是什么？给是应用实例？

CTA (Cooperative Thread Array) 是 CUDA 中的一个重要概念，它实际上就是CUDA编程模型中的线程块(Thread Block)，是逻辑上的概念。

- CTA 概念在CUDA早期版本就已引入，是CUDA编程模型的核心概念之一。
- 它代表了一组可以协同工作的线程，这些线程可以共享资源并进行同步。
- CTA 是 CUDA 程序的任务分发单位，与编程模型中的 block 是同一事物的不同表述
- 一个 CTA 最多可以由16个 warp 组成，即最多包含512个线程

CTA 特点：

- 独立执行：每个CTA可以独立于其他CTA执行，没有固定的执行顺序。
- 资源共享：CTA内的线程可以共享共享内存(Shared Memory)。
- 同步能力：CTA内的线程可以使用同步原语（如__syncthreads()）进行同步。
- 大小限制：一个CTA中的线程数量是有限的，通常不超过1024个。
- 调度单位：GPU硬件调度器以CTA为单位进行调度。
