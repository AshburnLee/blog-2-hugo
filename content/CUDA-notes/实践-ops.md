+++
date = '2025-01-31T12:45:28+08:00'
draft = true
title = '实践 Ops'
tags = ["CUDA"]
categories = ["CUDA"]
+++



## (log-)Softmax

softmax 的计算公式是：

`softmax(x)_i = exp(x_i) / sum(exp(x_j))`  计算reduce sum

实战中，需要考虑到数值稳定：

**分子和分母 e的指数 都减去 `max_x`，公式不变**： 计算reduce max

`softmax(x)_i = exp(x_i - max_x) / sum(exp(x - max_x))`  

这个变换不会改变 softmax 的结果，但它显著地降低了指数运算的结果，避免了溢出。 因为 `x_i - max_x` 总是小于等于 0，`exp(x_i - max_x)` 的值就不会过大。

元素个数是 N，则一共有 4N 次计算，这用于计算 Gflops。

~~~py
def softmax(x):
    x = np.array(x) # Ensure x is a NumPy array
    max_x = np.max(x, axis=-1, keepdims=True) # Find the maximum value along the last axis
    exp_x = np.exp(x - max_x) # Subtract max_x for numerical stability
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True) # Sum along the last axis
    softmax_x = exp_x / sum_exp_x # Compute softmax
    return softmax_x

def log_softmax_numpy(x):
    """Compute log softmax using NumPy with numerical stability."""
    x = np.array(x)
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    log_softmax_x = x - max_x - np.log(sum_exp_x)
    return log_softmax_x
~~~

在CUDA实现中，pytorch是这样做的：

1. 计算执行kernel前的线程配置，得到<<< >>>，函数名 `SpatialSoftMax_getLaunchSizes`，具体地，包括

  - 1. 计算block 大小：`SpatialSoftMax_getBlockSize`
    ~~~cpp
    inline dim3 SpatialSoftMax_getBlockSize(
    uint64_t dim_size, uint64_t inner_size) {
      uint32_t inner_threads = inner_size;
      inner_threads = std::min(inner_threads, static_cast<uint32_t>(max_threads));
      uint32_t dim_threads = 1;
      if (inner_threads <= 64 && dim_size >= 64) {
        while (inner_threads * dim_threads <= max_threads && dim_threads <= dim_size)
          dim_threads *= 2;
        dim_threads /= 2;
      }
      return dim3(dim_threads, inner_threads);
    }
    ~~~

  - 2. 计算最大activate block：`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, 这个是cuda 运行库中的一个函数

  - 3. 计算grid 大小：`SpatialSoftMax_getGridSize`

    ~~~cpp
    ////////////////////////////////////////////////////////////////////////////////
    // Spatial kernel (fast with large inner_size and small dim_size)
    ////////////////////////////////////////////////////////////////////////////////
    // Let's assume that our input has been flattened to have only three dimension:
    //     outer x dim x inner
    // The spatial algorithm tries to parallelize along all of them.
    // Within a 2d block threadIdx.y parallelizes over dim slices, and threads that
    // share it will speed up reductions over dim (along axis x).
    // The 2d grid is used to parallelize inner dimension over y axis and outer over x.
    inline dim3 SpatialSoftMax_getGridSize(dim3 block, uint32_t max_active_blocks,
        uint64_t outer_size, uint64_t inner_size) {
      // First, tile as many blocks as we can over the y axis
      uint32_t inner_blocks = (inner_size + block.y - 1) / block.y;
      if (inner_blocks > max_active_blocks) inner_blocks = max_active_blocks;
      // Fill the x axis with as many blocks as we can fit (a little more is ok too)
      uint32_t outer_blocks = (max_active_blocks + inner_blocks - 1) / inner_blocks;
      if (outer_blocks > outer_size)
        outer_blocks = outer_size;
      return dim3(outer_blocks, inner_blocks);
    }
    ~~~

2. 执行kernel函数

  根据工程中的计算逻辑，首先计算了 input_max，然后计算 sum = sum(exp(x - input_max)), 最后计算 一个线性组合，通过：`Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_input, sum);`，这个是 cutlass 中的 API，cutlass 是一个高度模板化的库，实际使用时需要为 `LinearCombinationGeneric` 提供更多的模板参数，以指定各种配置选项，如累加器和偏置的存储方式、是否使用共享内存等。给出一个简单实例：

  ~~~cpp
  #include <cutlass/epilogue/thread/linear_combination.h>
  #include <cutlass/numeric_types.h>

  // 这是我的实例化的 Epilogue 类
  // 假设你的数据类型是 float，向量长度为 1，不使用激活函数
  using Epilogue = cutlass::epilogue::thread::LinearCombinationGeneric<
      float,                    // 数据类型
      1,                        // 向量长度
      float,                    // 偏置数据类型（如果与主数据类型相同，则写为相同类型）
      cutlass::epilogue::thread::NoActivation // 激活函数类型，这里不使用激活函数
  >;

  __global__ void myKernel(...) {
      // ... 初始化和其他 CUDA 逻辑 ...

      // 假设 acc 是累加器的值，bias 是偏置值
      float acc = ...;
      float bias = ...;

      // 实例化我的类 epilogue 对象
      Epilogue epilogue;

      // 执行 epilogue 操作，结果存储在 output 中
      float output = epilogue(acc, bias);

      // ... 后续操作 ...
  }
  ~~~

  总之，上述的执行了线性组合，所以说 pytorch 的 cuda softmax 计算过程与上述 python 直接实现有些不同。如何不同，最后是一个线性变化【这个线性变换是啥，所以 pytorch 的实现中的计算逻辑是啥？？】


Paddle 中我是这样做的：见同名 .cu 文件，更直接 方便的是，将 Paddle checkout 到我的某一个commit，在repo中读代码 【2c66775be371d53ad0e089fcf3c11e997b774d13】这个commit是最后一个 grid sampler 的优化，应该包含了所有的我优化的op【check 已找到我的code】

KAQ: logsoftmax 中为什么当 `(inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096)` 时，是那种方式的计算?
答：inner_szie == 1 表示沿着最后一维度做操作，所以说这时，它是非 spacial 的，所以是直接计算。

pytorch 使用了 cutlass 作为计算的最后一步【cutlass的编程模型，nice to have】

## cuda 规约

logsoftmax 实现中给出了 Block 延某个方向的 reduce 和 Warp 中的 reduce。

## 双线性差值

什么是双线性差值？

假设我们有一个二维函数 z=f(x,y) 并且已知四个点的值：
f(0,0)=1，f(0,1)=3，f(1,0)=2，f(1,1)=4。我们要对点 (x=0.5,y=0.5) 即 f(0.5, 0.5) 进行双线性插值。

计算权重：

w00=(x1−x)(y1−y)=(1−0.5)(1−0.5)=0.5×0.5=0.25
w10=(x−x0)(y1−y)=(0.5−0)(1−0.5)=0.5×0.5=0.25
w01=(x1−x)(y−y0)=(1−0.5)(0.5−0)=0.5×0.5=0.25
w11=(x−x0)(y−y0)=(0.5−0)(0.5−0)=0.5×0.5=0.25

计算差值结果：

z=f(x,y)=f(x0,y0)⋅w00+f(x1,y0)⋅w10+f(x0,y1)⋅w01+f(x1,y1)⋅w11 = 1×0.25+2×0.25+3×0.25+4×0.25 =2.5

所以 f(0.5,0.5) = 2.5


## 最临近差值

已知图像中几个像素点的值：f(0,0)=1, f(0,1)=3, f(1,0)=2, f(1,1)=4。我们要对坐标为 f(0.6,0.4) 进行最近邻插值。

计算步骤：

1. 确定最近邻点：在 x 方向上，0.6 更接近 1。在 y方向上，0.4 更接近 0 。所以 f(0.6, 0.4) 最近邻点是 f(1,0)
2. 因为最近邻点为 f(1,0) 且 f(1,0)=2，所以对于坐标为 f(0.6,0.4) 的最近邻插值结果为 2

