+++
date = '2025-01-31T12:45:51+08:00'
draft = false
title = 'Kernel 向量化 SIMD'
tags = ["CUDA","SIMD","Vectorization"]
categories = ["CUDA"]
+++


向量化是利用 SIMD 指令的**软件优化技术**，而 SIMD 是提供并行计算能力的硬件条件。

## 什么是 SIMD 指令集

SIMD 指令集是一种在单个时钟周期内对多个数据执行相同操作的指令集。通过使用 SIMD 指令，CPU 可以并行处理多个数据，从而提高程序的性能。

SIMD 指令的优势

- 提高性能：通过并行处理多个数据，SIMD 指令可以显著提高程序的性能。
- 降低功耗：通过减少指令数量，SIMD 指令可以降低 CPU 的功耗。

例：比如 VNNI 指令是 SIMD 的，通过并行处理多个低精度数据（如 INT8）来提高计算效率。VNNI 指令集中的一个额具体指令 `VPDPBUSD`（AVX-512 VNNI 指令）设计为每次计算 4 个 INT8 元素的点积，并将结果累加到 INT32 输出。对于 AVX-512 寄存器，存储 64 个 INT8 元素, 4 个 INT8 为一个向量，两个 input 点积得到 1 个输出，所以 `VPDPBUSD` 得到 16 （64/4）个元素存在 INT32 输出。这个过程中，向量中的 4 个 int8 是同时计算的。体现了SIMD。

~~~cpp
#include <immintrin.h>
#include <cstdint>
#include <iostream>

void dot_product_vnni() {
    // 输入向量：每个包含 64 个 INT8 元素（512 位寄存器）
    int8_t vec1[64] = {
        1, 2, 3, 4,  // 子向量 1
        5, 6, 7, 8,  // 子向量 2
        // 示例
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    int8_t vec2[64] = {
        2, 3, 4, 5,  // 子向量 1
        1, 2, 1, 2,  // 子向量 2
        // 示例
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    int32_t result[16] = {0};  // 存储 16 个点积结果 (INT32)

    // 加载到 AVX-512 寄存器 ‘u’ 表示
    __m512i v1 = _mm512_loadu_si512(vec1);
    __m512i v2 = _mm512_loadu_si512(vec2);
    __m512i res = _mm512_setzero_si512();  // 初始化结果为 0

    // 使用 VPDPBUSD 计算点积
    res = _mm512_dpbusd_epi32(res, v1, v2);

    // 存储结果
    _mm512_storeu_si512(result, res);

    // 输出前两个子向量的点积
    std::cout << "Dot product 1: " << result[0] << std::endl;  // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    std::cout << "Dot product 2: " << result[1] << std::endl;  // 5*1 + 6*2 + 7*1 + 8*2 = 5 + 12 + 7 + 16 = 40
}
~~~

## SIMD 指令的工作原理

- 向量寄存器：SIMD 指令使用**向量寄存器**来存储多个数据。例如，一个 128 位的向量寄存器可以存储 4 个 32 位的整数或浮点数。

- 并行计算：SIMD 指令可以同时对向量寄存器中的多个数据执行相同的操作。例如，一条 SIMD 加法指令可以将两个向量寄存器中的 4 个整数同时相加。

- 数据对齐：为了保证高效的内存访问，SIMD 指令通常**要求数据是对齐**的。这意味着数据的**地址必须是向量寄存器大小的倍数**。

接上 `VPDPBUSD`，栈上分配的数组vec1 和 vec2，C++ 编译器通常只保证 16 字节对齐（或更低），而 AVX-512 要求 64 字节对齐。并且使用 `_mm512_loadu_si512`（`u` 表示 `unaligned`，未对齐），它允许加载未对齐的数据，但性能较低。


## 检查地址是否 64 字节对齐

方法：

~~~cpp
std::cout << "vec1 alignment: " << (reinterpret_cast<uintptr_t>(vec1) % 64) << std::endl;
~~~

若输出为 0，说明地址是 64 字节的倍数。

## 要保证 64 字节对齐

有多种方法

### 1. 在 heap 上分配内存时

`_mm_malloc(size, 64)` 分配 64 字节对齐的内存，也可使用 也可使用 `std::aligned_alloc(64, 64 * sizeof(int8_t))`（C++17）

~~~cpp
// 动态分配 64 字节对齐内存
    int8_t* vec1 = (int8_t*)_mm_malloc(64 * sizeof(int8_t), 64);
    int8_t* vec2 = (int8_t*)_mm_malloc(64 * sizeof(int8_t), 64);
    int32_t result[16] = {0};
    // 初始化
    // 对其加载到 AVX-512 寄存器
    __m512i v1 = _mm512_load_si512(vec1);
    __m512i v2 = _mm512_load_si512(vec2);
    __m512i res = _mm512_setzero_si512();
~~~

### 2. 强制数组对齐

`__attribute__((aligned(64)))` 或 `alignas(64)` 强制数组对齐。

~~~cpp
    alignas(64) int8_t vec1[64] = {  // 强制 64 字节对齐
        1, 2, 3, 4, 5, 6, 7, 8,
        // ... 略
    };
    alignas(64) int8_t vec2[64] = {
        2, 3, 4, 5, 1, 2, 1, 2,
        // ... 略
    };
    int32_t result[16] = {0};

    __m512i v1 = _mm512_load_si512(vec1);  // 使用对齐加载
    __m512i v2 = _mm512_load_si512(vec2);
    __m512i res = _mm512_setzero_si512();
~~~


## 利用到 SIMD 指令集，除了对齐，还需要程序满足哪些条件
### 1. 避免分支 (Branch Avoidance)：

分支语句会降低 SIMD 指令的效率。需要尽量避免分支语句，或者使用一些技巧来消除分支。

技术：

- **条件移动指令** (Conditional Move Instructions)：使用条件移动指令来代替分支语句。
- **掩码 (Masking)**：使用掩码来选择需要执行操作的数据。

~~~cpp
// 原始代码
for (int i = 0; i < N; i++) {
    if (a[i] > 0) {
        b[i] = a[i];
    } else {
        b[i] = 0;
    }
}

// 消除分支
for (int i = 0; i < N; i++) {
    b[i] = (a[i] > 0) ? a[i] : 0;
}
~~~

### 2 连续的内存访问模式
### 3 Intrinsics

- 如果编译器无法自动进行 SIMD 优化，可以使用 `Intrinsics` 或汇编语言手动编写 SIMD 代码。
- `Intrinsics` 是编译器提供的一组函数，映射到制定的CPU指令，使得可以直接调用特定的 CPU 指令，而无需编写汇编代码。使用 `intrinsics` 可以让你在 C/C++ 代码中利用 SIMD 指令的优势。
- 比如上述 `_mm512_load_si512` 是一个 AVX-512 Intrinsics 函数。


### 4 循环优化

- 循环展开 (Loop Unrolling)：将循环体展开多次，减少循环的迭代次数，从而提高 SIMD 指令的利用率。
- 循环向量化 (Loop Vectorization)：将循环中的标量操作转换为向量操作，使其能够使用 SIMD 指令。

~~~cpp
// 原始循环
for (int i = 0; i < N; i++) {
    a[i] = b[i] + c[i];
}

// 循环向量化 一次处理4个数，并且使用 Intrinsics
for (int i = 0; i < N; i += 4) {
    __m128 b_vec = _mm_loadu_ps(&b[i]);
    __m128 c_vec = _mm_loadu_ps(&c[i]);
    __m128 a_vec = _mm_add_ps(b_vec, c_vec);
    _mm_storeu_ps(&a[i], a_vec);
}
~~~


### 5 将结构体数组 (Array of Structures, AoS) 转换为数组结构体 (Structure of Arrays, SoA) 

为了充分利用 SIMD 指令的并行性，需要将需要并行处理的数据存储在**连续的内存空间**中。
在 AoS 格式中，结构体的成员变量是交错存储的。这意味着，如果需要对结构体数组中的某个成员变量进行并行处理，需要从不同的内存位置加载数据，这会降低 SIMD 指令的效率。

在 SoA 格式中，相同类型的成员变量存储在**连续的内存空间**中。

具体讲，假设有一个结构体 Point，包含 x 和 y 两个成员变量：

~~~c
struct Point {
    float x;
    float y;
};
~~~

在 AoS 格式中，Point 结构体数组的内存布局如下：

`x1, y1, x2, y2, x3, y3, ...`

如果需要对 x 坐标进行并行处理，需要从不同的内存位置加载数据，这会降低 SIMD 指令的效率。

在 SoA 格式中，Point 结构体数组被转换为两个数组：

~~~c
float x[N];
float y[N];
~~~

x 数组和 y 数组的内存布局如下：

`x1, x2, x3, ..., y1, y2, y3, ...`

现在，如果需要对 x 坐标进行并行处理，可以使用 SIMD 指令一次性加载多个 x 坐标，从而提高 SIMD 指令的效率。


## 常见 SIMD 指令集 【CPU】

- MMX (MultiMedia eXtensions)：Intel 在 1997 年推出的 SIMD 指令集，主要用于多媒体处理。
- SSE (Streaming SIMD Extensions)：Intel 在 1999 年推出的 SIMD 指令集，是**对 MMX 的扩展**，提供了更强大的浮点运算能力。
- SSE2 (Streaming SIMD Extensions 2)：Intel 在 2001 年推出的 SIMD 指令集，增加了对**双精度浮点数**的支持。
- SSE3 (Streaming SIMD Extensions 3)：Intel 在 2004 年推出的 SIMD 指令集，增加了一些新的指令，用于提高多媒体和科学计算的性能。
- SSE4 (Streaming SIMD Extensions 4)：Intel 在 2006 年推出的 SIMD 指令集，包含 SSE4.1 和 SSE4.2 两个版本，增加了一些新的指令，用于提高**字符串处理、文本处理和数据压缩**的性能。
- AVX (Advanced Vector Extensions)：Intel 在 2011 年推出的 SIMD 指令集，将向量寄存器的**宽度从 128 位扩展到 256 位**，从而提高了并行处理能力。
- AVX2 (Advanced Vector Extensions 2)：Intel 在 2013 年推出的 SIMD 指令集，增加了一些新的指令，用于提高**整数运算和位运算**的性能。
- AVX-512 (Advanced Vector Extensions 512)：Intel 在 2016 年推出的 SIMD 指令集，将向量寄存器的**宽度扩展到 512 位**，从而进一步提高了并行处理能力。

