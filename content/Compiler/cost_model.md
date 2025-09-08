+++
date = '2025-08-31T12:15:29+08:00'
draft = false
title = 'Cost Model'
tags = ["Compiler"]
categories = ["Compiler"]
+++


# 计算各个角度的cost

thread utilization 【done】
computation intensity
cache locality 【done】
memory requirements
computation unit efficiency
padding/pack cost 【done】
workload balance 【done】
communication
previous matmul

## vector register

~~~cpp
// calculate the cost of the hardware efficiency(whether the vector register is
// fully utilized)
double vectorRegEfficiencyCost(linalg::LinalgOp &linalgOp,
                               ArrayRef<uint32_t> shape,
                               const MatmulConfig &config,
                               CPUTargetDescriptionAnalysis &sysDesc) {
  size_t dtypeSize = DataLayout().getTypeSizeInBits(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  size_t maxVectorWidth = sysDesc.getMaxVectorWidth() / dtypeSize;
  // TODO: take matrix register like amx into account
  double cost = (maxVectorWidth - config.innerMostMBlock % maxVectorWidth) %
                    maxVectorWidth * 1.0 / config.innerMostMBlock +
                (maxVectorWidth - config.innerMostKBlock % maxVectorWidth) %
                    maxVectorWidth * 1.0 / config.innerMostKBlock +
                (maxVectorWidth - config.innerMostNBlock % maxVectorWidth) %
                    maxVectorWidth * 1.0 / config.innerMostNBlock;
  return cost;
}

~~~

1. 这个计算cost的原理是什么？

计算向量寄存器的利用效率。向量寄存器是CPU中用于存储多个数据元素以供SIMD指令并行处理的寄存器。理想情况下，为了最大化性能，你希望在每个SIMD指令中完全填满向量寄存器，没有浪费的空间。这个函数通过计算在矩阵乘法的**最内层循环**中，向量寄存器未被完全利用的程度来估算代价。

2. 为什么要这样计算cost？

如果向量寄存器没有被完全利用，那么你就没有最大化硬件的计算能力，这会导致性能下降。通过计算这个代价，你可以评估不同的分块大小（由config.innerMostMBlock、config.innerMostKBlock和config.innerMostNBlock指定）对向量寄存器利用效率的影响，通过给出一个cost值 指导选择能够最大化向量寄存器利用的配置。

3. 给出实例？


## padding cost

~~~cpp
double paddingCost(linalg::LinalgOp &linalgOp, ArrayRef<uint32_t> shape,
                   const MatmulConfig &config,
                   CPUTargetDescriptionAnalysis &sysDesc) {
  double cost = 0;
  uint32_t M = shape[0], N = shape[1], K = shape[2];
  bool isPadOnM = M % config.innerMostMBlock != 0,
       isPadOnK = K % config.innerMostKBlock != 0,
       isPadOnN = N % config.innerMostNBlock != 0;
  if (isPadOnM || isPadOnK) {
    cost += llvm::divideCeil(M, config.innerMostMBlock) *
            llvm::divideCeil(K, config.innerMostKBlock);
  }
  if (isPadOnK || isPadOnN) {
    cost += llvm::divideCeil(N, config.innerMostNBlock) *
            llvm::divideCeil(K, config.innerMostKBlock);
  }
  if (isPadOnM || isPadOnN) {
    cost += llvm::divideCeil(N, config.innerMostNBlock) *
            llvm::divideCeil(M, config.innerMostMBlock);
  }
  return cost;
}
~~~

因为填充（padding）而产生的额外计算成本。这里的填充指的是为了满足特定的硬件要求（如内存对齐或特定的块大小），在矩阵的边缘添加额外的零元素以调整矩阵的维度。
这里的cost是根据 需要填充的块的数量来计算的。

比如，形状为shape = [4096, 4096, 4096]，并且
config.innerMostMBlock = 512;
config.innerMostKBlock = 512;
config.innerMostNBlock = 512;
由于 4096 能被 512 整除，所以不需要在任何维度上进行填充，因此填充成本cost将是0。

## thread utilization

~~~cpp
double memoryConsumptionOnThreadCost(linalg::LinalgOp &linalgOp,
                                     ArrayRef<uint32_t> shape,
                                     const MatmulConfig &config,
                                     CPUTargetDescriptionAnalysis &sysDesc) {
  assert(shape.size() >= 3 && "shape.size() should >= 3");
  uint32_t M = shape[0], N = shape[1], K = shape[2];
  size_t dtypeSize = DataLayout().getTypeSize(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  // if use K split, there will be one more final reduce and break the post
  // fusion
  double KSplitPenalty = 8.0 * dtypeSize;
  double memoryConsumptionPerThread =
      M * K * 1.0 / config.MThreads / config.KThreads +
      K * N * 1.0 / config.KThreads / config.NThreads +
      M * N * ((config.KThreads - 1) * KSplitPenalty + 1.0) / config.MThreads /
          config.NThreads;
  return memoryConsumptionPerThread;
}
~~~

1. 这个计算cost的原理是什么？
2. 为什么要这样计算cost？
3. 给出实例？

## cache locality

~~~cpp
// calculate the cost of the computation intensity on the L2 cache
double computationIntensityOnL2Cache(linalg::LinalgOp &linalgOp,
                                     ArrayRef<uint32_t> shape,
                                     const MatmulConfig &config,
                                     CPUTargetDescriptionAnalysis &sysDesc) {
  double fullLoadRatio = 0.7;  // 缓存满载比例（Cache Load Factor）见 doc/hardare-basis.md
  uint32_t L2Cache = sysDesc.getCacheSize(2);  // 1024 * 1024
  size_t dtypeSize = DataLayout().getTypeSize(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  uint32_t outOfCachePenalty = 1024;
  // 理论上浮点计算次数
  double FLOPS = 2.0 * config.MBlock * config.NBlock * config.KBlock;
  // 内存消耗
  double memoryConsumption = config.MBlock * config.NBlock +
                             config.NBlock * config.KBlock +
                             config.MBlock * config.KBlock;
  // 计算强度，意味着每次内存访问可以完成更多的计算，所以这个值越大越好
  double computationIntensity = FLOPS / memoryConsumption;
  if (memoryConsumption * dtypeSize > L2Cache * fullLoadRatio)
    computationIntensity /= outOfCachePenalty;
  return 1 / computationIntensity;
}
~~~

1. 这个计算cost的原理是什么？

计算强度（computationIntensity）是通过将浮点运算的数量（FLOPS）除以内存消耗来计算的。FLOPS是指每次矩阵乘法操作中执行的**浮点运算次数**，而**内存消耗**是指执行这些运算所需读取或写入的数据量。

通过估算计算强度和考虑缓存未命中的影响，可以选择最优的分块策略，以提高矩阵乘法在特定硬件上的性能。

如果内存消耗超过了L2缓存的这个比例（0.7），那么就认为缓存不足以容纳所有数据，导致缓存未命中率上升。理想情况下，你希望尽可能多地执行计算，同时尽可能少地访问内存，特别是避免缓存未命中，这样可以提高计算效率。

2. 为什么要这样计算cost？

量化不同配置下这个 Op 对缓存的利用效率。每次内存访问可以完成更多的计算，则认为这个配置是高效的。

3. 矩阵乘的 浮点运算次数 和 内存消耗 如何计算？
浮点运算次数：乘法次数[M*N*K] + 加法次数[M*N*(K-1)]，一般MNK值很大，所以总的 FLOPS 可以认为是[2*M*N*K]。
内存消耗：写次数[M*N] + 读次数[M*K+N*K]，这里是通过shared memeory 使得A，B矩阵每个元素只读一次。

## workload balance

~~~cpp
// calculate the cost of the workload balance
double workloadBalancedCost(linalg::LinalgOp &linalgOp,
                            ArrayRef<uint32_t> shape,
                            const MatmulConfig &config,
                            CPUTargetDescriptionAnalysis &sysDesc) {
  assert(shape.size() >= 3 && "shape.size() should >= 3");
  // 1. 计算每个维度上的任务数量
  uint32_t M = shape[0], N = shape[1], K = shape[2];
  uint32_t MTaskNum = llvm::divideCeil(M, config.MBlock);
  uint32_t NTaskNum = llvm::divideCeil(N, config.NBlock);
  uint32_t KTaskNum = llvm::divideCeil(K, config.KBlock);
  // 2. 计算每个维度上任务数与线程数之间的不匹配程度。
  // 每个维度上任务数不能整除线程数所产生的余数，除以该维度的任务总数来实现的。这个比值反映了任务分配的不平衡程度
  double cost = (MTaskNum % config.MThreads) * 1.0 / MTaskNum +
                (NTaskNum % config.NThreads) * 1.0 / NTaskNum +
                (KTaskNum % config.KThreads) * 1.0 / KTaskNum;
  // 3. 如果任何维度上的任务数少于分配的线程数，则认为存在线程未充分利用的情况，这会通过乘以一个较大的惩罚因子（threadNotFulllyUtilizedPenalty）来增加成本。
  if (MTaskNum < config.MThreads || NTaskNum < config.NThreads ||
      KTaskNum < config.KThreads) {
    double threadNotFulllyUtilizedPenalty = 10.0;
    cost *= threadNotFulllyUtilizedPenalty;
  }
  return cost;
}
~~~

1. 这个计算cost的原理是什么？

首先计算每个维度上的任务数，然后得到每个维度上任务数不能整除线程数所产生的余数，除以该维度的任务总数，这个比值反映了任务分配的不平衡程度。最后如果某一维度上的任务数小于线程数，则认为没有充分利用线程。此时上述得到的cost 会乘以一个较大的因子。

2. 为什么要这样计算cost？

这种计算方式旨在量化工作负载在并行计算环境中的平衡程度。如果某些线程比其他线程拥有更多的工作量，那么整体性能将受到这些“瓶颈”线程的限制。通过计算成本，可以评估不同配置下工作负载的均衡性，进而选择cost最小的配置。

3. 给出实例？

假设有以下矩阵乘法配置：

矩阵形状：M=100, N=100, K=100
块大小：MBlock=25, NBlock=25, KBlock=25
线程配置：MThreads=4, NThreads=4, KThreads=4
计算过程：

MTaskNum = 100 / 25 = 4
NTaskNum = 100 / 25 = 4
KTaskNum = 100 / 25 = 4
每个维度的任务数都能被线程数整除，因此：

不平衡成本 = (0/4 + 0/4 + 0/4) = 0
无需应用线程未充分利用惩罚
最终成本为0，表示这是一个非常均衡的工作负载分配。

## thread utilization

~~~cpp
double dynamicBufferizationCost(linalg::LinalgOp &linalgOp,
                                ArrayRef<uint32_t> shape,
                                const MatmulConfig &config,
                                CPUTargetDescriptionAnalysis &sysDesc) {
  assert(validateConfig(config) && "config is invalid");
  assert(shape.size() >= 3 && "shape.size() should >= 3");
  uint32_t M = shape[0], N = shape[1];
  double cost = 0;

  // M 维度上每个线程要处理的block数量
  uint32_t MNumBlockPerThread =
      llvm::divideCeil(M / config.innerMostMBlock, config.MThreads);
  // 每个block含有多少个inner block
  uint32_t MNumInnerBlockPerBlock =
      llvm::divideCeil(config.MBlock, config.innerMostMBlock);
  // 如果满足下面的计算，则返回true，表示没有额外的cost
  uint32_t MCost = MNumBlockPerThread % MNumInnerBlockPerBlock != 0 ||
                   (M / config.innerMostNBlock % config.MThreads != 0 &&
                    config.MBlock != config.innerMostMBlock);

  uint32_t NNumBlockPerThread =
      llvm::divideCeil(N / config.innerMostNBlock, config.NThreads);
  uint32_t NNumInnerBlockPerBlock =
      llvm::divideCeil(config.NBlock, config.innerMostNBlock);
  uint32_t NCost = NNumBlockPerThread % NNumInnerBlockPerBlock != 0 ||
                   (N / config.innerMostNBlock % config.NThreads != 0 &&
                    config.NBlock != config.innerMostNBlock);

  cost = MCost + NCost;
  return cost;
}
~~~

1. 这个计算cost的原理是什么？

它计算的是某个线性代数 op 在给定的 shape 和 MatmulConfig 下，是否有额外的 cost。具体讲，它考虑了M和N维度如何分割成更小的块，以及如何进一步分割给不同的线程。

块分割：矩阵M和N被分割成多个块，这些块的大小由配置（config）中的参数（如MBlock, NBlock, innerMostMBlock, innerMostNBlock）决定。这些块的大小和数量影响了数据访问模式和并行化的效率。
线程分配：每个块被进一步分配给多个线程处理，具体数量由MThreads和NThreads决定。线程的数量和块的分配方式影响了计算的负载均衡和线程间的同步成本。
计算成本：判断块和线程分配的均衡性，以及是否有特殊边界需要处理。如果块不能被线程均匀分配，或者块的分割方式与配置中的内层块大小不匹配，那么将有额外的cost，该函数就是返回是否有额外的cost。

2. 为什么要这样计算cost？

原因是为了评估在给定的线程和块配置下，数据块是否可以被均匀地分配给各个线程，以及是否存在因块大小不匹配或分配不均而导致的额外处理成本。如果块不能被均匀分配，或者分配的方式导致了一些线程比其他线程更忙或更闲，那么这可能会降低并行计算的效率，增加同步的复杂性，或导致缓存未命中率上升。

3. 实例

假设我们有以下配置：

M = 128
config.MBlock = 32
config.innerMostMBlock = 16
config.MThreads = 4
MNumBlockPerThread = llvm::divideCeil(128 / 16, 4) = llvm::divideCeil(8, 4) = 2（每个线程处理2个最内层块）
MNumInnerBlockPerBlock = llvm::divideCeil(32, 16) = 2（每个外部块包含2个最内层块）
MCost = 2 % 2 != 0 || (128 / 16 % 4 != 0 && 32 != 16) = false || (8 % 4 != 0 && false) = false（因为所有条件都不满足，所以MCost为false）

但是，如果我们将config.MThreads改为3，那么：

MNumBlockPerThread = llvm::divideCeil(128 / 16, 3) = llvm::divideCeil(8, 3) = 3（每个线程处理3个最内层块，但最后一个线程可能不足3个）
MCost = 3 % 2 != 0 = true（因为每个线程处理的块数不能被每个块内部的最内层块数整除）


## 如何利用上述的所有cost
~~~cpp
// Analyze the workload and system description to generate the default config
// Factor to consider:
/// thread utilization 【done】
/// computation intensity
/// cache locality 【done】
/// memory requirements
/// computation unit efficiency
/// padding/pack cost 【done】
/// workload balance 【done】
/// communication
/// previous matmul
MatmulConfig MatmulConfigAnalysis::getConfig() {
  if (!hasConfig) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(root)) {  // 动态类型转换尝试获取当前的线性代数操作
      CPUTargetDescriptionAnalysis sysDesc(root);            // 获取当前系统的描述，这可能包括CPU的架构、核心数、缓存大小等信息

      // 提取矩阵维度并操作
      SmallVector<SmallVector<DimType>> oprandDimType =
          *getOprandDimType(linalgOp);
      // get the origin M,N,K size
      SmallVector<unsigned> MDimTypeIdx =
          extractDimTypeIdx(oprandDimType[0], DimType::M);
      SmallVector<unsigned> KDimTypeIdx =
          extractDimTypeIdx(oprandDimType[1], DimType::K);
      SmallVector<unsigned> NDimTypeIdx =
          extractDimTypeIdx(oprandDimType[1], DimType::N);
      uint32_t M = 1U, N = 1U, K = 1U;
      for (auto &&[s, dimType] :
           llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(0)),
                     oprandDimType[0]))
        if (dimType == DimType::M)
          M *= s;
      for (auto &&[s, dimType] :
           llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(1)),
                     oprandDimType[1])) {
        if (dimType == DimType::N)
          N *= s;
        else if (dimType == DimType::K)
          K *= s;
      }

      // innermost Block, if the layout is blocked layout, the innermost block
      // will derived from the layout directly
      uint32_t defaultBlock = 32;
      config.innerMostMBlock = M % defaultBlock == 0 ? defaultBlock : M;
      config.innerMostNBlock = N % defaultBlock == 0 ? defaultBlock : N;
      config.innerMostKBlock = K % defaultBlock == 0 ? defaultBlock : K;
      SmallVector<uint32_t> givenInnermostBlock;
      if (MDimTypeIdx.size() > 1) {
        config.innerMostMBlock = 1;
        for (auto &&[i, d] : llvm::enumerate(MDimTypeIdx))
          if (i != 0)
            config.innerMostMBlock *=
                linalgOp.getShape(linalgOp.getDpsInputOperand(0))[d];
        givenInnermostBlock.push_back(config.innerMostMBlock);
      } else {
        givenInnermostBlock.push_back(0);
      }
      if (NDimTypeIdx.size() > 1) {
        config.innerMostNBlock = 1;
        for (auto &&[i, d] : llvm::enumerate(NDimTypeIdx))
          if (i != 0)
            config.innerMostNBlock *=
                linalgOp.getShape(linalgOp.getDpsInputOperand(1))[d];
        givenInnermostBlock.push_back(config.innerMostNBlock);
      } else {
        givenInnermostBlock.push_back(0);
      }
      if (KDimTypeIdx.size() > 1) {
        config.innerMostKBlock = 1;
        for (auto &&[i, d] : llvm::enumerate(KDimTypeIdx))
          if (i != 0)
            config.innerMostKBlock *=
                linalgOp.getShape(linalgOp.getDpsInputOperand(1))[d];
        givenInnermostBlock.push_back(config.innerMostKBlock);
      } else {
        givenInnermostBlock.push_back(0);
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "M: " << M << ", N: " << N << ", K: " << K << "\n");

      // try to read the config from the attributes
      SmallVector<NamedAttribute> attrs(linalgOp->getAttrs());
      bool hasPredefinedConfig = readConfigFromAttrs(config, attrs);

      // if there is a given config, skip the cost model
      if (!hasPredefinedConfig) {
        LLVM_DEBUG(llvm::dbgs() << "No predefined config\n");
        // TODO: Could add a weight or priority for cost model
        SmallVector<std::tuple<CostModelFn, std::string, double>>
            costModelList = {
                // threshold 0 mean using static shape if possible
                {dynamicBufferizationCost, "dynamicBufferizationCost", 0},
                {workloadBalancedCost, "workloadBalancedCost", 1},
                {vectorRegEfficiencyCost, "vectorRegEfficiencyCost ", -1},
                {computationIntensityOnL2Cache, "computationIntensityOnL2Cache", -1},
                {memoryConsumptionOnThreadCost, "memoryConsumptionOnThreadCost", -1},
                {paddingCost, "paddingCost", -1}};
        SmallVector<uint32_t> shape = {M, N, K};
        // 某种规则下穷举出来的所有matmul config 集合作为candidate
        std::vector<MatmulConfig> configCandidates = prepareConfigCandidates(
            root, sysDesc, shape, givenInnermostBlock, allowIndivisibleInnerBlock);
        for (auto &&[fn, name, threshold] : costModelList) {
          LLVM_DEBUG(llvm::dbgs() << name << "\n");
          // 这里对configCandidates进行这个fn的cost计算，根据指定的 preserveRatio=0.5 得到对于这个fn的cost最小的前50% 的config，作为下一个fn的configCandidates
          // 相当于我有6个fn，configCandidates会从最初的scope缩小为原来的 0.5**5
          configCandidates = filterConfigByCostModel(
              configCandidates, linalgOp, shape, sysDesc, fn, 0.5, threshold);
        }
        // 最终只取cost最小的config
        if (!configCandidates.empty())
          config = configCandidates[0];
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "Final config\nNumThreads: " << sysDesc.getNumThreads()
                 << ", MatmulConfig: " << config << "\n");
    }
    hasConfig = true;
  }

  assert(validateConfig(config) && "config is invalid");
  return config;
}
~~~

上述getConfig() 在matmul相关的变换的 匹配重写（matchAndRewrite）过程中调用，根据task中的输入输出矩阵的shape和硬件信息得到最少的cost配置。见 Pass deep-tile-contraction-op 相关的.mlir 文件，其实shape是给出了的。确实，AI compiler编译的是一个计算，这个计算的前端表达中可定是给出shape信息的。所以上述的cost计算过程发生在compile time，就是应用这个pass的过程中。

关于Cost计算，并不是真正的就这个config执行matmul，而是根据【Factor to consider】中提到的各个方面计算对期望的偏离，偏离越多，cost越大，所以你需要知道对于各个Factor，什么样的情况是最优的(期望)。

## KAQ

这个计算cost 的过程发生在什么阶段？compile OR runtime？计算cost需要知道shape，但是compile时应该不知道shape，所以是在runtime？
