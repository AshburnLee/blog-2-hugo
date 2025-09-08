+++
date = '2025-08-31T12:45:50+08:00'
draft = false
title = 'From Triton'
tags = ["CUDA","Triton"]
categories = ["CUDA"]
+++


# TritonNvidiaGPU 的 3 个pass

- `triton-nvidia-gpu-plan-cta`
- `triton-nvidia-gpu-fence-insertion`
- `triton-nvidia-tma-lowering`


## triton-nvidia-gpu-plan-cta

这个 pass 为 `DotOp`、`ReudceOp`、`StoreLikeOps` 计算并应用优化过的 CTA。

以 `DotOp` 为例，逻辑是：遍历 funcOp 中所有的的 DotOp，获取类型和操作数，计算 Block 分块大小，应用这个分块，并且更新输入输出的 Layout。源码如下：

~~~cpp
bool CTAPlanner::processDot(triton::FuncOp &funcOp) {
  // TODO: This is a naive implementation and should be refactored
  // 这个lambda函数根据 MNK和CTA个数 来确定分块大小 splitM，splitN
  auto getCTATiling = [](int64_t M, int64_t N, int64_t K,
                         unsigned numCTAs) -> std::pair<unsigned, unsigned> {
    // prefer a larger chunk size, at most 128; first assign splitM.
    unsigned chunk_m = 128;
    auto isLegal = [](unsigned chunk) { return chunk >= 64; };
    unsigned splitM, splitN;
    for (; isLegal(chunk_m); chunk_m /= 2) {
      splitM = std::clamp<unsigned>(M / chunk_m, 1, numCTAs);
      splitN = numCTAs / splitM;
      if (isLegal(N / splitN)) // chunk_n;
        break;
    }
    return {splitM, splitN};
  };

  // 使用Walk 遍历funcOp 中的所有DotOp
  funcOp.walk([&](triton::DotOp dot) {
    MLIRContext *ctx = dot.getContext();

    // 获取类型
    auto aTy = cast<RankedTensorType>(dot.getA().getType());
    auto bTy = cast<RankedTensorType>(dot.getB().getType());
    auto dTy = cast<RankedTensorType>(dot.getD().getType());

    assert(isa<ttg::DotOperandEncodingAttr>(aTy.getEncoding()) &&
           isa<ttg::DotOperandEncodingAttr>(bTy.getEncoding()) &&
           isa<ttg::BlockedEncodingAttr>(dTy.getEncoding()) &&
           "PlanCTAPass should follow immediately after CoalescePass");

    // 获取编码
    auto aLayout = cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
    auto bLayout = cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
    auto dLayout = cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());

    // 获取shape
    unsigned M = dTy.getShape()[0];
    unsigned N = dTy.getShape()[1];
    unsigned K = aTy.getShape()[1];

    unsigned splitM, splitN;
    // 根据lambda函数计算 splitM，splitN
    std::tie(splitM, splitN) = getCTATiling(M, N, K, ttg::getNumCTAs(dLayout));
    // 设置分块
    setTiling({splitM, splitN, 1});

    // 创建新的Layout属性
    auto newCTALayout = ttg::CTALayoutAttr::get(ctx, {splitM, splitN},
                                                {splitM, splitN}, {1, 0});
    auto newDLayout = ttg::BlockedEncodingAttr::get(
        ctx, dTy.getShape(), dLayout.getSizePerThread(), dLayout.getOrder(),
        ttg::getNumWarpsPerCTA(dLayout), 32, newCTALayout);
    auto newALayout = ttg::DotOperandEncodingAttr::get(ctx, aLayout.getOpIdx(),
                                                       newDLayout, 0);
    auto newBLayout = ttg::DotOperandEncodingAttr::get(ctx, bLayout.getOpIdx(),
                                                       newDLayout, 0);

    // 更新操作数和结果的 Layout
    insertCasts(dot.getOperation(), {newALayout, newBLayout, newDLayout},
                {newDLayout});
  });
  return true;
}
~~~

其中 insertCasts 表达如下：

~~~cpp
void CTAPlanner::insertCasts(Operation *op,
                             llvm::ArrayRef<Attribute> newOperandLayouts,
                             llvm::ArrayRef<Attribute> newResultLayouts) {
  assert(op->getNumOperands() == newOperandLayouts.size() &&
         "NumOperands mismatched");
  assert(op->getNumResults() == newResultLayouts.size() &&
         "NumResults mismatched");

  Location loc = op->getLoc();
  OpBuilder builder(op->getContext());

  builder.setInsertionPoint(op);
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    auto operandTy = operand.getType();
    if (triton::isTensorOrTensorPointerType(operandTy)) {
      operandTy = replaceLayout(operandTy, newOperandLayouts[i]);
      auto cast = markBackward(builder.create<CastOp>(loc, operandTy, operand));
      op->setOperand(i, cast.getResult(0));
      queue.push(cast);
    }
  }

  builder.setInsertionPointAfter(op);
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    Value result = op->getResult(i);
    auto resultTy = result.getType();
    if (triton::isTensorOrTensorPointerType(resultTy)) {
      resultTy = replaceLayout(resultTy, newResultLayouts[i]);
      auto cast =
          markForward(builder.create<CastOp>(loc, result.getType(), result));
      result.setType(resultTy);
      result.replaceAllUsesExcept(cast.getResult(0), cast.getOperation());
      queue.push(cast);
    }
  }
}
~~~

## triton-nvidia-gpu-fence-insertion
## triton-nvidia-tma-lowering

