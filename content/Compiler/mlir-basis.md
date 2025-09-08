+++
date = '2025-08-31T12:15:30+08:00'
draft = false
title = 'Mlir Basis'
tags = ["Compiler"]
categories = ["Compiler"]
+++



:sweat_drops: :sweat_drops: :sweat_drops: :sweat_drops: :sweat_drops:

# Learn & reflect

部分来自：https://github.com/KEKE046/mlir-tutorial

## 1. 多层Dialect 理解到了什么？

MLIR 编译从高层 的IR到底层的IR，每个阶段都是多个Dialect的混合。
每次Lowering 都往往针对一个dialect 进行。

Dialect是独立的。例如，在做循环展开等优化的时候，我不需要关心加法和减法可以合并；而在做算数表达式优化的时候，也不需要关心当前在哪个函数里边。

MLIR 可以从各个层次优化 IR：例如：

在 affine 层面，可以根据循环大小做展开，向量化；

在 scf 层面，可以发现循环不变量；

在 arith 层面，可以用算数恒等式优化代码。

比如在 linalg 层，我们很容易发现矩阵被转置了两次，但一旦 lower 到 scf，所有转置操作都变成循环，优化就很难进行了。所以从high level IR 到 low level IR 要及时做优化。

MLIR用处是：

复用已有的 Dialect；扩展已有的 Dialect；复用已有的 Pass。常见 Pass 直接复用（CSE DCE）

一般讲 high level 的 IR 是与硬件无关的，low level 的 IR是与硬件有关的。

## 2. MLIR的结构

MLIR 结构是树形的，Region 包含Block，Block 包含 Operation，Operation 包含其他的 Region，...。

~~【done】Repo 从另一个角度，其实是基本角度，理解MLIR，动手试试这个repo中的 基本用法。~~

## 3. MLIR Op的结构

Op 的Attribute 指的编译器已知的量，而 Operand 指只有运行时才能知道的量。

`%c0 = arith.constant 0 : i32`

上述0 是属性，而不是操作数。

## 4. MLIR Op的存储格式

MLIR 的所有 Op 都有一个统一的储存格式，叫 Operation。Operation 里面存了 OpName 和所有的 operands, results, attributes 和其它的东西。

## 5. MLIR的图结构

MLIR 里，有两个层次的图：

第一个是 Region 嵌套构成的树，这个图表示 **控制流**
第二个是 Op/Value 构成的图，这个图表示 **数据流**

- 数据流的遍历很修改：
Op的 `getOperands、getResults、getOpOperands` 非常常用。
Value 的 `getDefiningOp、getUses、getUsers`

- 控制流的遍历和修改：
`op.getParentOp, op.getParentOfType`：获取父亲Op
`op.getBlock`：注意是返回父亲block，而不是函数block
`op.getBody`：这个才是返回内部 block / region

`op.walk`：递归地遍历所有子孙 op
`block`：直接就是一个 iterator，可以直接遍历

控制流图的修改主要用 OpBuilder 完成。强烈建议把找到 OpBuilder 的代码，把里面有什么函数都看一看，常见的：

`builder.create`：创建op
`builder.insert`：插入remove的op
`op->remove()`：从当前块移除，但不删除，可以插入到其他块内
`op->erase()`：从当前块移除，并且删除

## 6. Basic Dialect project

- Tablegen Language Server
vscode 提供 mlir 扩展，可以为我们写 tablegen 文件提供帮助：找到你编译好的 `llvm-install/bin/mlir-lsp-server`，
在 vscode 的设置里找到 `mlir-lsp-server` 的设置，设好绝对路径，还有 database 的路径。

同理，找到
`mlir-pdll-lsp-server`
`tblgen-lsp-server`

分别设置

toy-opt 可执行文件可以添加 `--canonicalize` `--cse` 做这两个优化，凡是Op中标记了 Pure，它会帮助我们注册对应Op 的 CSE DCE。

## 7. `LogicalResult`

MLIR 用 `LogicalResult` 用来表示类似 bool 的值，它的特点是：

mlir的一些其他类型可以自动转换为 LogicalResult，如上面 emitError 就可以自动转换
用 success(), failure() 生成 true 和 false
用 succeed(x), failed(x) 来判读是否为 true, false

## 8. Op 声明时：

`Variadic<Type>` 来描述可变参数：`let arguments = (ins Variadic<AnyInteger>:$inputs);`
`Optional<AnyInteger>` 来描述可选参数：`let arguments = (ins Optional<AnyInteger>:$data);`

## 9. AssemblyFormat

常用关键字：

- `$xxx` 用来表示 operand 或者 attribute
- `type($xxx)` 用来表示 xxx 的类型。
- `keyword`： 插入 keyword
- `functional-type($inputs, results)`，生成形如 (i32, i32) -> i32 的函数类型
- `attr-dict`：表示额外的 attr 字典。

## 10. 为 Op 添加自定义函数

tablegen 允许用户为 Op 添加自定义函数，例如，我想直接获取 ConstantOp 的类型的位宽：
~~~mlir
def ConstantOp : ToyOp<...> {
  let extraClassDeclaration = [{
    int64_t getBitWidth() {
      return getResult().getType().getWidth();
    }
  }];
}
~~~
这样，之后想要获取位宽的时候，就可以更简洁了：

~~~mlir
auto w = op.getResult().getType().getWidth();
auto w = op.getBitWidth();
~~~

可以只在 tablegen 里写一个方法定义，然后在 toy.cpp 里面写实现（**使用 ninja MLIRToyIncGen 生成头文件** 就可以在.cpp 中include了）


## 11. 构建, 找到了 函数 `add_mlir_dialect` 的定义：

~~~sh
# Declare a dialect in the include directory
function(add_mlir_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(MLIR${dialect}IncGen)   # 这里是得到的target的名字
  add_dependencies(mlir-headers MLIR${dialect}IncGen)
endfunction()
~~~


## 12. ninja MLIR${dialect}IncGen

指定build 哪个target


## 13. Traits

MLIR 提供了很多好用的 Traits，被 Pure 标记的 Op 会自动注册 CSE，DCE Pass。

使用 Traits 注意：

- Interface 可能会要求用户实现一些固定的接口
- 在 td 里要 include trait 的 td 文件，在 h 里也要 include 对应的 h 文件

其他Traits：
`SideEffectInterfaces`
`InferTypeOpInterface`
`SameOperandsAndResultType`
`InferTypeOpAdaptor`   推荐使用

~~~sh
# td中
def ConstantOp : ToyOp<"const", [Pure, InferTypeOpAdaptor]> {
  let summary = "const operation";
  let arguments = (ins APIntAttr:$value);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$value attr-dict"; // 这里不需要写 type($result) 了
}
~~~

## 14. TableGen 定义函数，FunctionOpTrait

定义Dialect 中的函数比如 FuncOp、CallOp、ReturnOp 相关的操作是为了让方言能够完整地表达程序的**控制流**和**函数调用机制**。这些定义使得 MLIR 能够**支持高级编程语言的特性**，如函数抽象、调用和返回机制，同时也**为优化和转换**提供了基础。

方言不仅限于定义特定领域的**操作和类型**，还可以定义与**控制流相关**的基本构建块，如 func、call 和 return。这些构建块是许多编译器和转换工具的核心部分，它们使得 MLIR 能够作为一个多层次、可扩展的编译器基础设施。

虽然许多方言可能会定义自己的 func、call 和 return 操作，但 MLIR 的标准方言（Standard Dialect）已经提供了这些操作的通用实现。如果你的方言需要特定的行为，你可以选择扩展或重新定义这些操作以满足你的需求。

实战：ex4-beautiful-dialect  **函数的这一套代码非常固定，每次照搬就好，没有太多的解释** (暗指了，mlir只是一个框架，主要的是用它做什么而不是mlir本身)


## 15. 添加Pass

定义了一个Dialect和Op，即定义了一个IR，但只有IR没有啥意义，需要有在IR上运行的Pass。Pass 通过tablegen来定义。

两类Pass 指定Op的 + 用相同Interface实现的一类Op

其中还可以指定带参数的Pass。

:brain:
**Pass 的实现，就是灵活使用 IR 的遍历与修改**
:brain:

## 16. Patter rewrite

Pattern 会匹配 IR 的一个子图，然后将其更改为新的格式。

`ex6-opt` 可以用 `--debug` 来启动程序，程序会打印出转换的详细过程。

Lowering 可以通过 `OpConversionPattern` 或 `OpRewritePattern` 实现。

实例：
~~~cpp
struct ConvertToyToArithPass
    : toy::impl::ConvertToyToArithBase<ConvertToyToArithPass> {

  void runOnOperation() final {
    // 1
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    // 2
    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpPat, SubOpPat, ConstantOpPat>(&getContext());
    // 3
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
}
~~~

这里我们使用了 applyPartialConversion，MLIR 支持三种 Conversion 模式：
1. partialConversion：如果 Pattern 转换结果是 Legal，则保留转换结果。如果输入存在 IllegalOp 或 IllegalDialect，立刻报错。
2. fullConversion：开始时可能是 Illegal 的。调用 Pattern 将其转换，直到全部 Legal 为止。
3. greedyPatternRewrite：不需要提供 Target，贪心地尝试尽量多次修改。

前两个常用于 Dialect Lowering 之中。而 geedyPatternRewrie 很适合用来写优化，比如我可以写一个把形如 toy.sub %a, %a 替换为 const 0: i32 的 pattern，希望 MLIR 尽量多优化它。

## 17. Type conversion

上文道：`OpConversionPattern` 特别用于Op lowering，它会把Type进行转换
`addConversion`：添加一个 Type 的转换规则
`addTargetMaterialization`：生成将 SourceType 转换为 TargetType 的代码块

## 18 Tips

MLIR 为我们写好了大量的 Dialect，我们想要的功能，那些 dialect 多半都已经实现过了。

可以用 `mlir-opt --help`，`mlir-opt --help-hidden` 看看有那些 dialect 哪些选项，找到可能是和自己想要做的相似的，然后过去看代码，边看边抄大概就能实现好了。


## 19 Vscode 中跳转到函数类的定义

在 .vscode/ 下添加 c_cpp_properties.json
~~~json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/home/junhui/workspace/llvm-project/llvm-install/include/**",
                "${workspaceFolder}/include/**",
                "/home/junhui/workspace/llvm-project/mlir/include/**",
                "/home/junhui/workspace/llvm-project/build/tools/mlir/include/mlir/IR/**",
                "/home/junhui/workspace/llvm-project/mlir/include/mlir/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/clang",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-clang-x64"
        }
    ],
    "version": 4
}
~~~


## 20. addNestedPass 和 addPass

1. addNestedPass

当你使用 `addNestedPass` 时，你需要指定操作的类型，这个 Pass 将只会在这些操作的上下文中运行。如

`pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());`

在这个例子中，`createDecomposeAggregatedOps` 创建的 Pass 将作为嵌套 Pass 添加到所有 `func::FuncOp` 操作中。这意味着，只有当 Pass 管理器遍历到 `func::FuncOp` 操作时，`createDecomposeAggregatedOps` 创建的 Pass 才会执行。

2. addPass

addPass 方法用于向 Pass 管理器添加一个全局 Pass，这种 Pass 作用于整个 IR 或 Pass 管理器当前管理的 IR 层次。如：

`pm.addPass(createFoldTensorOperation());`

在这个例子中，createFoldTensorOperation 创建的 Pass 将作为全局 Pass 添加到 Pass 管理器中。这个 Pass 将作用于整个模块或当前 Pass 管理器正在处理的 IR 层次，而不局限于特定类型的操作。

总结：
addNestedPass 通常用于那些需要在特定操作的上下文中进行优化或变换的场景，例如在函数操作内部进行局部优化；
addPass 通常用于那些需要在更广泛范围内进行优化或变换的场景，例如跨多个函数或整个模块的优化。



## 什么情况下，需要给我的 Dialect 的某个Op 指定 let hasCustomAssemblyFormat = 1

来指示该操作有一个自定义的汇编格式。这意味着你需要为这个操作提供自定义的打印（printer）和解析（parser）逻辑，而不是使用 MLIR 默认生成的逻辑。

需要自定义的情况：
复杂的操作数或属性，特殊的语法要求，性能优化，与现有格式兼容


## mlir 我的td文件中含有这两个声明：

`def MithSwitchBarFoo: Pass<"mith-switch-bar-foo", "::mlir::ModuleOp"> {}`
`def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {}`
这两者的区别是什么？可以放在同一个文件中吗？

一个Pass可以作用于不同的IR层次，例如模块级别、函数级别或者更细粒度的操作级别。

- Pass 默认是关于IR的所有Op的，它定义了编译器的一个阶段，用于分析和修改IR。
- OpInterface 是应用于都使用 OpInterface 定义的Op的。


## Pass有OperationPass 和 InterfacePass， 这里两种Pass 的区别是什么？AI compiler过程中为什么需要这两种 Pass？

1. Operation Pass： Operation Pass 是针对特定 MLIR 操作类型的 Pass。它通常用于对特定类型的操作或操作的集合进行变换和优化。例如，你可以有一个 OperationPass 专门用于优化所有的 AffineForOp 操作。OperationPass 可以在操作的粒度上进行细粒度的变换。
在 AI 编译器过程中，Operation Pass 可以用于执行特定操作的优化，如循环展开、常量传播、死代码消除等。

2. InterfacePass： Interface Pass（在 MLIR 中通常称为 OpInterface Pass）是针对实现了特定接口的所有操作的 Pass。这种类型的 Pass 不是针对特定的操作类型，而是针对实现了特定接口的所有操作。例如，你可以有一个 OpInterface Pass 用于优化所有实现了 MemoryEffectOpInterface 的操作。
在 AI 编译器过程中，OpInterface Pass 可以用于跨不同操作类型的优化，如内存访问优化、数据流分析等，这些优化依赖于操作的接口而不是操作的具体类型。

两种Pass 它们提供了不同层次的优化能力。

## Pass 和 PatternRewrite 区别和侧重点

1. ~~Pass 通常有更广泛的视野，可以访问和修改整个 IR 结构。它可以对整个模块、函数或特定类型的操作进行变换和优化。可以是模块级别的，也可以是操作级别的。Pass 通常由 Pass 管理器管理，并按照特定的顺序执行。~~

2. ~~Pattern Rewrite 是 MLIR 中的一种局部变换机制，它使用模式匹配和替换来优化和变换操作。每个模式定义了一个特定的匹配规则和相应的替换规则。当一个操作与模式匹配时，它会被替换为模式指定的新操作或操作序列。Pattern Rewrite 提供了一种局部的、基于规则的变换机制。Pattern Rewrite 通常在 Dialect Conversion Framework 中使用，该框架允许将一种方言的操作转换为另一种方言的操作。~~

关系是，通过 PatternRewrite 来实现一个Pass



## OpConversionPattern VS OpRewritePattern

1. OpConversionPattern: 特指Lowering，将一个IR的Op转换为另一个IR的Op，
2. OpRewritePattern: 用于更一般的场景，但是不涉及类型转换，对Op进行匹配和重写，专注于Op的结构。比如 `transpose(transpose(x)) -> x` 的匹配和重写。当然也可以是Op的lowering
功能包含关系，都可以实现Lowering Op，如下两个Lowering 分别继承了 上述两者：
~~~cpp

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};
~~~

## 使用td文件定义Pass时，是否给出 constructor

构造函数（constructor）用于初始化 Pass 的实例。在 .td 文件中，你可以指定 Pass 的构造函数，以便在创建 Pass 实例时传递参数或执行初始化代码。

如果你在 .td 文件中为 Pass 提供了构造函数，那么当 Pass 被创建时，这个构造函数将被调用。这允许你传递配置选项或其他参数给 Pass。

~~~tablegen
def MyPass : Pass<"my-pass"> {
  let constructor = "createMyPass(OptionType option)";
  let options = (ins Option<"option", "OptionType", "default_value",
                            "Description of the option">:$option);
}
~~~

如果你没有在 .td 文件中指定构造函数，MLIR 的 TableGen 将使用**默认的无参数构造函数**来创建 Pass 的实例。在这种情况下，**你的 Pass 应该有一个无参数的构造函数，或者你需要在 C++ 代码中定义一个显式的构造函数**。进而，如果你需要在 Pass 中使用参数或配置选项，但 .td 文件中没有定义构造函数，你可以在 C++ 代码中手动添加构造函数和相应的创建函数。例如：

~~~cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  MyPass(OptionType option) : optionValue(option) {}

  OptionType optionValue;
};

std::unique_ptr<Pass> createMyPass(OptionType option) {
  return std::make_unique<MyPass>(option);
}
~~~

## 如何执行mlir文件得到计算结果？

搜索 `mlir-cpu-runner`, 这个文件可以及时执行得到结果

~~~sh
mlir-opt /home/junhui/workspace/graph-compiler/llvm-project/mlir/test/Integration/Dialect/ControlFlow/assert.mlir -test-cf-assert -convert-func-to-llvm | mlir-cpu-runner -e main -entry-point-result=void

mlir-opt /home/junhui/workspace/graph-compiler/llvm-project/mlir/test/Integration/Dialect/Linalg/CPU/runtime-verification.mlir \
 -generate-runtime-verification \
 -one-shot-bufferize="bufferize-function-boundaries" \
 -convert-linalg-to-loops \
 -expand-strided-metadata \
 -lower-affine \
 -convert-scf-to-cf \
 -test-cf-assert \
 -convert-index-to-llvm \
 -finalize-memref-to-llvm \
 -convert-func-to-llvm \
 -reconcile-unrealized-casts | \
 mlir-cpu-runner -e main -entry-point-result=void \
     -shared-libs=./llvm-project/build/lib/libmlir_runner_utils.so,./llvm-project/build/lib/libmlir_c_runner_utils.so \
     2>&1 | FileCheck /home/junhui/workspace/graph-compiler/llvm-project/mlir/test/Integration/Dialect/Linalg/CPU/runtime-verification.mlir

mlir-opt /home/junhui/workspace/graph-compiler/llvm-project/mlir/test/Integration/Dialect/Linalg/CPU/test-collapse-tensor.mlir \
 -one-shot-bufferize="bufferize-function-boundaries" \
 -finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
 -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-cf-to-llvm -convert-arith-to-llvm \
 -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
 mlir-cpu-runner -e main -entry-point-result=void \
   -shared-libs=./llvm-project/build/lib/libmlir_runner_utils.so,./llvm-project/build/lib/libmlir_c_runner_utils.so

mlir-opt /home/junhui/workspace/graph-compiler/llvm-project/mlir/test/Integration/Dialect/Linalg/CPU/test-conv-1d-call.mlir \
  -test-transform-dialect-erase-schedule \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-arith-to-llvm \
  -convert-scf-to-cf \
  --finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts | mlir-cpu-runner -e main \
    -entry-point-result=void \
    -shared-libs=./llvm-project/build/lib/libmlir_runner_utils.so,./llvm-project/build/lib/libmlir_c_runner_utils.so
~~~

`mlir-cpu-runner` 是MLIR项目提供的一个工具，它用于直接执行MLIR文件中定义的函数，而不是生成可执行文件。当你使用 `mlir-cpu-runner` 时，它会在运行时编译MLIR代码并执行指定的函数。这个过程通常涉及将MLIR代码转换为LLVM IR，然后使用即时编译（JIT）技术执行生成的代码。使用`-e`选项后跟函数名（如main）可以指定要执行的函数。`mlir-cpu-runner`会查找`MLIR`文件中名为`main`的函数，并执行它。这个工具主要用于测试和调试MLIR代码，它允许开发者快速运行和验证MLIR函数的行为，而无需经过完整的编译链接生成可执行文件的过程。

