+++
date = '2025-08-31T12:57:45+08:00'
draft = false
title = 'Python'
tags = ["python"]
categories = ["tools"]
+++


##  pyenv && poetry

~~~sh
sudo apt update
sudo apt upgrade
pyenv install 3.9.0 -v #建议不要使用系统python，而是为虚拟环境安装独立python
# 如果上述命令失败，看看log，大概率是某个package没有安装，log 通过 去掉 ‘-v’

# 创建一个名为 ppo_me 的 virtualenv，它基于 Python 3.9.0 版本。
pyenv virtualenv 3.9.0 ppo_me

# 激活虚拟环境。在当前目录下创建一个 .python-version 文件，并将 ppo_me 写入该文件。 当你进入这个目录或其子目录时，pyenv 会自动读取 .python-version 文件，并将 Python 环境设置为 ppo_me 这个 virtualenv。
pyenv local ppo_me

# 删除ppo_me 的虚拟环境，并删除对应的Python版本。
pyenv deactivate
pyenv virtualenv-delete ppo_me
pyenv uninstall 3.10.14
rm -rf .python-version 


pyenv install 3.9.0
pyenv virtualenv 3.9.0 ppo_me
# 创建一个 .python-version 文件，指定该项目使用的虚拟环境。 当你进入项目目录时，pyenv 会自动激活该虚拟环境。
pyenv local ppo_me

~~~

【KAQ】： 然后在当前目录下 执行 which python， 还是返回系统 Python 版本，而不是 指定的虚拟空间中的3.9.0 为什么? 通过 pyenv rehash

【KAQ】 : 如何证明我是在一个 pyenv 中的？

~~~sh
$ which python
/home/junhui/.pyenv/shims/python
$ echo $PYENV_VERSION
ppo_me
$ pyenv versions
  system
  3.9.0
  3.9.0/envs/ppo_me
* ppo_me --> /home/junhui/.pyenv/versions/3.9.0/envs/ppo_me (set by PYENV_VERSION environment variable)
~~~

【KAQ】: 回到上一级目录后，python 版本还是虚拟环境中的版本，为什么不是系统 Python？


安装使用 poetry 包管理工具
~~~sh
sudo apt install python3-poetry
poetry init # 指明python 版本是虚拟环境中的
poetry add gym torch stable-baselines3 tensorboard wandb -vvv
~~~

暂时没有用明白。在那时还用 conda 


## lambda

`(lambda base, height: 0.5 * base * height)(4, 8)` 返回 16.0

Lambda 函数在你想定义小型、单次使用的函数时非常方便

## 在 ide 中 import 有黄色波浪线

包正确安装的前提下，IDE 没有找到你期望的 **Python 解释器**。选择正确的虚拟环境中的正确的解释器。


## 函数体暂停执行 (pdb)

使用 `breakpoint()`, 或 `pdb.set_trace()`，暂停执行，进入pdb模式。

`pass` 不行，因为 `pass` 只是占位符, 表示什么都不做。当 Python 解释器执行到 `pass` 语句时，它会直接跳过该语句，继续执行后面的代码，而不会中断程序的执行。

pdb中打印变量 `pp args.keys()`


## W&B (Weights & Biases) 

是一个机器学习开发平台，它提供了一系列工具，用于跟踪、可视化和协作处理机器学习实验。


## conda && pyenv

两者功能是几乎一样的，用一个就好。 

conda 确实有自己的管理 Python 包依赖的功能，因此在某些情况下，conda 可以替代 pyenv + poetry 的组合。


## poetry && pyenv

- pyenv: 负责管理 Python 的版本，精确控制python 版本。 它可以让你在同一台机器上安装和切换多个 Python 版本。
- Poetry: 负责管理项目的依赖和虚拟环境。 它可以让你声明项目依赖，并创建一个独立的虚拟环境来隔离这些依赖。**poetry 使用 pip 作为其底层包管理工具，因此可以与 pip 生态系统无缝集成**。 如果你更喜欢使用 pip 生态系统，或者需要使用一些 conda 无法提供的 Python 包，那么 poetry 是一个更好的选择。


## conda-forge 

~~~sh
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# 当下载速度太慢而无法进行时，尝试下载到其他设备后通过 scp 传输到目标设备
bash Miniforge3-$(uname)-$(uname -m).sh

# 初始化到配置文件 ~/.bashrc
export PATH="~/miniforge3/bin:$PATH"
source ~/miniforge3/etc/profile.d/conda.sh
~~~

在一个 conda 环境中推荐使用 conda install 安装包，当指定的包不在 conda 源中，可以是用 pip install 安装，这也会安装在当前 conda env 中。最佳实践是通过创建 `environment.yml` 文件：

~~~yml
name: myenv
dependencies:
  - python=3.10.12
  - numpy
  - pandas
  - pip
  - pip:
    - requests
    - beautifulsoup4
~~~

然后通过 conda 命令创建环境：`conda env create --file environment.yml`

如果需要更新环境，先更新 yml 文件，然后使用 `conda env update --file environment.yml --prune`，prune 是移除环境中不再使用的包。

使用conda search pandas 可以检查pandas 是否在 conda 源中。

## wait 操作符 和异步code
## 实现一个异常类，捕获时，客制化打印语句

~~~py
class MyCustomError(Exception):
    def __init__(self, message="An error occurred in MyCustomError"):
        self.message = message  # 存储额外的错误信息，如错误代码
        super().__init__(message)  # 调用父类的__init__方法，传入message
        
    # 返回自定义的错误信息字符串  
    def __str__(self):
        return f"[Error: {self.message}"  

# 示例函数，该函数在特定条件下引发自定义异常  
def test_func(condition, message):  
    if condition:  
        raise MyCustomError(message)  

# 调用示例函数并捕获自定义异常  
try:
    test_func(True, "Resource not found")  
except MyCustomError as e:  
    print(e)  # 这将调用e的__str__方法
~~~


## __str__ 

它用于定义一个对象的字符串表示形式。当使用`str()` 函数或 `print()` 函数打印一个对象时，Python 会调用该对象的 `__str__` 方法来获取其字符串表示。

~~~py
class Person:  
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):  # 重写（override）来自定义对象的字符串打印方式
        return f"Person(name={self.name}, age={self.age})"

p = Person("Alice", 30)
# 打印Person对象，将调用__str__方法 
print(p)  # 输出: Person(name=Alice, age=30)
~~~

## DAG Python实现

~~~py
from typing import Dict, List

class DAG:
    def __init__(self):
        self.graph: Dict[str, List[str]] = {}

    def add_node(self, node: str) -> None:
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, from_node: str, to_node: str) -> None:
        if from_node not in self.graph:
            self.add_node(from_node)
        if to_node not in self.graph:
            self.add_node(to_node)
        self.graph[from_node].append(to_node)

    def print_graph(self) -> None:
        if not self.graph:
            print("Graph is empty!")
            return
        for node in self.graph:
            print(f"{node} -> {self.graph[node]}")

    def __topological_sort_util(self, node: str, visited: Dict[str, bool], stack: List[str]) -> None:
        print(f"node -: {node}")
        visited[node] = True
        for neighbor in self.graph[node]:
            if not visited[neighbor]:
                self.__topological_sort_util(neighbor, visited, stack)
        print(f"node +: {node}")
        stack.insert(0, node)
        print(f"stack: {stack}")

    def topological_sort(self) -> List[str]:
        visited: Dict[str, bool] = {node: False for node in self.graph}
        stack: List[str] = []
        for node in self.graph:
            print(f"{node}")
            if not visited[node]:
                self.__topological_sort_util(node, visited, stack)
        return stack


if __name__ == "__main__":
    dag = DAG()
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    dag.add_edge("D", "E")

    # dag.print_graph()
    print(dag.topological_sort())

# output
"""
A
node -: A
node -: B
node -: D
node -: E
node +: E
stack: ['E']
node +: D
stack: ['D', 'E']
node +: B
stack: ['B', 'D', 'E']
node -: C
node +: C
stack: ['C', 'B', 'D', 'E']
node +: A
stack: ['A', 'C', 'B', 'D', 'E']
B
C
D
E
['A', 'C', 'B', 'D', 'E']
"""
~~~

上述拓扑排序的算法是`DFS`，保证每一个节点只被访问一次。理解`DFS`的关键是**计算机如何执行的，你就如何想**。**递归就是要先跳进，进入最里边，然后逐渐跳出，跳到最外边，在这个过程中，做了额外的操作**：

~~~txt
A -> B -> D -> E
     B <- D <- E
     C
A <- C
~~~

DAG 相关概念：
1. 顶点（Vertex）：顶点是图中的基本单位，表示图中的一个节点。在 DAG 中，顶点可以表示任务、事件、状态等。

2. 边（Edge）：边是连接两个顶点的线段。在有向图中，边有方向，从一个顶点指向另一个顶点，表示一种依赖关系或顺序。

3. 入度（In-degree）：一个顶点的入度是指有多少条边指向该顶点。在 DAG 中，入度为 0 的顶点通常表示没有依赖关系的起始点。

4. 出度（Out-degree）：一个顶点的出度是指从该顶点出发有多少条边。在 DAG 中，出度为 0 的顶点通常表示没有后续依赖的终止点。

5. 路径（Path）：路径是从一个顶点到另一个顶点的一系列边。在 DAG 中，路径是有方向的，并且由于无环性，路径不会回到起点。



## Torch EagerMode 和 GraphMode

- EagerMode：及时执行，依赖 Python 解释器
- GraphMode：不依赖Python解释器，TorchScript 是一个中间表示，是一个静态图，可在 c++ 的 runtime 环境中执行。优势：1. 减少了Python解释器的开销。2. 适合嵌入式设备。3. TorchScript 是跨平台的。

实例：

~~~py
import torch
import torch.nn as nn
# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet()
# 将模型转换为TorchScript
scripted_model = torch.jit.script(model)
# 保存TorchScript模型
scripted_model.save("simple_net.pt")
~~~

在C++应用程序中使用 `LibTorch` 加载和执行:

~~~cpp
// 需要下载和配置LibTorch，
#include <torch/script.h> // One-stop header. 通过包含一个单一的头文件，开发者可以访问库或框架的所有功能
#include <iostream>
#include <memory>

int main() {
    // 加载TorchScript模型
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("simple_net.pt");
    // 创建输入张量
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 10}));
    // 执行模型
    at::Tensor output = module->forward(inputs).toTensor();
    // 打印输出
    std::cout << output << std::endl;
    return 0;
}
~~~
提高性能、简化部署并增强可移植性。

Note: `LibTorch` 是 PyTorch 的 C++ 库版本，允许开发者在 C++ 环境中使用 PyTorch 的功能。可以从 PyTorch 官方网站下载预编译的 `LibTorch` 库，并使用 CMake 配置和构建我的项目。`LibTorch` 提供了与 PyTorch Python API 类似的接口，使得开发者可以在不依赖 Python 解释器的情况下进行深度学习模型的开发和部署。


## setup.py

setup.py 文件中的内容通常用于构建 & 打包含有 C++ 项目的 python 扩展。会有一个类 `class CMakeBuild()`，其中有 方法 `build_extension(self)` ，其作用是构建和编译 C++ 项目。方法 `run(self)` 的作用是将 `build_extension()` 执行起来。最后使用 `setuptools.setup` 工具打包为 Python 包。详细实例见 opensource repo 中的`setup.py` 文件。

打包函数 `setup()` 中有很多可配置参数，其中有

~~~py
cmdclass={
    'build': CMakeBuild,
    'custom_command': CustomCommandClass,
    }
~~~

`cmdclass` 参数是字典，它用于将自定义命令与` setuptools` 的命令接口关联起来。这样用户就可以通过 `Python setup.py build` 和 `python setup.py custom_command` 执行对应的动作。


## @property

`@property` 装饰器是一个内置装饰器，用于将一个方法转换为 `getter` 方法。这允许你将一个类的方法以属性的方式访问，而不需要在调用时使用括号。同时，使用这个装饰器也是访问类私有成员的优雅方法

~~~py
@property
def pass_rate(self):
    """Pass rate."""
    if self.total == 0:
        return 0.0
    return round(100 * self.passed / self.total, 2)

obj = MyClass()
print(obj.pass_rate)  # 不需要括号
~~~

## 外部函数修改类私有成员的方法

- 通过 Name Mangling。Python会将私有成员的名称改写为 _ClassName__MemberName 的形式
- getter & setter
- 使用property装饰器
- 使用反射（Reflection）。使用Python的内置函数getattr和setattr来访问和修改私有成员


## hashlib 做什么的？

它提供了一系列常见的哈希算法接口，允许开发者**进行数据的哈希计算**。哈希算法是一种将任意长度的数据转换为固定长度（通常是较短的）数据的函数，这个转换过程是不可逆的，即不能从哈希值恢复原始数据。**哈希算法通常用于确保数据的完整性和验证数据的来源**。

~~~py
def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest() # 使用hashlib.sha256根据提供的源代码src生成一个MD5哈希值作为缓存键。这个哈希值用于唯一标识这段源代码，以便在缓存中查找和存储编译后的模块
    cache = get_cache_manager(key)         # 调用get_cache_manager函数，传入上一步生成的哈希值key，以获取一个缓存管理器实例
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dir, include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
~~~

解释上述函数是做什么的？其中的 hashlib.md5 是做什么的？

`hashlib.md5` 函数用于计算字符串的MD5哈希值。在这个上下文中，它用于根据源代码内容**生成一个唯一的标识符**，这样相同的源代码会映射到相同的哈希值，从而可以使用这个哈希值作为缓存键来存储和检索编译后的模块。
`usedforsecurity=False` 参数表明这个MD5哈希**不是用于安全目的**（如密码散列），而是**用于缓存键生成**，这是因为MD5不再被认为是安全的哈希函数。

这个函数是一个编译和缓存系统，用于将源代码编译为动态链接库，并将其加载为Python模块，同时使用缓存机制来避免重复编译相同的源代码。


## hashlib   【检查数据的完整性】

~~~py
import hashlib

def get_sha256():
    data = "hello world"
    hash_object = hashlib.sha256()
    hash_object.update(data.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    print(hash_digest)

def check_file():
    def get_hash_digits(data):
        hex_dig = hashlib.sha256(data.encode()).hexdigest()
        return hex_dig

    original_data = "Hello, world! This is a test to ensure data integrity."
    original_hash = get_hash_digits(original_data)
    print(f"Original Hash: {original_hash}")

    received_data = "Hello, world! This is a test to ensure data integrity."
    received_hash = get_hash_digits(received_data)
    if original_hash == received_hash:
        print("Data is compileted, not tampered!")
    else:
        print("Alert data has been tampered!")

    tampered_data = "Hello, world! This is a test to BREAK data integrity."
    tampered_hash = get_hash_digits(tampered_data)
    if original_hash == tampered_hash:
        print("Data is compileted, not tampered!（no possible in this case）")
    else:
        print("Alert data has been tampered!")
~~~



## 捕获异常  【不一定要使用】

try catch 和不使用 try catch ？

使用 try 和 except 块来捕获异常是一种处理潜在错误的方法，但它并不是强制性的。是否捕获异常取决于你的特定情况和你想要如何处理错误。在某些情况下，**你可能不想在函数内部立即处理异常，而是希望将异常传播给调用者，让调用者决定如何处理**。这种情况下，你可以直接在函数中抛出异常，而不是使用 try 和 except 块。

~~~py
def check_cols(target_cols, all_cols):
    diff = set(target_cols).difference(all_cols)
    if len(diff) != 0:
        raise ValueError(f"Couldn't find required columns: '{diff}' among available '{all_cols}'")

# 在调用函数的地方
check_cols(target_cols, all_cols)  # 可能会抛出异常
~~~

~~~py
def check_cols(target_cols, all_cols):
    diff = set(target_cols).difference(all_cols)
    if len(diff) != 0:
        raise ValueError(f"Couldn't find required columns: '{diff}' among available '{all_cols}'")
try:
  check_cols(target_cols, all_cols)
except ValueError as e:
    print(f"Error: {e}")
~~~

如果你的函数是一个库函数，或者你正在编写一个API，那么通常最好让异常传播给调用者，因为调用者可能有更多的上下文来决定如何处理这些异常。

- `ValueError` 通常用于指示一个函数接收到了一个正确类型但不合适的值。例如，如果一个函数期望一个特定范围或条件的参数，而传入的参数不满足这些条件，就可以抛出 `ValueError`。
- `AssertionError` 通常用于在代码中显式地表示某个断言（assert）失败。断言用于在代码中设置检查点，这些检查点用于确保程序中的某个条件为真。如果条件不为真，则引发 `AssertionError`。
- `RuntimeError` 是一个更一般的异常，用于指示在运行时发生的错误，这些错误不属于其他更具体的异常类别。当一个错误不容易归类到其他更具体的内置异常时，通常会使用 `RuntimeError`。

`assert 1 == 2, "1 不应该等于 2"`，条件 1 == 2 是假的，所以会引发 `AssertionError`，并打印出错误消息 "1 不应该等于 2"。
虽然 assert 语句会自动处理 `AssertionError` 的引发，但在某些情况下，你可能需要在代码中显式地引发 `AssertionError`，特别是当你想要基于更复杂的逻辑条件来验证某个状态时：

~~~py
def check_positive(number):
    if number <= 0:
        raise AssertionError("数字必须是正数")    # 显式地 raise AssertionError 也可以引发 AssertionError。

try:
    check_positive(-1)
except AssertionError as e:
    print(e)  # 输出: 数字必须是正数
~~~

用于debug个开发中，不用于产品中。


## __name__

`__name__` 的行为与普通全局变量不同，因为它不是由程序员直接赋值的。 它的值是由 Python 解释器在运行时自动设置的。 你不能像普通变量那样直接修改 `__name__` 的值。在Python中，`__name__` 变量主要有两个可能的值：

1. `__main__`：当Python文件被直接运行时，`__name__` 被自动设置为字符串 `__main__`。这是Python解释器的一个约定，用于指示该文件是程序的入口点。此时，Python解释器会读取该文件，并执行文件中定义的代码。

2. 模块名：如果Python文件被另一个文件通过import语句导入，那么 `__name__` 将被设置为该文件的模块名（即文件名，不包括.py扩展名）。在这种情况下，该文件中的代码将不会直接执行（除非它被明确地调用，比如通过调用函数或类），但该文件中的函数、类和变量等将被导入到当前的命名空间中，以便在其他地方使用。


## conda 恢复环境【充分利用当前工具中的内容】

~~~sh
conda list --export > miniconda_list.yml  # 其中给出了一键安装所有包, 如下
conda create --name <env> --file <this-file>
~~~

注意，找不到的包，可能是：
1. 没有指定的版本
2. 原本是通过pip install的
3. 原本是通过whl安装的

这些需要手动安装。


## pip install & conda install

无论是通过 pip 还是 conda 安装，包都**只会安装到你当前激活的环境中**。如果你想在不同的 conda 环境中使用同一个包，你需要在每个环境中分别安装它。

这是因为 pip 和 conda 都是环境管理工具，它们的设计目的是为了隔离不同项目的依赖，防止版本冲突，并使得环境可复制.

一个 conda 环境中通过 pip 安装的包，conda list 也可以访问这个包。

## pip install -e . --verbose

这句话会找当前文件夹下的 `setup.py`  文件，并且安装一个python 包，它可以通过 pip list 查看，使用方式是在 python 文件中 import 这个安装的包，进而使用这个包所提供的功能。很多 python 项目都是这样使用的。


## python 中 * 号的作用

1. 解包list或tuple
~~~py
a = [1, 2]
b = [3, 4]
c = [0, a, b, 5]  # 结果为 [0, [1, 2], [3, 4], 5]
c = [0, *a, *b, 5]  # 结果为 [0, 1, 2, 3, 4, 5]
~~~

可理解为扁平化

2. 两个星号（**）用于解包字典，将字典中的键值对作为关键字参数传递。

~~~py
kwargs = {'a': 1, 'b': 2}
print(**kwargs)  # 相当于 print(a=1, b=2)
~~~

3. 在函数定义中，*用于接收任意数量的位置参数，参数被打包成一个元组。

~~~py
def func(*args):
    for arg in args:
        print(arg)
~~~

4. 捕获剩余元素

`first, *rest = [1, 2, 3, 4]  # first = 1, rest = [2, 3, 4]`

5. 重复序列

`repeated_list = [0] * 5  # 结果为 [0, 0, 0, 0, 0]`

## from importlib import reload

在 Python 中，reload 函数（从 importlib 模块导入）用于重新加载一个之前已经导入的模块。这通常在你修改了模块的源代码后，希望在不重新启动 Python 解释器的情况下重新加载它时非常有用。

~~~py
import some_module  # 假设你已经有一个名为some_module的模块

# ... 在这里可能有一些使用some_module的代码 ...

# 现在你修改了some_module.py文件的内容，并且希望在不重启Python解释器的情况下重新加载它
from importlib import reload
reload(some_module)

# 现在some_module中的代码应该是你修改后的版本
~~~

## 使用 setuptools 构建一个python 包

setuptools是Python的一个工具集，主要用于构建和分发Python包，尤其是那些具有依赖关系的包. setuptools极大地简化了Python包的创建、分发和安装过程.

这段 Python 代码是一个 setuptools 配置脚本，用于构建和安装一个名为 "benchgc" 的 Python 包。这个包包含了一个自定义的 CMake 构建过程，用于编译一个名为 "graphcompiler" 的扩展模块。以下是代码的详细解释：

1. 导入模块： 导入了必要的 Python 模块，包括 pathlib、sys、setuptools、os 和 subprocess。这些模块提供了文件路径操作、系统相关信息、包安装工具、操作系统接口和子进程管理的功能。

2. 定义 PyscExtension 类： 这是一个继承自 setuptools.Extension 的自定义类。它用于表示一个扩展模块，但不包含源文件列表（sources=[]）。它接受一个名字和一个源代码目录，并将源代码目录解析为绝对路径。它还定义了一个 library_dirs 属性，这里设置为 "benchgc"。

3. 定义 PyscBuild 类： 这是一个继承自 setuptools.command.build_ext.build_ext 的自定义类。它重写了 build_extension 方法，用于构建 PyscExtension 扩展模块。在这个方法中，它使用 CMake 和 Make 工具来配置和编译扩展。

4. 构建过程： 在 build_extension 方法中，首先计算扩展模块的输出目录（extdir），然后定义了一系列 CMake 参数（cmake_args），包括输出目录、Python 解释器路径、构建类型和其他选项。如果环境变量 SC_LLVM_CONFIG 存在，它也会被添加到 CMake 参数中。

5. 运行 CMake 和 Make： 使用 subprocess.run 运行 CMake 配置命令，指定源代码目录和 CMake 参数，并在构建临时目录中执行。然后运行 Make 命令来实际编译扩展模块。

6. setuptools.setup 调用： 这是配置脚本的核心部分，它调用 setuptools.setup 函数来配置包的元数据和构建指令。它指定了包的名称、描述、包目录、包列表、依赖项、扩展模块和自定义构建命令类。

7. 安装和构建： 当你运行 python setup.py install 或 python setup.py build_ext 时，setuptools 会使用这个配置脚本来安装或构建 "benchgc" 包和 "graphcompiler" 扩展模块。

这个配置脚本是为了在 Python 包中集成 C++ 扩展模块的构建过程，使得 Python 包可以包含由 CMake 管理的本地代码。这种方法在深度学习、科学计算和其他需要本地性能优化的领域中非常常见。

TritonXPU 中的Benchdmark 模块作为一个实例


## 同名函数处理多种参数和返回值类型 && Union

`Union` 是一个来自 typing 模块的类型提示，它表示一个值可以是多个类型中的任何一个。`Union` 用于表示一个变量、函数参数或返回值可以是多个类型中的任意一个。

使用 `Union` 的好处是，它允许你编写更灵活和可重用的代码，因为函数可以处理不同类型的输入。同时，它也有助于提高代码的可读性和可维护性，因为通过查看类型提示，你可以清楚地知道函数期望什么样的输入，以及它会返回什么类型的输出。

下面两种写法功能上是等价的，但是

~~~py
def flip_coin(seed: Union[Any, torch.Tensor], prob: Union[float, torch.Tensor]) -> Union[bool, torch.Tensor]:
    pass
def flip_coin(seed: Any, prob: Any) -> Any:
    pass
~~~

后一种写法牺牲代码的可读性和类型安全性。在可能的情况下，最好使用更具体的类型提示，以便利用静态类型检查的好处，并提高代码的可维护性。



## 注意 torch 在处理浮点数精度和截断方面的问题   【改变运算顺序】

~~~py
>>> a = 109641275
>>> b = torch.tensor([109641275])

>>> a / 2
54820637.5
>>> b / 2
tensor([54820636.])

>>> a / 2 * 1637
89741383587.5
>>> (b / 2 * 1637).item()
89741377536.0
~~~

torch 中默认浮点数是float32，所以上述与python结果相差很多。
调整计算顺序，或者将初始值声明为float64


如何处理 torch 精度问题：

1. 使用更高精度的数据类型：你可以尝试使用双精度浮点数（float64）来进行运算，以减少精度损失。在PyTorch中，你可以通过指定dtype来实现这一点。
2. 注意运算顺序：有时候，改变运算的顺序或方式可以减少精度损失。例如，你可以先执行乘法再执行除法，或者将大数分解为小数来避免溢出。

## torch 的 tensor 广播是有规则的

从每个 tensor 的最右边（即最后一个维度）开始比较它们的形状。

1. 对于每个维度，如果两个 tensor 的形状相同，或者其中一个 tensor 的形状在该维度上是 1，那么这两个 tensor 在这个维度上就是兼容的，可以广播。
2. 如果在比较过程中发现任何维度不兼容（即两个 tensor 的形状在该维度上都不为 1 且不相等），则无法广播。
3. 广播过程中，形状为 1 的维度会“扩展”以匹配另一个 tensor 的对应维度。

在我们的例子中，even_indices 的形状是 `[4]`，而 tensor（假设我们仍然叫它 xxx）的形状是 `[4, 4]`。由于 even_indices 只有一个维度，而 xxx 有两个维度，我们需要在 even_indices 的前面添加一个维度，使其形状变为 `[1, 4]`。然后，这个额外的维度（值为 1 的维度）会沿着行方向扩展，以匹配 xxx 的形状



## 编译一个 c++ 函数为python 模块，给python使用

我在使用一个别人提供的 python 包，其中需要使用一个名为 vec_add 功能的函数，但包中的函数没有向量化的，我想添加一个自己的向量化版本的C++函数编译成python 接口，供我自己使用。

0. 安装pybind11

~~~sh
pip install pybind11
git clone https://github.com/pybind/pybind11.git
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/me_include
make install

# 通过-I给出pybind11的 位置，确保编译器能够找到 pybind11/pybind11.h
g++ -I~/me_include/inlclude -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` signature.cpp -o me_signature`python3-config --extension
-suffix`
~~~

1. 给出自己函数的源文件

~~~cpp
// vec_add.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::vector<double> vec_add(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result;
    result.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

PYBIND11_MODULE(vec_add_module, m) {
    m.def("vec_add", &vec_add, "Add two vectors element-wise");
}
~~~

2. 编译

~~~sh
c++ -I~/me_include/inlclude -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` vec_add.cpp -o vec_add_module`python3-config --extension-suffix`
~~~

执行过后会编译出一个共享库.so文件 `vec_add_module.cpython-310-x86_64-linux-gnu.so`.
或者通过cmake 来构建编译 (没有成功，编译结果没能成功import )

3. python 包中导入

~~~py
# __init__.py
from .vec_add_module import vec_add
~~~

如此就可以在python文件中调用自己的c++底层了

4. 库文件 `XXX.cpython-310-x86_64-linux-gnu.so` 为什么在py文件中直接 `import XXX` 就可以了？所有的.so文件都可以在py文件中使用吗？
答：`pybind11` 编译生成的 `.so` 文件（在 Linux 上是共享库文件）是专门为 Python 的 C 扩展设计的，并且包含了 Python 的解释器可以识别和加载的元数据。当你在 Python 脚本中 `import XXX` 时，Python 解释器会查找名为 `XXX.cpython-310-x86_64-linux-gnu.so`。

`import XXX` 能够成功的原因在于：

-  文件名约定：pybind11 生成的 .so 文件名遵循 Python 的命名约定，这允许 Python 解释器找到并加载它。文件名中的 cpython-310-x86_64-linux-gnu 部分表示这个库是为 CPython 3.10 版本，x86_64 架构，Linux GNU 系统构建的。
-  初始化函数：每个 C 扩展模块都必须有一个名为 `PyInit_<模块名>` 的初始化函数。这个函数由 pybind11 自动生成，并在 .so 文件中导出。Python 解释器在加载 .so 文件时会**查找这个函数，并调用它以获取模块的 Python 对象**。
-  Python 的模块机制：Python 的 import 机制知道如何查找和加载 .so 文件（在 Linux 上）作为扩展模块。它会在特定的目录（如 site-packages）中查找这些文件，并根据需要在运行时加载它们。


## 切片操作时，python 会自己处理索引

如果切片操作的起始或结束索引超出了序列的范围，它并不会引发错误，而是返回一个空序列。这是因为切片操作在内部会检查索引的有效性，并自动处理越界的情况。

## torch 中的操作有的有原地操作的版本

过度使用原地操作可能会导致难以追踪的错误，尤其是在复杂的计算图中。因此，在设计神经网络和其他复杂模型时，通常建议避免使用原地操作，除非有明确的性能需求或特殊原因。

## 运行时 bool 参数 不必赋值

~~~py
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--verbose', action='store_true', help='increase output verbosity')

args = parser.parse_args()

if args.verbose:
    print("Verbose mode is on.")
else:
    print("Verbose mode is off.")
~~~

如果你运行程序时没有提供 `--verbose` 参数，那么 `args.verbose` 的值将为 False，因此会打印 "Verbose mode is off."。
如果你提供了 `--verbose` 参数（例如，通过命令行运行 `python script.py --verbose`），那么 `args.verbose` 的值将为True，因此会打印"Verbose mode is on."。

即，这种方式时序提供 --verbose 或不提供，不能给他赋值。

给一个运行时参数**赋布尔值**，是不推荐的（因为不需要，用户给出就是True，没有给出就是False），因为 Python的argparse模块和其他许多命令行工具都遵循这样的约定


## 随机数种子1  【理解原理就都理解随机数了】

我有两种生成随机数的方式，解释他们的行为的不同？
~~~py
>>> ra = random.Random(23)
>>> ra.randint(10,100)
47
>>> ra.randint(10,100)
20
>>> ra.randint(10,100)
12
>>> ra.randint(10,100)
85

>>> random.Random(23).randint(10,100)
47
>>> random.Random(23).randint(10,100)
47
>>> random.Random(23).randint(10,100)
47
>>> random.Random(23).randint(10,100)
47
~~~
这两种生成随机数的方式之间的主要区别在于随机数生成器的实例化和种子设置的时机。

在第一种方式中：
~~~py
ra = random.Random(23)
ra.randint(10,100)
ra.randint(10,100)
ra.randint(10,100)
ra.randint(10,100)
~~~

你首先创建了一个名为 ra 的 random.Random 对象，并为它设置了种子 23。然后，你连续调用 ra 的 randint(10,100) 方法来生成随机数。
由于你只设置了一次种子，随机数生成器的内部状态在每次调用 randint 之后都会更新，因此每次调用都会产生不同的随机数。
这是因为随机数生成器是基于前一个状态来计算下一个状态的，所以即使种子相同，连续的调用也会产生不同的结果。

在第二种方式中：

~~~py
random.Random(23).randint(10,100)
random.Random(23).randint(10,100)
random.Random(23).randint(10,100)
random.Random(23).randint(10,100)
~~~

每次调用 randint(10,100) 时，你都创建了一个新的 random.Random 对象，并且每次都用相同的种子 23 对它进行初始化。
这意味着每次调用都是从**相同的初始状态开始的**，因此每次都会产生**相同的第一个随机数**。由于随机数生成器在*每次调用后都被丢弃*，内部状态不会保留，所以每次都会重复相同的序列。

总结来说，第一种方式是在单个随机数生成器实例上连续生成随机数，而第二种方式是每次都创建一个新的随机数生成器实例，并从相同的种子开始。
第一种方式会产生不同的随机数序列，而第二种方式每次都会重复相同的第一个随机数。

pytorch 中的随机数也是一样的。不能并行！


## 随机数种子2  【理解原理就都理解随机数了】

两次执行结果不同的本质原因是**随机数生成器的状态**在生成随机数后发生了变化
~~~py
torch.manual_seed(42)
random_tensor_uniform = torch.rand(1, 5)
random_tensor_uniform = torch.rand(1, 5)
# 生成：
tensor([[0.8823, 0.9150, 0.3829, 0.9593, 0.3904]])
tensor([[0.6009, 0.2566, 0.7936, 0.9408, 0.1332]])
~~~

首先设置了随机种子为42，然后生成了一个形状为`[1, 5]`的随机张量。当你再次调用torch.rand(1, 5)时，**随机数生成器的状态已经因为前一次生成随机数而改变**，
所以它会继续在随机数序列中向前移动，并生成一组新的随机数.

~~~py
torch.manual_seed(42)
random_tensor_uniform = torch.rand(1, 5)
torch.manual_seed(42)
random_tensor_uniform = torch.rand(1, 5)
# 生成：
tensor([[0.8823, 0.9150, 0.3829, 0.9593, 0.3904]])
tensor([[0.8823, 0.9150, 0.3829, 0.9593, 0.3904]])
~~~

第二次执行 将随机数生成器的**状态重置**为与第一次调用torch.manual_seed(42)时相同的状态。
因此，第二次生成的随机张量与第一次生成的随机张量相同，因为它们都是从**相同状态**的随机数生成器开始生成的。


## 随机数种子的作用域  【该线程全局】

作用域是全局的，对于该进程中的所有随机数生成操作都是有效的，直到种子被重新设置或者进程结束。


## 随机数种子的原理

与前一个状态有关的 取模运算。线性同余生成器： `X_{n+1} = (a * X_n + c) % m`
~~~py
class LCGRandom:
    def __init__(self, seed):
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32
        self.state = seed

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

seed = 42
rng = LCGRandom(seed)
print(rng.rand())  # 生成第一个随机数
print(rng.rand())  # 生成第二个随机数
~~~


## ProcessPoolExecutor(多进程) VS ThreadPoolExecutor（多线程）

问题：CPU 密集型任务使用 ThreadPoolExecutor 还是很慢？
答：这种任务 可能不会从多线程中获得太多性能提升，因为python的GIL（全局解释器锁）会限制并行执行。对于CPU密集型任务，使用多进程（ProcessPoolExecutor）可能是更好的选择。

ThreadPoolExecutor
多线程最适合**I/O密集型**任务，如文件读写、网络请求等，因为线程可以在等待I/O操作完成时让出CPU给其他线程.

实际上前者比后者更慢！


## 在处理大量简单任务时，最好避免使用多线程

以为例：

~~~py
def thread_func(x: int) -> int:
    return x * 2

## 方法一 多线程执行 很慢
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(thread_func, i) for i in range(1 << 20)]
    print("+++++")
    res = 1
    for i in range(1<<20):
        res += i
        # 获取结果
    print(res)
    print("-----")
    results = [future.result() for future in futures]

## 方法二 单线程执行 很快
for i in range(1 << 20):
    thread_func(i)

~~~
方法一使用了多线程执行，而方法二是单线程执行。尽管直觉上我们可能认为多线程应该更快，但实际上由于几个因素，方法一可能会比方法二慢：

- 任务的性质： thread_func 函数非常简单，只是将输入的整数乘以2。这种类型的计算非常快，几乎不需要时间。因此，多线程的优势在这里并不明显，**而线程的创建和管理的开销却成为了主要的瓶颈**。
- 全局解释器锁（GIL）： 在CPython中，由于GIL的存在，即使是多线程，也无法在多核CPU上实现真正的并行执行。对于CPU密集型任务，多线程可能不会带来性能提升，反而会因为线程间的竞争和上下文切换导致性能下降。
- 线程管理开销： 在方法一中，您创建了 1 << 20（大约104万）个线程任务。线程池需要管理这些任务的调度，这本身就是一个非常耗时的过程。每个任务的执行时间可能远远小于线程池调度任务的时间。
- 结果收集： 在方法一中，您使用了 future.result() 来收集每个任务的结果。这个调用会阻塞，直到对应的任务完成。由于任务数量巨大，等待所有任务完成的时间可能非常长。
- 线程池大小： 如果没有指定 max_workers 参数，ThreadPoolExecutor 默认的线程池大小是CPU的核心数。如果核心数较少，那么同时执行的线程数也会很少，这意味着大量的任务会在等待队列中排队，从而导致整体执行时间增加。

相比之下，方法二是单线程执行，没有线程创建和管理的开销，也没有GIL的影响，因此对于这种简单的计算任务来说，它可能更快。

在处理大量简单任务时，通常最好避免使用多线程，因为线程的开销可能会超过它们带来的性能提升。


## 多线程中的 Future 是个什么样的概念？

它代表了一个异步执行的操作的结果，这个结果可能在将来的某个时间点才可用。Future 之所以被称为这个名字，是因为它代表了一个未来可能会发生的事件的结果。

它提供了一种机制来访问异步操作的结果，而无需阻塞**当前线程**等待结果的完成。你可以提交多个任务，然后继续执行其他代码，之后再回来检查这些任务的结果。如下列：

~~~py
# 一个简单的任务函数
def task(n):
    return n * n

# 创建线程池
with ThreadPoolExecutor(max_workers=4) as executor:
    # 提交任务并获取Future对象
    future = executor.submit(task, 5)

    # 做一些其他的事情，而任务在后台执行
    # ...
    # 不阻塞地检查任务是否完成
    while not future.done():
        print("Task is still running...")
        time.sleep(0.5)  # 每0.5秒检查一次

    # 获取结果，因为我们已经知道任务完成了，所以不会阻塞
    try:
        result = future.result()
        print(f"The result is: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")

~~~

- Future 对象允许你提交一个异步操作，然后继续执行其他任务。主线程不会等待异步操作完成，而是继续执行其他工作。当异步操作完成时，Future 对象会更新其状态，并存储操作的结果或异常。
- 使用 Future.result() 方法来获取异步操作的结果。如果异步操作尚未完成，result() 方法将阻塞当前线程，直到结果可用。为了避免阻塞，你可以使用 Future.done() 方法检查异步操作是否已完成，然后再调用 result()。
- 异常处理: 如果异步操作抛出异常，你可以使用 Future.exception() 方法来获取异常信息。

## submit 方式 vs map方式

- submit 方法的好处是你可以对不同的任务使用不同的函数和参数，
- map 方法则适用于对同一个函数应用到一个参数序列的情况。

此外，submit 方法允许你在任务提交后立即继续执行代码，而不需要等待所有任务完成。这可以让你在等待结果的同时执行其他逻辑


## 向量化

常用 torch.Tensor 操作：`a_reshaped.unsqueeze(1).expand(-1, ch).reshape(-1)`


## 向量化操作一个 torch.Tensor

~~~py
flags: bool = True
def set_single_value(p: int, res: torch.Tensor):
    value: float = (1.0 / 13) * (1 << (p % 3))
    if flags & 1:
        value *= 7.0
    res[p] = value

n_elements = 1 << 20
tar: torch.Tensor = torch.empty(n_elements)
for i in range(n_elements):
    set_single_value(i, tar, )
~~~

通过 pytorch 的索引张量实现先量化，如下：

~~~py

flags: bool = True
n_elements = 1 << 20
tar: torch.Tensor = torch.empty(n_elements)

# 创建一个与tar同样大小的索引张量
indices = torch.arange(n_elements)

# 向量化地 计算所有值，value也是个tensor
values = (1.0 / 13) * (1 << (indices % 3))
# 向量化地 根据 flags 的值来决定是否需要将所有的值乘以7.0
if flags & 1:
    values *= 7.0

# 向量化地 将计算好的值赋给tar
tar = values
~~~

当需要计算的数量是 1<<30 时，循环耗时33min，向量化耗时9sec，向量化的快 已经是共识了

使用这个索引张量来计算所有的值，这是通过对整个张量执行操作来完成的，而不是逐个元素地执行。这种向量化的方法比逐个元素地计算要快得多，因为它减少了Python循环的开销，并且充分利用了底层优化的数学运算。


## pythonic 化

以下是一些编写Pythonic代码的特点：

- 遵循PEP 8风格指南：PEP 8是Python的官方风格指南，它提供了关于代码格式化的建议，包括缩进、行长度、变量命名、空白处理等。

- 使用 Python 的**内置功能**：Python提供了许多内置函数和数据类型，如**enumerate、range、zip、列表推导式、生成器表达式**等，它们可以使代码更简洁、更易读。

- 编写可读性强的代码：清晰的代码比晦涩难懂的代码更受欢迎。这包括使用有意义的变量名、添加注释和文档字符串（docstrings）。

- 避免过度工程：简单通常比复杂好。**不要使用复杂的设计模式或架构**，除非它们真的需要。

- 利用异常处理：使用**try和except块来处理潜在的错误情况**，而不是过多地使用条件语句来检查错误。

- 遵循EAFP原则：**EAFP**（Easier to Ask for Forgiveness than Permission）意味着你应该直接编写尝试做某事的代码，然后处理可能出现的问题，*而不是事先检查所有的前提条件*。

- 使用**函数式编程工具**：Python支持函数式编程的元素，如 map、filter、functools.reduce 等，它们可以帮助你编写更简洁的代码。

- **避免全局变量**：尽量减少全局变量的使用，因为它们可以使代码难以理解和维护。

- **模块化**和重用代码：将代码分解成模块和函数，以便重用和测试。

- 遵循DRY原则：**DRY**（Don't Repeat Yourself）原则鼓励你避免重复代码，通过抽象和函数封装来重用代码。

- 使用上下文管理器：**使用 with 语句来管理资源**，如文件操作，这样可以确保资源被正确地清理。

- 编写自包含的函数：**函数应该尽可能独立**，具有明确的输入和输出，这样它们更容易被理解和测试。
    自包含的函数是函数式编程范式的一个重要概念，它鼓励使用不可变数据和纯函数来构建程序，这样可以提高程序的可读性、可维护性和可测试性。在实际编程中，虽然不是所有的函数都能或应该是自包含的，但是追求函数的自包含性是一个好的实践。
    纯函数：它只依赖于其输入参数，并且只通过其返回值来提供输出

- 利用对象和类：当**合适**时，使用面向对象编程的特性，如继承、封装和多态性，但也不要过度使用。

编写 Pythonic 代码不仅仅是遵循规则，更多的是一种编程哲学，它鼓励你写出既符合 Python 语言特性又易于其他 Python 开发者理解的代码。

两种编程范式：

- 命令式编程 更注重过程，需要详细地描述每一步操作。它通常效率更高，但代码可能更冗长、更难理解。
- 声明式编程 更注重结果，只需要描述想要的结果，而不用关心实现细节。它通常更简洁、更易读，但可能效率略低。
- 声明式编程 中的一个特殊形式：函数式编程，它将计算视为数学函数的求值，更加强调纯函数和不可变性。


## 使代码 pythonic

~~~py
for i in range(a):
    for j in range(b):
        for z in range(c):
            idx = i * j + z
            # other code
# ==> 如果 other code 不依赖于 i 、j 和 z 之间的嵌套关系，可以考虑使用 `itertools.product` 来生成所有可能的组合，这样可以使代码更简洁

import itertools
for i, j, z in itertools.product(range(a), range(b), range(c)):
    identifier = i * j + z
~~~


## 生成器表达式  【一个迭代器】

生成器表达式（Generator Expression）是Python中的一种语法结构，它类似于列表推导式（List Comprehension），但生成器表达式返回的是一个生成器对象。**生成器对象是一个迭代器，它在迭代时按需计算和产生值，而不是一次性生成所有的值并存储在内存中**。这使得生成器表达式非常适合处理大数据集，因为它们**可以减少内存的使用**。

(expression for item in iterable if condition)

例子：

~~~py
# 假设我们有一个大的数字列表，我们想要找出所有偶数的平方
numbers = range(1 << 30)

# 定义了一个生成器，该生成器迭代 numbers 中的每个数字 x。如果 x 是偶数 (x % 2 == 0)，则计算 x ** 2 并将其作为生成器的下一个值“yield”出来。
# 它只在循环中需要时才计算下一个平方偶数，而不是预先计算所有平方偶数并将其存储在内存中。这使得它能够处理非常大的数据集，而不会导致内存溢出。
squared_evens = (x ** 2 for x in numbers if x % 2 == 0)

for i, num in enumerate(squared_evens):
    if i >= 10:  # 只打印前10个
        break
    print(num)
~~~

注意：
- `enumerate` 函数接受一个可迭代对象（这里是 squared_evens 生成器）作为输入，并返回一个迭代器。
- 在 `for` 循环遍历 `enumerate(squared_evens)` 返回的迭代器时，Python 会隐式地调用 `next()` 方法。
- 迭代器每次迭代都返回一个包含两个元素的元组：`(i, num)`，其中 `i` 是当前元素的索引，`num` 是当前元素的值。

如果是 List comprehension，创建后会把所有对象放入内存，耗时耗空间：

~~~py
a = range(1 << 30)
print(len(a))
# 快
sr1: Generator[int, None, None] = (x ** 2 for x in a if x % 2 == 0) # 见注释
print(type(sr1))

# 很很慢
sr2 = [x ** 2 for x in a if x % 2 == 0]
print(type(sr2))
~~~

Generator 类型有三个参数：

第一个参数 int 表示生成器产生的元素的类型。
第二个参数 None 表示生成器的 `send()` 方法接受的参数类型，因为在这个例子中我们不使用 `send()` ，所以用 None。
第三个参数 None 表示生成器的返回类型，当生成器结束时返回的值的类型。在大多数情况下，生成器不会返回任何值，因此也用 None。


## yield 关键字  【生成器函数】

yield 关键字用于创建生成器函数。生成器函数是一种特殊的函数，它可以返回一个迭代器，而不是一次性返回所有结果。这使得生成器函数非常适合处理大型数据集或无限序列，因为它可以按需生成值，而不是将所有值都加载到内存中。

有 yield 的函数被称为**生成器函数**。它们并不直接返回结果，而是返回一个迭代器对象。**每次调用 `next()` 方法（或在循环中隐式调用）时，生成器函数会从上次中断的地方继续执行，直到遇到下一个 `yield` 语句，并将 `yield` 后面的值作为返回值**。 如果生成器函数执行完毕，则抛出 `StopIteration` 异常。***

应用实例，无限序列：

~~~py
def fibonacci():
    a, b = 0, 1
    while True:
        yield (a, b)  # 每次调用next(), 执行到yield时，a, b的值会返回，然后暂停，等待下一次next()的调用 
        a, b = b, a + b

# 使用示例：
fib_gen = fibonacci()  # 生成器对象
for i in range(10):
    print(next(fib_gen))   # 遇到next函数才会实际执行一次
~~~

每次调用 next(), 执行到 yield 时，会返回 `(a, b)`，然后暂停，等待下一次next()的调用。 ***

## 函数可变参数

~~~py
def my_function(*args: Any):
    for arg in args:
        print(arg)

# 调用函数 时给出运行时的实参
my_function(1, 2, 3, 4)
~~~

~~~py
def my_function(**kwargs: Any):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 调用函数 时给出运行时的实参
my_function(a=1, b=2, c=3)
~~~

~~~py
def my_function(*args: Any, **kwargs: Any):
    for arg in args:
        print(arg)
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 调用函数
my_function(1, 2, a=3, b=4)
~~~


## 可变参数 **kwargs  【字典类型】

这里关键是 “**”，它是一个运算符，其后的 `kwargs` 只是一个约定俗称的关键字，它可以是任何写法。

~~~py
def my_function(**kwargs: Any):
    print(type(kwargs))  # 输出 <class 'dict'>
    for key, value in kwargs.items(): # kwargs 表示任意数量 参数字典
        print(f"{key}: {value}, type: {type(value)}")

my_function(name="Alice", age=30, city="New York")
~~~

## ‘is’ VS ‘==’

is 用于比较两个对象的身份（它们是否是同一个对象，是否是同一个内存地址），而 == 用于比较两个对象的值。
对于字符串和其他基本数据类型的比较，应该使用 ==。使用 is 可能会导致意外的行为，因为它比较的是对象的内存地址

## tuple 有啥不知道的？

- 它是一个**不可变**的序列，这意味着一旦创建，你就不能修改它的元素。
- 由于 tuple 是不可变的，Python 可以对其进行一些优化。
- 在多线程环境中，不可变对象是线程安全的，因为它们的状态不会改变。

~~~py
single_element_tuple = (1,)  # 注意逗号
multiple_elements_tuple = (1, 2, 3)
tuple_without_parentheses = 1, 2, 3 # 被称为元组打包（tuple packing）
tuple_from_list = tuple([1, 2, 3])  # 使用tuple的构造函数从其他可迭代对象中创建元组
~~~

## 给表达式中的 lambda 函数 标注类型

Agument type is partially unknown 如何修改？

~~~py
outer: int = list(itertools.accumulate(shape[:axis], lambda x, y: x * y))[-1]

# 改为
mul_func: Callable[[int, int], int] = lambda x, y: x * y
outer: int = list(itertools.accumulate(shape[:axis], mul_func))[-1]
~~~

## 创建一个可调用对象

`me_func: Callable[[int], int] = XXX`

XXX中实际做的是执行 input * input 返回 output，如何将这个 XXX 写成一个可调用对象赋值给 me_func？

~~~py
from typing import Callable

# 使用lambda表达式创建一个可调用对象
me_func: Callable[[int], int] = lambda x: x * x

# 测试me_func
output = me_func(5)
print(output)  # 应该打印25
# 在这个例子中，lambda x: x * x是一个匿名函数，它接受一个参数x并返回x的平方。

# 使用普通函数定义
# 定义一个函数
def square(x: int) -> int:
    return x * x

# 将函数赋值给me_func
me_func: Callable[[int], int] = square

# 测试me_func
output = me_func(5)
print(output)  # 应该打印25
~~~

## lambda 和一般函数的区别

Lambda 函数由于其简洁性（函数体只能有一个表达式），非常适合用于简单的、一次性使用的场景，尤其是在需要将函数作为参数传递给其他函数时（比如map()、filter()和reduce()等高阶函数）。而普通函数则更适合复杂的逻辑和长期复用的情况。


## ThreadPoolExecutor() 需要抛出异常

由于 `executor.map` 返回的是一个迭代器，你必须迭代它以确保所有的任务都被执行并且所有的异常都被处理。如果你不迭代结果，那么即使任务中发生了异常，你也不会看到它们。在多线程执行的函数中，如何有任何错误，那么这个函数会停止执行，并且**不会返回任何错误信息**，这就很迷惑。

~~~py

def parrallel():
    # ... (省略其他代码)

    def fill_chunk(idx_chunk):
        # ... (省略其他代码)
        # 引入一个错误，例如除以零
        raise ValueError("An error occurred in fill_chunk")

    # Create ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Parallel loop over chunks
        futures = executor.map(fill_chunk, range(n_chunks))

        # 迭代futures以确保任何抛出的异常都被捕获
        for future in futures:
            try:
                # 这里不需要做任何事情，因为我们只是想捕获异常
                pass
            except Exception as e:
                # 打印或处理异常
                print(f"An exception occurred: {e}")

# 调用函数
parrallel()
~~~

~~~py
with ThreadPoolExecutor() as executor:
    # 提交多个任务
    futures = [executor.submit(me_func, i, nums) for i in range(10)]

    # 等待每个任务完成并处理可能的异常
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            print(f"Task returned: {result}")
        except ValueError as e:
            print(f"A ValueError occurred: {e}")
        except Exception as e:
            print(f"An exception occurred: {e}")
~~~


## f-string  py3.6+

新的字符串格式化机制，允许你在字符串中直接嵌入表达式。这些表达式会在运行时求值，并将结果插入到字符串中。
~~~py
cmake_arg = [
    f"-DPYTHON_EXECUTABLE={sys.exec}",
    "-DCMAKE_BUILD_TYPE=Debug",
]
~~~
当前 Python 解释器的路径）将被插入到字符串中，构成了 cmake 命令的一个参数。这样，cmake 就会使用正确的 Python 解释器来构建项目。


## 装饰器

装饰器让被装饰的函数有其他功能。比如下面，让函数 adder 除了计算之外，打印出日志，而不再函数本身添加打印代码。使得不需要log时，不添加装饰器就好了

~~~py
from functools import wraps

def logit(func):
    @wraps(func)  # 保留 被装饰函数 的metadata
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return with_logging

@logit
def adder(x):
   return x + x


result = adder(4)
# Output: adder_function was called
~~~

添加`@logit` 等价于 `adder = logic(adder)`，可以理解为**我接受一个函数，让这个函数有了其他功能，返回添加了功能的这个函数。**

`@wraps()` 主要用于在使用装饰器包装函数时，保留被装饰函数的元数据（metadata）。 如果没有 `@wraps`，装饰器会改变被装饰函数的一些属性，例如 `__name__`、`__doc__`（文档字符串）和 `__annotations__`（类型提示）。 这会影响到函数的自我描述和调试。

## 装饰器本质上

本质上装饰器是在装饰器中调用了目标函数，同时做了些其他操作，返回一个包着目标函数的 wrapper。理解下面例子，你就理解了装饰器的本质了

~~~py
def decorator(a_func):
    @wraps(a_func)
    def wrapTheFunction():
        print("AA")
        a_func()
        print("BB")
    return wrapTheFunction

def target_func():
    print("这里是目标函数本来的功能")

target_func()
#outputs: "这里是目标函数本来的功能"

target_func = decorator(target_func)
#now target_func is wrapped by wrapTheFunction()

target_func()
#outputs: AA
#         这里是目标函数本来的功能
#         BB
~~~

python 有个特点：**一切皆对象**，具体表现在：

- 函数 func 对象，只要没有写 func() 就可以到处传递。一旦加了括号，就表示执行调用。
- 函数可以返回一个函数
- 函数内部可以定义函数


## 子进程以及同步

~~~py
for key_ in all_logs:
    url = key_
    xlsx_name = all_logs[url]
    command = ["wget", "-O", "LOG", url]
    subp = subprocess.Popen(command)
    subp.wait()   # 子进程同步
    to_excel("./LOG", xlsx_name + ".xlsx")
~~~


## 全局符号表   【运行时数据结构】

~~~py
def add_func():
    return 1 + 2
def max_func():
    return max([1, 2, 3])
def min_func():
    return min([1, 2, 3])

op = ['add', 'max', 'min']

for i in op:
    result = globals()[f"{i}_func"]()
    print(f"Result of {i}: {result}")
~~~

`globals()` 函数返回一个表示当前模块中全局符号表，它是一个字典，包含了当前模块中所有全局变量和函数的名称和对应位置。主要用于用于在运行时动态访问和修改全局变量和函数。在上述代码中，`globals()[f"{i}_func"]` 的作用是根据字符串 i 拼接出对应函数名，然后通过全局符号表找到该函数并执行它，

代接上述，码段的全局符号：

~~~py
for name, value in globals():
    print(name, ": ", value)  # 返回 RuntimeError: dictionary changed size during iteration
## 原因是：循环 for name, value in globals(): 迭代遍历 globals() 字典。
## 然而，在循环内部，语句 print(name, ": ", value) 会隐式地修改该字典。
## 这是因为 globals() 是一个动态字典，在迭代过程中访问它可能会触发内部更新，从而导致 RuntimeError 错误。

## 解决办法：拷贝一份，然后再遍历
global_vars = dict(globals())
for name, value in global_vars.items():
    print(name, ": ", value)
~~~

注意：这种直接访问 global 的方式很不优雅，不是最佳实践。***

KAQ：python 中的 globals() 返回一个全局符号表，动态链接库中也存在符号表，这两者是什么关系？
答：后者作用域是整个动态库，场景是在程序运行时解析和链接库中的符号。


# 面向对象
## res = self.func(src, **self.args) 解释

`**` 语法是 Python 中的参数解包（**unpacking**）操作符，它将字典的键值对转换为多个关键字参数。self.args 应该是一个字典，其中的键是参数名称，值是对应的参数值。

## __init__.py
`__init__.py` 文件用于标识一个目录是一个Python包。它可以为空文件，也可以包含初始化包时需要运行的Python代码。这个文件可以用来初始化包的变量、导入模块、设置路径等操作。

## super().__init__(aa)
`super()` 函数是用来调用父类（超类）的一个方法。`super().__init__(aa)` 这行代码的意思是调用当前类的父类的 `__init__` 方法，并传递参数 aa 给它


## Callable & callable()
Callable 是 typing 模块中的一个**泛型类型**，用于**类型注解**，表示一个对象是可调用的，并且可以指定调用时的参数类型和返回值类型。例子：

~~~py
from typing import Callable

def execute(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

def add(x: int, y: int) -> int:
    return x + y

result = execute(add, 5, 7)
print(result)  # 输出: 12
~~~

`callable()` 用于表示对象是否是“可调用的”。可调用对象的例子包括：

- 普通函数
- 类实例方法
- 类静态方法和类方法
- 生成器函数  (含有 yield 的函数，与 next() 函数相关)
- 内置函数和方法
- Lambda 表达式
- 类对象（如果类定义了 `__call__` 方法）

可以使用内置的 `callable()` 函数来检查一个对象是否是可调用的. 例子：

~~~py
def my_function():
    return "Hello, World!"

print(callable(my_function))  # 输出: True

class MyClass:
    def __call__(self):
        return "Hello, World!"

my_instance = MyClass()
print(callable(my_instance))  # 输出: True

print(callable(42))  # 输出: False
~~~


## 单双下划线开头的函数区别

- 单下划线前缀 `_func()`：是一个受保护的成员，它是按照约定用来指示这个函数是*内部使用*的，不是公共API的一部分。这是一种提示给其他程序员，表明这个函数主要供模块内部或子类中使用，而*不是设计为模块外部使用的*。然而，这只是一个约定，并不会在Python解释器层面强制实施访问限制。

- 双下划线前缀 `__func()`：是一个私有成员，它在Python中触发了名称改写（name mangling）。名称改写是Python中的一种机制，用于避免子类意外重写基类的方法。当你在一个类中定义了一个以双下划线开头的方法时，Python解释器会将其名称改写为`_ClassName__func()`，其中ClassName是定义该方法的类的名称。这使得这个方法对于外部来说更难以访问，但技术上仍然是可能的。

这两种方法都不能提供真正的私有性，因为Python**没有强制的访问控制**。

## "xx is possibly unbound"

这通常意味着你尝试访问一个可能没有被赋值的变量。换句话说，代码中存在一条路径，在这条路径上变量在被引用之前没有明确的赋值操作，因此解释器或静态代码分析工具无法确定该变量在使用时是否已经绑定到了一个值。

~~~py
def my_function(flag):
    if flag:
        my_var = 10
    print(my_var)  # 如果flag为False，my_var在这里可能是未绑定的

my_function(False)  # 这将导致一个错误，因为my_var没有被赋值 UnboundLocalError: local variable 'xx' referenced before assignment
~~~

在 Python 中，函数内部的代码块（如 if、else、for、while 等）共享相同的作用域。 C++ 作用域是块级的。***

## 变量作用域

在Python中，`if，elif，else`块不会创建新的局部作用域. 如：

~~~py
flag = True
if flag:
    my_var = 10  # 在if块中定义变量
else:
    my_var = 20  # 在else块中定义变量

print(my_var)  # 无论flag的值如何，my_var都可以在这里访问
~~~

但函数、类、模块和其他一些结构会创建新的局部作用域. 在这些结构中定义的变量在外部是不可见的，除非它们被**声明为全局变量**或**非局部变量**（使用`global`或`nonlocal`关键字）.

变量的作用域（scope）是由变量定义的位置决定的。Python中的作用域规则遵循`LEGB`规则，即：

- L（Local）：局部作用域. 局部作用域是指在函数内部定义的变量。这些变量只能在函数内部访问。
- E（Enclosing）：外层函数的作用域. 外层函数作用域是指嵌套函数中的外层函数的局部作用域。在内层函数中可以访问外层函数的变量，但不能修改它们（除非使用nonlocal关键字）。
- G（Global）：全局作用域. 全局作用域是指在模块级别定义的变量。这些变量可以在整个模块内的任何地方访问。
- B（Built-in）：内置作用域. 内置作用域是指Python解释器内置的变量和函数。这些内置的名称在任何地方都可以访问。

~~~py
## L - Local（局部作用域）
def my_function():
    local_var = "I am a local variable"  # Local scope
    print(local_var)

my_function()
# print(local_var)  # 这会引发错误，因为local_var在函数外部不可见


## E - Enclosing（外层函数作用域）
def outer_function():
    enclosing_var = "I am an enclosing variable"  # Enclosing scope
    def inner_function():
        print(enclosing_var)  # 可以访问外层函数的变量
    inner_function()

outer_function()


## G - Global（全局作用域）
global_var = "I am a global variable"  # Global scope

def my_function():
    print(global_var)  # 可以访问全局变量

my_function()
print(global_var)  # 在函数外部也可以访问


## B - Built-in（内置作用域）
print(dir(__builtins__))  # 打印所有内置变量和函数的列表

def my_function():
    print(len([1, 2, 3]))  # 使用内置的len函数

my_function()
~~~



~~~py
def my_function(flag):
    if flag:
        my_var = 10
    print(my_var)  # 如果flag为False，my_var在这里可能是未绑定的

my_function(False)  # 这将导致一个错误，因为my_var没有被赋值 UnboundLocalError: local variable 'xx' referenced before assignment
~~~
