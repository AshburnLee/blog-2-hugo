+++
date = '2025-08-31T12:57:44+08:00'
draft = false
title = 'CICD'
tags = ["CICD"]
categories = ["tools"]
+++


回忆 Jenkin 是如何实现当 PR push 时自动触发执行的？需要找一个 jenkins 插件帮助我实现 PR 通过 关键字 触发 CI.

使用什么插件？尝试了 “GitHub Branch Source” 中的 “Multibranch Pipeline projects”，需要在目标 repo 的根目录中创建并编辑 JenkinsFile 文件并且在 github repo 中配置一个webhook，每当有新 PR 到这个 repo，扫描 repo 并触发新 branch 执行 JenkinsFile。每一个 branch 触发一个 job，而这个 job 是不能通过修改 job config 来参数化的，需要通过 property step 在 JenkinsFile 中将这个branch 的 job 参数化。如此就可以执行任何内容了。

# Pytest
## Pytest 测试框架

pytest 提供了许多装饰器，比如 `@pytest.mark.parametrize` 让你为一个测试用例提供多个输入（输出参数，不同参数之间会进行**笛卡尔积组合**）。减少了重复代码。

更多用法 看看 pytest 命令参数。结果有4中状态，xfailed 等。实例：

~~~py
import pytest

@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", ['x', 'y'])
def test_example(a, b):
    print(f"Testing with a={a} and b={b}")

# 输出将会是：
# Testing with a=1 and b=x
# Testing with a=1 and b=y
# Testing with a=2 and b=x
# Testing with a=2 and b=y
~~~

## 跳过某些 test cases

- 使用 `pytest.mark.skip` 标记 case。
- 使用 `--deselect-from-file` 接受一个文件，这个文件中的所有 cases 都会被跳过。
- 使用 `--ignore=test_xxx.py`

## 与 pytest 一同使用的插件

pytest-select

pytest-xdict

pytest-timeout

## Texture：给 pytest 命令一个用户自定义选项

如何给 pytest 命令一个用户自定义的选项比如 `--device cpu`？答：使用 `pytest_addoption` 钩子在 `conftest.py` 文件中添加这个选项。需要一个名为 `conftest.py` 文件，并且包含一下内容：

~~~py
import pytest

def pytest_addoption(parser):
    parser.addoption("--device", action="store", default='cuda')

@pytest.fixture
def device(request):
    return request.config.getoption("--device")
~~~

搜索其他 texture 及其用法。

## Cache 机制

pytest 提供了一个内置的缓存插件，称为 `pytest-cache`，它允许你存储和重用测试之间的数据。这个插件可以帮助提高测试的效率，特别是当测试涉及到耗时的设置步骤时。`pytest-cache` 插件通过 `.pytest_cache` 目录来存储缓存数据。这个目录通常位于项目的根目录中。


# Pre-commit [用于管理hook的框架]
## 是什么？

- pre-commit 是一个用于管理和运行多语言代码格式化、linting、类型检查等任务的工具。它允许你在提交代码之前自动运行这些检查，以确保代码符合项目的质量标准。
- pre-commit 可以在你提交代码到版本控制系统之前自动运行一系列的检查。这些检查可以包括代码格式化、语法检查、查找敏感数据等
- pre-commit 是一个用于管理 Git 钩子（hooks）的框架，通过配置文件`.pre-commit-config.yaml`来指定 hook 和传给 hook 的参数，以及在什么阶段运行 hook。并可以在本地自动修正。

当你安装了 pre-commit 工具，他会根据`.pre-commit-config.yaml`中的定义来管理 Git 钩子。

什么是 Hook？在 pre-commit 上下文中，钩子（hook）指的是在特定事件（如 Git 提交）发生时自动运行的脚本或工具。

## 获益

- 自动化：你可以在提交代码之前自动运行各种检查，无需手动触发。
- 配置简单：使用 `.pre-commit-config.yaml` 文件可以轻松配置要运行的任务和它们的顺序。
- 多语言支持：pre-commit 支持各种编程语言的工具，如 Python 的 flake8、mypy，JavaScript 的 eslint，Go 的 golint 等。
- 可移植性：由于配置是项目的一部分，因此可以在任何支持 pre-commit 的环境中轻松设置和运行。
- 易于集成：可以与 Git 钩子轻松集成，以在提交时自动运行。
- 灵活性：可以定义要在每个文件、文件匹配模式或所有文件上运行的任务。

包括检查破坏的符号链接、尾随空格、文件末尾的换行符、YAML 文件、TOML 文件、抽象语法树、大文件、合并冲突、脚本的 shebang 行、私钥和调试语句

## 用法

项目根目录创建文件：`.pre-commit-config.yaml`。


# 坑

`yml` 文件不能使用 `tab`，只能还是用 `space`。
