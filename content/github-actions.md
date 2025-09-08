+++
date = '2025-08-31T12:57:44+08:00'
draft = false
title = 'Github Actions'
tags = ["github actions","CICD"]
categories = ["tools"]
+++


## 使用 actions/cache

如果你的工作流程经常运行，使用缓存来存储依赖项（如 Bandit）可以减少安装时间。
比如：
~~~yml
      - name: Cache Python dependencies
        uses: actions/cache@v2
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-
~~~

## 我的 github workflow 工作流程经常运行，但是不一定是在同一台机器上，这种情况下actions/cache 如何cache

`actions/cache@v2` 是跨工作流程和跨运行器设计的，这意味着即使工作流程在不同的机器上运行，缓存仍然是有效的。**缓存是与仓库相关联的**，而不是与特定的运行器实例相关联。

这是 `actions/cache` 的一个重要特性。缓存不是存储在某一台特定的运行器或机器上，而是**存储在 GitHub 的云端基础设施中**。当你使用 `actions/cache` 创建缓存时，GitHub 会将缓存内容保存在其服务器上，这样无论后续的工作流程在哪里运行，都可以访问和恢复这些缓存。

## 坑

1. actions/checkout@v4 只会最少的 repo 内容给你，与手动 clone 不同，所以你需要提供一些其他参数来给出更多的 commit 等信息。**因为actions/checkout@v4 默认不会把历史commit 给你** 需要添加 `fetch-depth: 0`

2. github 不希望你访问除了当前repo之外的目录，所以**所有的操作都应该在这个repo的目录中**，比如你想clone例外一个repo，那么把它clone到当前repo的 external/new_repo

3. 由于GitHub Actions的安全限制，工作流中的步骤**不能直接传递**输出到触发下一次工作流的schedule触发器。因此，你需要考虑其他方法来存储和检索PR编号，例如使用仓库的Secrets、工作流的**工件(artifacts)**或者外部存储服务。

4. `${{ env.TARGET_PRID }}` 这种方式的变量是 yml 变量，不应该在 shell 命令中使用（即不应该使用在 `run: |`）,但功能上也可以在shell 中这样访问。

5. `echo "TARGET_PRID=$(<file_downloaded/PRID.txt)" >> $GITHUB_ENV` 这句话表示将变量 `TARGET_PRID` 放在环境变量中，并且在**相同的job**中的后续步骤中的 shell 总可以直接访问 `$TARGET_PRID`。而且 在非shell的地方可以通过 `{{ env.TARGET_PRID }}` 访问其值。在当前job中之后的步骤里，已经有值了，生效了。

6. `actions/upload-artifact@v4` 过程上传的文件或文件夹会被自动压缩成一个ZIP文件。这是GitHub Actions的工件存储服务的一部分，旨在优化存储和传输效率。当你使用 `actions/download-artifact@v4` 下载工件时，GitHub Actions 会自动为你解压这个ZIP文件。你不需要手动执行任何解压缩步骤。

7. `actions/checkout@v4` 你给什么 ref 值？如果 ref 为空，它将检出当前工作流触发的仓库，就是 `on:` 的时候你指定的那个分支

8. `defaults:run:` 会在之后步骤中的所有 `run:` 下生效，小心不要被覆盖

9. `PAT` 和 `GITHUB_TOKEN` 的不同

10. 在workflow中执行 `gh` 命令需要你的repo管理员提供 `GITHUB_TOKEN`

11. yml 中的根目录是其所在repo的根目录，`$GITHUB_WORKSPACE` 环境变量指向的是当前仓库被检出到的目录，而GitHub Actions出于安全考虑，通常不允许你访问或修改该目录之外的文件系统。

12. 使用官方action时，注意其行为可能与你认为都不同，比如 actions/checkout@v4，结果是其 commitlog 并不完整，需要提供其他参数。


## `defaults: run:`

~~~yml
defaults:
  run:
    shell: bash -noprofile --norc -eo pipefail -c "source /home/runner/setvars.sh &gt; /dev/null; source {0}"
~~~
defaults 关键字允许你为整个工作流中的所有 run 步骤设置默认的行为。在你提供的代码片段中，defaults 被用来自定义运行shell命令时使用的shell程序及其选项。

具体来说，这段代码做了以下几件事情：

- 设置默认shell：为所有的 run 步骤设置默认的shell为 bash。

- 设置shell选项：
  1. `-noprofile`：告诉 bash 在启动时不读取profile文件（如 `~/.bash_profile` 或 `~/.profile`），这些文件通常在登录时执行。
  2. `--norc`：告诉 bash 在启动时不读取 `~/.bashrc` 配置文件，这个文件通常在非登录shell会话中执行。
  3. `-eo pipefail`：设置两个选项，`-e` 使得如果任何命令返回非零退出状态，整个命令列表将立即退出；`-o pipefail` 选项会导致管道（pipe）中的任何命令失败都会使整个管道命令返回非零状态。
  4. `-c`：后面跟随的是要执行的命令字符串。

- 执行初始化命令：在执行任何 run 步骤中的脚本之前，先执行 `source /home/runner/setvars.sh > /dev/null` 命令。这个命令似乎是为了初始化Intel oneAPI环境变量，使得后续的步骤可以使用oneAPI工具链。输出被重定向到 /dev/null，意味着你不会在日志中看到任何输出。

- 执行用户脚本：`source {0}` 是一个占位符，它将被后续 run 步骤中的脚本内容替换。这意味着每个 run 步骤中的脚本都会在初始化oneAPI环境之后执行。

总的来说，这段代码为工作流中的所有 run 步骤设置了一个自定义的shell环境，其中包括了初始化Intel oneAPI环境的步骤，以确保所有的步骤都在正确配置的环境中运行。

注意：如果你在某个特定的 run 步骤中指定了 shell，那么它将覆盖在 defaults 中为 run 步骤设置的默认 shell 配置！！！


## needs 关键字

如果你在一个工作流中上传工件，在另一个工作流中下载，后者需要有一个`needs`关键字来指定它依赖于前者。


## `actions/checkout@v4`

~~~yml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    ref: llvm-target
    fetch-depth: 0
~~~

注意这里的ref的值！！！如果 ref 为空，它将检出当前工作流触发的仓库，就是 `on:` 的时候你指定的那个分支


## 设置步骤的输出

Deprecated:
~~~yml
steps:
  - id: get_pr_id
    run: echo "::set-output name=pr_id::$pr_id"

steps:
  - name: Use the PR ID
    run: echo "The PR ID is ${{ steps.get_pr_id.outputs.pr_id }}"
~~~

GitHub Actions 的某个版本开始，推荐使用新的环境文件语法来设置输出，因为旧的 `::` 注解命令可能会被弃用


## artifacts 新的覆盖久的

非也，在GitHub Actions中，每次workflow运行时上传的artifacts都会**与那次运行关联**，并且不会自动覆盖之前的artifacts。

`actions/download-artifact@v4` 默认会尝试从与当前运行关联的 workflow 中下载 artifact。如果当前运行没有生成 artifact，它会回退到最近的成功运行中查找同名的 artifact。或者你提供给一个 id。

~~~yml
- name: Download artifact from a specific workflow run
  uses: actions/download-artifact@v4
  with:
    name: my-artifact
    run_id: 12345678
~~~
如果没有给定` run_id`，默认会尝试下载最新的可用工件。【正是所需】


## `run_id`

github 会给每个新的**工作流运行**分配一个唯一的 `run_id`。

## 定义： workflow & jobs

一个工作流（workflow）由一个或多个作业（jobs）组成.

假设你有一个工作流文件 `.github/workflows/my-workflow.yml`，它包含两个作业：job1 和 job2。你可以在 job1 中设置一个输出，然后在 job2 中使用这个输出. 需要在job2 中设置： `needs: job1`

然而，如果你有另一个工作流文件 `.github/workflows/another-workflow.yml`，它不能直接访问 my-workflow.yml 中的作业输出。不同工作流之间的数据传递需要使用其他方法，例如工件（artifacts）、缓存（cache）或者将数据写入到仓库的文件中。


## 自动PR的作者是谁

当使用 GitHub Actions 自动创建一个 Pull Request (PR) 时，PR 的作者通常是 GitHub Actions bot，也就是使用的 `GITHUB_TOKEN` 对应的 GitHub 用户。要在工作流中识别这个作者，你可以直接检查**PR 的元数据**


## 一个 workflow 这一次执行中创建的文件只能在当前执行中访问，其他执行访问不了

是的

在GitHub Actions中，每次工作流执行时都是独立的。这意味着每次执行都会创建一个新的运行环境，这个环境是临时的，只在当前的工作流执行期间存在。当工作流执行完成后，所有的运行环境和其中的文件都会被清理掉。如果你在工作流的某次执行中创建了文件，并且希望这些文件在后续的执行中可用，你需要将**文件持久化**。GitHub Actions提供了两种主要的方法来持久化工作流中的数据：

1. 上传工件（Artifacts）： 使用 `actions/upload-artifact Action`
2. 缓存依赖项和文件： 使用 `actions/cache Action`

工件和缓存都有自己的限制和使用场景。工件主要用于在工作流之间共享数据，而缓存则用于加速工作流的执行，例如通过缓存依赖项来减少安装时间。


## Artifacts 被存放在哪里

上传的文件被存储在 GitHub Actions 的工件存储中。保留一段时间（默认是90天）。Artifacts 存储在 GitHub 的服务器上，但具体的物理位置并不公开。GitHub 为每个仓库提供了一定量的免费存储空间来存放工件和日志，这些数据存储在 GitHub 的数据中心中。


## 遇到的语法解释

1. uses 关键字来指定一个动作
2. with 关键字用于传递参数给动作
3. ${{ }} 是用来插入表达式的语法。这种语法允许你在工作流中动态地引用变量、环境变量、上下文对象和表达式的结果
4. env: 下的变量 scope 是整个job，与 $GITHUB_ENV 中的变量 scope 一样


## yml 调用 yml

一个 yml 可以通过 `uses: ./.github/workflows/a.yml` 来复用另一个 yml 中的 job。

所以如果 b.yml 使用了 a.yml ，那么 b.yml 的 inputs 是 b 的**调用者给的**，a 的 inputs 是 a 的**调用者给的**。

a.yml 中通过:

~~~yml
on:
  workflow_call:
    inputs:
      python_version:
        description: Python version
        type: string
        default: "3.9"
~~~

来表示这个值需要调用者提供，在调用者中通过：

~~~yml
uses: ./.github/workflows/a.yml
    with:
      device: ${{ inputs.python_version }}
~~~


## steps 之间传递信息

~~~yml
- name: Load conda cache
  id: conda-cache
  uses: ./.github/actions/load
  env:
    CACHE_NUMBER: 5
  with: ...
- name: Update conda env
  if: ${{ steps.conda-cache.outputs.status == 'miss' }}  # 这里
  run: ...
~~~



# 完整语法和功能见：

这里是workflow的语法：
https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions

这里是action的语法：
https://docs.github.com/en/actions/creating-actions/metadata-syntax-for-github-actions

