+++
date = '2025-08-31T12:57:45+08:00'
draft = false
title = 'Linux Shell'
tags = ["shell","linux"]
categories = ["tools"]
+++


## WSL-1 的网络

WSL1实际上是在 Windows 网络堆栈之上运行的，也就是说，WSL1中的应用程序可以直接访问 Windows 的网络配置和服务。

即，127.0.0.1 指向 Windows。在WSL1中，访问 127.0.0.1（本地回环地址）实际上是指向 Windows 主机自己。因为 WSL1 没有虚拟网络隔离，所有网络请求都在主机的网络环境下处理。

WSL2 就不同了，它是一个虚拟机，有自己的网络栈（由多层网络协议组成的层级结构，每一层负责实现不同的网络功能和任务。这些层叠加起来协同工作，实现数据从一台设备传输到另一台设备的全过程，如TCP/IP协议栈）。


## 内存占用排序

实时查看和排序： `top → Shift+M` 键入 'c' 隐藏/显示 完整命令

快速查看排序列表： `ps aux --sort=-rss | head -10`

在资源受限的 Jetson 系统上是资源检测的有力工具。


## cpptools-srv

cpptools语言服务器。该进程提供代码智能提示（IntelliSense）、代码补全、语法检查、调试支持等功能。负责解析你的C/C++项目源代码，提升编辑和调试体验。


## vscode ssh 链接服务 问题

VSCode 通过 SSH 连接目标服务器时，发现该服务器的主机密钥（host key）与本地保存的记录不一致。

查看host上的key：`C:\Users\name\.ssh\kown_hosts` 中的记录 

和

服务器上的key：`ssh-keyscan -t rsa 192.168.x.x` 应该一致。

不一致时，将 host 上的删除，重新链接。不能删除服务器上的。因为你是host，想要链接服务器，你改变服务器的 key，很荒唐。

`-t` 指定密钥算法类型（rsa、ecdsa、ed25519等），所以你要先看看是哪个算法。发现两者密钥不一致，应该先确认服务器端主机密钥的正确性和安全性，然后删除本地客户端的旧密钥记录，再重新连接接受新的服务器密钥。


## 服务器上的 key 【Secure Shell Host Key】

ssh host key 是服务器用来唯一标识自身的SSH密钥对中的公钥部分。

这个密钥是**服务器的身份证**，用来确保客户端连接的是正确的服务器。当你用 VSCode 连接服务器时，VSCode（通过它的SSH扩展）会自动从服务器**拉取**该服务器的host key（公钥），并在本地 **known_hosts 文件里缓存**起来。这个过程并不是生成密钥，而是“拿到”和“保存”服务器提前生成好的密钥，以便后续验证。

在大多数 Linux 服务器系统安装过程中，系统会自动调用 ssh-keygen 生成一组 SSH 主机密钥对（包括公钥和私钥），文件通常存放在 `/etc/ssh/` 目录下，如 `ssh_host_rsa_key` 和对应的 `.pub` 文件等。用于 永久性保存用于识别服务器身份。

如果**服务器重装**或**管理员手动重新生成主机密钥**，这些密钥才会改变，客户端才会出现密钥变化的警告。

SSH host key 基于3种算法的，rsa、ecdsa、ed25519，ed25519 是最新的，安全性更高。

在服务器上查看服务器上的key： `ssh-keyscan -t ed25519 localhost`


## 显示器

WSL `echo $DISPLAY` 输出 `127.0.0.1:0.0`，这表示你的程序正在尝试连接到连接到宿主机 (Windows) 上的 X11 显示服务器。


## 内存占用高，如何找到占用内存高的程序

  - `top`	实时进程监控	                  top → Shift+M
  - `htop` 交互式进程管理	                 htop → F6 排序
  - `ps aux --sort=-%mem`	静态内存排序	ps aux --sort=-%mem | head -10
  - `free -h`	查看内存总量	free -h

## ltrace

用于追踪程序运行时调用的**动态链接库函数**。它通过拦截程序与动态链接库之间的函数调用，显示这些函数的名称、参数和返回值，从而帮助开发者调试程序、分析性能瓶颈以及理解程序的行为。

你可以用于查看参数和返回值，查看调用顺序，调用耗时。

~~~sh
ltrace curl baidu.com            # 默认是主程序对动态库函数的调用
ltrace -e 特定函数 curl baidu.com   # 显示某一个特定函数的调用，哪些动态库调用了这个特定函数
ltrace -tt -T -e XXX curl baidu.com   # 显示调用时间信息
ltrace -e @libcrypto.so.3 curl baidu.com    # 显示指定库 libcrypto.so.3 对于其他动态库的调用
ltrace -c curl baidu.com    # 显示每个函数的调用耗时和次数
~~~

## 查看文件的信息 Meta data

`file FILE`
`stat FILE`

## 调度是什么

"调度"是指对多个任务、进程或线程的执行顺序进行管理和控制的过程。调度器是操作系统的一部分，负责根据一定的策略和算法，**决定哪些任务或进程应该在何时执行，以达到系统性能、资源利用、响应性和公平性等目标**。

## 调度开销

是指在多任务处理或多线程环境中，由于在不同的任务或线程之间切换所引起的开销和延迟。

当程序需要同时运行多个任务或线程时，操作系统或调度器需要决定何时执行哪个任务或线程，并在它们之间进行切换。这个切换过程会引入一些开销，包括上下文切换、保存现场、恢复现场等操作。

调度开销包括但不限于以下方面：

- 上下文切换（Context Switching）：将当前任务或线程的上下文（包括寄存器值、程序计数器等）保存起来，并加载下一个任务或线程的上下文。这需要一定的时间和计算资源。
- 寄存器保存和恢复（Register Saving/Restoring）：在切换任务和线程时，需要保存和恢复寄存器的值，以确保程序状态的正确性。
- 内核态与用户态之间的切换：在多任务处理系统中，任务或线程可能在用户态和内核态之间切换。这种切换也需要一定开销。
- 调度器开销：操作系统的调度器需要根据一定的调度算法进行任务和线程的选择和排队。这个过程本身会引入一定的开销。

调度开销通常是不可避免的，特别是在高度并发的系统中。然而，过多的调度开销可能会导致系统性能的下降。因此，在设计程序和系统时，需要考虑和优化调度开销，以提高系统的效率和响应性。


## cgroup 来限制进程的资源使用

（Control Group）是 Linux 内核提供的一种功能，用于对进程进行资源限制、优先级控制、资源统计和监控。cgroup 允许系统管理员和开发者对进程组进行精细的资源管理，从而提高系统的性能和稳定性。

~~~sh
sudo cgcreate -g memory:my_group  # <subsystem>:<group_name>
sudo cgset -r memory.limit_in_bytes=64G my_group  #
sudo cgexec -g memory:my_group your_command

lssubsys -a  # 查看当前系统上的 cgroup 层次结构，已加载的子系统
ll /sys/fs/cgroup/  # 列出所有的cgroup子系统
ll /sys/fs/cgroup/memory  # 将列出 memory 子系统中已有的 cgroup

sudo cgget -r memory.usage_in_bytes my_group # 查看资源使用情况
sudo cgdelete my_group  # 危险
~~~


## dmesg

命令用于显示内核环缓冲区的内容，其中包含了关于内核和与之交互的硬件设备的消息


## killed 是内核发出的信号

通过 `dmesg` 输出信息查看，

~~~sh
Memory cgroup out of memory: Killed process 58676 (benchdnn) total-vm:29863408kB, anon-rss:20925428kB, file-rss:51072kB, shmem-rss:0kB, UID:0 pgtables:42156kB oom_score_adj:0

Killed process 58676 (benchdnn)
total-vm:29863408kB,  # 表示该进程的总虚拟内存大小为 29863408 千字节（kB）。虚拟内存是进程可访问的全部地址空间，包括实际分配的物理内存和交换空间。
anon-rss:20925428kB,  # 表示该进程占用的匿名页面的常驻内存集（Resident Set Size，RSS）为 20925428 千字节（kB）。匿名页面是指不与文件关联的内存页，通常是进程的堆栈和堆中的数据。
file-rss:51072kB,     # 表示该进程占用的文件映射的 RSS 为 51072 千字节（kB）。文件映射是指通过 mmap 等方式映射到文件的内存区域。
shmem-rss:0kB,        # 表示该进程占用的共享内存（shmem）的 RSS 为 0 千字节。在这个情况下，该进程没有占用共享内存。
UID:0                 # 表示该进程的用户标识符（User ID）为 0，通常表示是超级用户（root）运行的进程。
pgtables:42156kB      # 表示该进程的页表占用的内存大小为 42156 千字节。页表是操作系统用于管理虚拟内存的数据结构。
oom_score_adj:0       # 表示 OOM Killer 终止进程时所考虑的进程的调整分数，这个值为 0。这个分数可以影响 OOM Killer 选择终止哪个进程的决策。
~~~


## Linux 内核（Linux Kernel）和 Linux 系统（Linux Operating System）是两个相关但不同的概念。

### 1. Linux 内核（Linux Kernel）:

内核是操作系统的核心部分，负责管理系统的底层**硬件资源**，提供对硬件的抽象和访问。

Linux 内核是由 Linus Torvalds 创造的，是开源的、免费的，且是一个 Unix-like 操作系统的核心组件。

内核的主要功能包括**进程管理**、**内存管理**、**文件系统管理**、**设备驱动程序**、**网络管理**等。它提供了一个*基本的操作系统框架*，但*本身并不包含用户界面或标准工具*。

### 2. Linux 系统（Linux Operating System）:

Linux 操作系统是基于 Linux 内核构建的完整操作系统，包括内核、系统库、用户界面、应用程序以及一系列系统工具。
一个完整的 Linux 系统通常由 Linux 内核、GNU 工具（例如 Bash、gcc、glibc）、图形用户界面（如 X Window System）、系统库、应用程序和其他组件组成。

Linux 操作系统可以运行在各种硬件平台上，并且因其开源、灵活、稳定的特性而广泛应用于服务器、嵌入式系统、个人计算机等领域。

简而言之，Linux 内核是 Linux 操作系统的核心组件，而 Linux 系统是由 Linux 内核及其他支持组件构建而成的完整操作系统。 Linux 内核提供基本的系统服务，而 Linux 操作系统提供了用户界面和运行用户应用程序所需的一切。


## linux 如何查看某一个进程从开始到结束的内存占用总量

### 1. `Valgrind` 工具套件

~~~sh
valgrind --tool=massif <你的程序>
ms_print massif.out.<PID>  #根据输出文件分析内存使用情况 可以看到内存使用的峰值
~~~

### 2. 使用 `/proc` 文件系统

Linux 的 `/proc` 文件系统提供了有关**正在运行**的进程的各种信息

~~~sh
/proc/<PID>/status  #文件包含有关进程当前状态的信息，包括它的虚拟内存（VMem）和物理内存（RSS）的使用情况。
/proc/<PID>/statm  #文件显示了进程的内存状态信息，包括实际内存使用情况。
~~~

## linux 用户，所有用户组，及其权限，信息查询

~~~sh
cat /etc/passwd
cat /etc/group
id
id username
~~~


## jemalloc

有的情况发现，不使用 jemalloc 没问题，使用就会 core dump。原因很可能是 jemalloc 绑我们检查出了访问越界


## fstab 文件

位于 `/etc/fstab`，它是一个静态配置文件，包含了系统启动时需要挂载的文件系统的信息。fstab 文件的每一行定义了一个文件系统的挂载点、类型和挂载选项。

在系统启动时，fstab 文件会被读取，并根据其中的配置挂载文件系统。你也可以使用 mount 命令手动挂载 fstab 文件中定义的文件系统：`sudo mount -a`. 这个命令会根据 fstab 文件中的配置挂载所有未挂载的文件系统。

## linux 中如何安全地删除软连接

`rm <symbolic_link_name>`  仅仅删除链接，只能使用这种方式。
`rm <symbolic_link_name>/`  尝试将符号链接解释为一个目录，并删除该目录。然而，符号链接本身不是目录，因此 rm 会报错。
`rm -r <symbolic_link_name>` 删除链接 & 和指向的文件，危险！！！ 不要用！！！

## 创建链接

`ln -P file.txt link`  hard link
`ln -s file.txt link`  soft link 软连接是一个特殊类型的文件，它包含指向另一个文件或目录的路径。软连接本质上是一个指针或快捷方式，指向目标文件或目录

## 软连接的一般使用场景

编译软件时指定版本号(/application/apache2.2.23)访问时希望去掉版本号  (/application/appache)，可以设置软链接到编译的路径。所有程序都访问软链接文件(/application/appache)，当软件升级高版本后，只需要删除文件重建到高版本路径的软链接即可(/application/appache)。例：

~~~cpp
drwxrwxr-x  9 gta gta 4096 Mar 27 08:34 2023.0.0/
drwxrwxr-x 10 gta gta 4096 May  9 00:34 2023.1.0/
drwxr-xr-x  9 gta gta 4096 May  9 00:32 2023.4.27/
lrwxrwxrwx  1 gta gta    9 May  9 00:41 latest -> 2023.4.27/
~~~

使用时 路径中的 latest 表示你指定的版本


## 当 wget 需要认证时

`wget --user user --password pass http://example.com/`


## linux 命令中文件夹 mydir & mydir/ 区别

在大多数情况下，mydir 和 mydir/ 可以互换使用，但在某些命令和特定情况下，它们的行为可能会有所不同。比如:

- `cp -r mydir destination`：将 mydir 目录及其内容复制到 destination 目录中，结果是 destination/mydir。
- `cp -r mydir/ destination`：将 mydir 目录中的内容复制到 destination 目录中，结果是 destination 中包含 mydir 目录的内容，而不是 mydir 目录本身。


## linux inode 表示什么

inode（索引节点）是一个**数据结构**，用于存储文件的元数据 (metadata)。每个**文件**和**目录**都有一个唯一的 `inode`，它包含了文件的属性和位置信息，但不包含文件名或目录名。文件名和目录名存储在目录条目中，并与 inode 关联。inode 包含了以下信息：

- 文件类型：文件是普通文件、目录、符号链接、设备文件等。
- 文件权限：文件的读、写、执行权限。
- 文件所有者：文件的所有者（用户）和所属组。
- 文件大小：文件的字节数。
- 时间戳：文件的创建时间、修改时间和访问时间。
- 链接计数：指向该 inode 的硬链接数量。
- 数据块指针：指向文件数据块的指针，存储文件内容的实际位置。

inode 不包含文件名或目录名。这些信息存储在目录条目中，并与 inode 关联。

inode 在文件系统中非常重要，主要体现在以下几个方面：

- 文件系统结构：inode 是**linux文件系统**的基本组成部分，帮助组织和管理文件。
- 文件操作效率：通过 inode，文件系统可以快速访问**文件的元数据**和内容，提高文件操作的效率。
- 硬链接实现：inode 支持硬链接，允许多个文件名指向同一个文件内容，实现文件共享。**硬链接与源文件有相同的 inode，软连接不是**


## linux 设置用户组

- 简化权限管理：可以将权限赋予用户组，而**无需为每个用户单独**设置权限。

- 安全隔离：通过设置组权限控制不同用户组对资源的访问，提升系统安全。

- 协作需求：同一个组的用户可以共享访问某些文件或目录，**方便团队协作**。

- 灵活性高：用户可以属于多个组，获得**多种权限组合**。


## bash 行为：命令替换（Command Substitution）

在 Bash 中，$(xxx) 是一种命令替换（Command Substitution）的语法

## bash 行为：展开（expansion），Glob 模式

在 Bash 中，如果你想在给变量赋值时直接将通配符（如 *）展开成时匹配的文件列表，你不能直接在变量赋值这么做，**因为 Bash 不会在赋值时自动展开通配符**。

当变量值被双引号包围时，Bash不会执行文件名展开。

当变量值没有被双引号包围时，Bash会先执行文件名展开（如果可能的话），然后再进行单词分割等处理。

在实际使用中，为了避免意外的文件名展开或单词分割，通常建议使用双引号来包围变量值，除非你有明确的原因不这样做。

~~~sh
var1="*.txt"
echo $var1    # haha-3.0.0.txt    当变量值没有被双引号包围时，Bash会先执行文件名展开（如果可能的话），然后再进行单词分割等处理。
echo "$var1"  # *.txt             当变量值被双引号包围时，Bash不会执行文件名展开。

var2=*.txt
echo $var2    # haha-3.0.0.txt    当变量值没有被双引号包围时，Bash会先执行文件名展开（如果可能的话），然后再进行单词分割等处理。
echo "$var2"  # haha-3.0.0.txt    当变量值被双引号包围时，Bash不会执行文件名展开。
~~~

- bash 会在什么时候进行展开通配符: 在命令执行前自动进行的，确保命令接收的是实际的内容，而不是通配符字符串本身。

- **if 判断中不会展开通配符**，为什么？这是因为 if 语句的条件测试期望的是具体的值或表达式，而不是可能展开成多个值的通配符。条件测试 -f 是用来检查一个具体的文件路径是否存在并且是一个普通文件的，它并不理解或期望通配符的存在。

- `files=($PWD/*.wheel)`: 如果你想要在没有匹配的文件时让数组 files 为空，你可以在设置数组之前使用 `shopt -s nullglob` 来启用 `nullglob` 选项。这样，当没有文件匹配通配符时，数组将为空。


## command1 || command2 如何执行

在shell脚本中，`command1 || command2` 是一个条件执行语句。它的行为是：

1. 首先执行 command1。

2. 如果 command1 执行成功（即其返回值为0），那么 command2 将不会被执行，并且整个 `command1 || command2` 语句的返回值是 command1 的返回值（即0）。

3. 如果 command1 执行失败（即其返回值非0），那么 command2 将会被执行。此时，整个 `command1 || command2` 语句的返回值将是 command2 的返回值（如果 command2 执行成功，则为0；如果 command2 也执行失败，则为非0）。

这种结构常用于在 command1 失败时提供一个备选的或补救的 command2。例如，你可能在尝试安装某个软件包时使用这个结构，如果第一次尝试失败了，你可以尝试另一个安装源或方法。

另一个实际用途是 `command1 || true`, 即使 command1 失败，依然返回true

## 通配符 * 如何理解

`if [[ "$VAR1" = "$VAR2"* ]]; then`
一个场景：`[[ fd4fce2ee30d95ee09c30aead96bc2be994057a7 = \f\d\4\f\c\e\2* ]]`, 懂了

## shell 解释器

有多种不同的Shell解释器的存在是因为不同的开发者和组织对Shell的需求和偏好有所不同，并希望在不同的方面得到改进和扩展。

1. **Bash（Bourne Again Shell）**：Bash是Linux和Unix系统中最常用的Shell。它是Bourne Shell（sh）的升级版本，提供了许多附加功能和改进，支持交互式命令行和批处理脚本编程。

2. **sh（Bourne Shell）**：Bourne Shell是Unix系统中最早的Shell解释器之一。它相对较简单，功能有限，但在脚本编程方面仍然很常用。许多现代的Shell都是以Bourne Shell为基础进行扩展和改进的。

查看可用的解释器：`cat /etc/shells`

查看正在用的解释器: `echo $SHELL`

## shell 执行设置

sehll 脚本头部可以进行设置的，来

~~~sh
set -euo pipefail
command3 | command4 | command5
command6
# 这个脚本，当pipline有一个命令返回非零，command6 不会被执行到，因为有 -e
~~~

`set -euo pipefail` 这3个是编写健壮 shell 脚本的常见做法。(但不一定满足你的需求)

  - `-e`: 如果 command1 执行失败（返回非零值），那么整个脚本将立即停止执行，不会执行 command2. 没有 `-e` 时，command1返回非零时，command2仍会执行。
  - `-u`: 如果变量 VAR 没有被设置，那么尝试使用 $VAR 将导致脚本退出.有助于捕获那些可能由于拼写错误或逻辑错误而未被发现的变量。
  - `-o pipfail`: 不指明时，pipline 返回最后一个命令的返回状态；指明时，管道中任何一个命令的非零退出状态码。当你严格希望pipline中每一步都正确时，加上这个。

`set +o xtrace` 和 `set +x` 是等效的（关闭调试），同样地，`set -o xtrace` 和 `set -x` 也是等效的（打开调试）。这个调试信息有助于观察shell都执行了哪些东西。

## 通过 ./test.sh 执行时，说没有权限执行，但是 bash test.sh 可以正确实行

在 Unix-like 系统中，每个文件都有一组权限，决定了哪些用户可以读取、写入或执行该文件。对于脚本文件（如 .sh 文件），你需要确保该文件有执行权限。当你使用 `bash test.sh` 命令时，你实际上是在告诉 shell 直接使用 `bash` 程序来执行 `test.sh` 文件的内容，而不是尝试直接执行该文件。因此，文件的执行权限在这种情况下不是必需的。

但是，当你使用 `./test.sh` 命令时，shell 会尝试将 test.sh 作为可执行文件来执行。为了这样做，文件必须有以下两个条件：

1. 添加 `#!/bin/bash` 它指定了应该使用哪个解释器来执行文件的内容
2. 文件必须具有执行权限。你可以使用 `chmod +x xxx.sh` 命令来添加执行权限。

`sudo chsh -s /bin/bash 用户名` 改sh 为 bash, 重启 terminal 生效


## shell 脚本 只是 source 的用法，需要添加 shebang 吗

否

`Shebang（#!）`是脚本文件的第一行，用于指定解释器的路径。当脚本被直接运行（如 `./a.sh`）或通过 `bash a.sh` 这样的命令来执行时，需要指定解释器的路径。

但是，当使用 `source` 或 `.` 命令时，脚本的内容在当前 shell 环境中**一条一条被执行**（当前脚本是有shebang的），因此 该脚本`shebang` 行不会被用到。所以有无Shebang 无所谓。

## 坑：shell脚本中的命令执行所在路径是哪里

当一个不在根目录下的shell脚本被执行时，其中的命令执行时所在的路径（也称为当前工作目录或PWD）通常是**启动该脚本的用户的当前目录**，**而不是脚本文件所在的目录**。

但是，有一些方法可以改变这种行为：

### 1. 明确指定路径

在脚本中，你可以使用绝对路径或相对于脚本所在目录的相对路径来执行命令。

### 2. 使用`$0`获取脚本路径

在bash中，`$0`变量包含了脚本的名字（如果脚本通过直接路径执行）或脚本的名字（不包含路径，如果脚本在`PATH`中或通过`.`或`source`命令执行）。但是，你可以使用`$0`和`dirname`命令来获取脚本的目录，并使用`cd`命令切换到该目录。例如：

~~~sh
bash
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit
# 现在你的工作目录是脚本所在的目录
# 你可以在这里执行脚本中的其他命令
~~~

注意：上面的方法使用了`BASH_SOURCE[0]`而不是`$0`，因为`BASH_SOURCE[0]`即使在脚本被`source`或`.`执行时也能正确提供脚本的路径。


### 3. 在启动脚本时改变工作目录

你可以在调用脚本之前使用`cd`命令来改变工作目录到脚本所在的目录，然后执行脚本。但是，这会影响启动脚本的shell的当前工作目录，可能不是你想要的结果。

总之，默认情况下，脚本中的命令是在启动脚本的用户的当前工作目录中执行的，但你可以使用上述方法来改变这一点。

实例：给出base目录的绝对路径

~~~sh
if [ ! -v BASE ]; then
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "$BASE"
fi
export SCRIPTS_DIR=$(cd $(dirname "$0") && pwd)
~~~

实例：当前脚本的绝对路径

~~~sh
SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
~~~


## tr （Test Replacer）命令

将所有：替换为 换行符，让`PATH`结果好看 `echo $PATH | tr ':' '\n'`


## 当一个包 在环境中有多个路径时，我的程序使用的是哪个

~~~sh
whereis gh
gh: /usr/bin/gh /home/sdp/miniconda3/envs/junhui-py310/bin/gh
~~~

上述 gh 的位置有两个，那么我使用的是哪个？答：这取决于 PATH 变量中 两个路径的前后顺序，在**前的优先级高于在后的**。如果想调整优先级，需要手动修改`PATH`中路径的相对位置。

如果你想要改变这个行为，使得 `/usr/bin/gh` 被优先执行，你可以将 `/usr/bin` 移到PATH的开头：

~~~sh
export PATH=/usr/bin:$PATH  # insert to head
export PATH=$PATH:/usr/bin  # append from tail
~~~


## 列出 shell 中的用户自定义函数

~~~sh
declare -F # 列出所有用户自定义函数
type myfunc  # 打印出指定函数的定义, 确认是否是你期望的函数
~~~


## 语法

`TRITON_TEST_REPORTS="${TRITON_TEST_REPORTS:-false}"` 这行代码的意思是：

- 检查变量 `TRITON_TEST_REPORTS` 是否已经设置，并且它的值不是空的字符串。
- 如果 `TRITON_TEST_REPORTS` 已经设置且非空，那么 `TRITON_TEST_REPORTS` 的值保持不变。
- 如果 `TRITON_TEST_REPORTS` 没有设置或者它的值是空的字符串，那么将 `TRITON_TEST_REPORTS` 的值设置为 false。


## 执行 shell 脚本时，指定脚本参数

1. 添加文件执行时参数
2. 循环中的shift，**相当于是 i--**

~~~sh
QUIET=false
ARGS=
for arg in "$@"; do
  case $arg in
    -q|--quiet)
      QUIET=true
      shift
      ;;
    --help)
      echo "Example usage: ./my.sh [-q | --quiet]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done
~~~

shift 命令会将所有的位置参数向左移动一个位置，使得 `$2` 的值变为 `$1` 的值，`$3` 的值变为 `$2` 的值，依此类推。同时，`$#`（表示参数个数）的值也会减少1。这在处理多个参数的循环中特别有用，特别是当你想要逐个处理这些参数时。

循环中因为移动位置了，所以参数**始终会在第一个位置** `$1`，参数个数 `$#` 每次循环减一，不会进入死循环

~~~sh
while [ "$#" -gt 0 ]; do
    echo "The current param: $1"
    shift
done
~~~


## 使用 type 而非 which

`type xxx` 告诉你 xxx 是 shell 内置命令还是 shell 的 keyword，或者是外部命令。并且会给出路径，所以其功能包含了`which`, 输出更多的信息


## 前台/后台执行指令

如果你想在后台运行一个长时间运行的脚本 `long_running_script.sh`，你可以这样做：`./long_running_script.sh &` 当你这样做时，shell 会打印一个作业 ID 和一个进程 ID（PID），然后返回到命令提示符，允许你继续工作。

如果你想要检查后台作业的状态，你可以使用 `jobs` 命令

如果你想要**将一个已经在前台运行的命令移到后台**，你可以先使用 `Ctrl+Z` 组合键将其暂停，然后使用 `bg` 命令将其移到后台继续运行

要将后台作业带回到前台，你可以使用 `fg` 命令，后面可以跟作业 ID：`fg %1  # 将作业 ID 为 1 的作业带回到前台`

请注意，当你关闭终端或注销会话时，后台作业可能会被终止。如果你想要**在注销后继续运行后台作业**，你可以使用 `nohup` 命令: `nohup ./long_running_script.sh &`

`jobs` 命令只显示当前 **shell 会话中**启动的作业。如果你想查看**系统上所有进程**的信息，你应该使用 ps 或 top 命令


## 状态码

命令成功执行时的退出状态码为 0。可以通过 `$?` 变量**获取上一条命令的退出状态码**。通常，0 表示成功，非零值表示失败。当命令执行失败或出现错误时，会返回一个非零的退出状态码。不同的非零状态码可能表示不同的错误类型。


## capability 概念

在 Linux 中，**capability**（能力）是一种细粒度的权限控制机制，允许进程在不具备完整 root 权限的情况下执行特定的特权操作


## Basic

~~~sh
#!/bin/bash
# 1. pass parameters to a file
echo "Process ID: $$"
echo "File Name: $0"
echo "First Parameter : $1"
echo "Second Parameter : $2"
echo "All parameters 1: $@"
echo "All parameters 2: $*"
echo "Total: $#"

# 2. pass parameters to a function
echo "==========================="
function _func() {
    echo "script name: ${0}"
    echo "Language1: ${1}"
    echo "Language2: $2"
    echo "First Para : $1"
    echo "Second Para : $2"
    echo "All para 1: $@"
    echo "All para 2: $*"
    echo "Total para: $#"
    echo "ID: $$"
    for pa in $@; do
        echo "in for loop: $pa"
    done
}
# 3. 调用函数
_func c c++ cpp hpp

echo "==========================="
echo "return: $?" # 上一条命令的返回信号
echo "==========================="

# 4. 所有变量类型都是字符串
url=http://c.biancheng.net/shell/
echo $url
name='C语言中文网'
echo $name
# 5. 变量定义时最好都加上双引号“”，定义变量时加双引号是最常见的使用场景
author="严站长"
echo $author
echo ${author}
# 6. 推荐给所有变量加上花括号{ }，这是个良好的编程习惯

echo "I am good at ${author}Script"

# 7. 单引号VS 双引号
echo "==========================="
url="http://c.hh.net"
website1='C语言中文网: ${url}' # 原样输出
website2="C语言中文网: $url, ${url}" # 解析输出
echo $website1
echo $website2

# 8. 将命令的结果赋值给变量
# variable=`command`
# variable=$(command)  # 推荐
demo=$(python --version)
echo $demo


# 9. 全局变量. 在 Shell 中定义的变量，默认就是全局变量
# 所谓全局变量，就是指变量在当前的整个 Shell 进程中都有效
# 打开两个shell窗口 就是两个shell了（两个shell 进程，两个进程ID）
function _func(){
    local var_in_func=99 # 局部变量
    var_in_func2=89      # 全局变量
}
_func
#输出函数内部的变量
echo ${var_in_func}
echo ${var_in_func2}

# 10. 全局变量的作用范围是（且仅仅是）当前的 Shell 进程(其子进程都不可见)，而不是当前的 Shell 脚本文件

# 11. 在一个 Shell 进程中可以使用 source 命令执行多个 Shell 脚本文件，此时全局变量在这些脚本文件中都有效
# 12. `bash zz.sh` : 生成一个子进程，在其中执行 (内外ID不同)
# 13. `chmod +x zz.sh`, `./zz.sh` : 添加执行权限，后再子进程中执行

# 14. `source zz.sh` : 在当前shell进程中执行，全局变量在所有shell脚本中使用 (内外ID相同)
# source 是 Shell 内置命令的一种，它会读取脚本文件中的代码，并依次执行所有语句
# source 命令会强制执行脚本文件中的全部命令，而忽略脚本文件的权限
# 15. `. ./zz.sh`: 同 `source ./zz.sh`  (内外ID相同)
echo ${zzz}

echo "ID: $$"
# 10. 全局变量只在当前 Shell 进程中有效，对其它 Shell 进程和子进程都无效
# 11. 如果使用export命令将全局变量导出，那么它就在[所有的]子进程中也有效了，这称为“环境变量”
# 当 Shell 子进程产生时，它会继承父进程的环境变量为自己所用
# 环境变量只能向下传递而不能向上传递 "传子不传父"
# 16. 创建 Shell 子进程最简单的方式是运行 bash. exit 退出这一层shell

# 17. echo $-: 查看是否是交互式
echo $-

# 18. 接受键盘输入
echo "Please type you name: "
read name
echo "Hey ${name}"


# 建议 if 判断条件时:
# 20. 用 [[ ]] 来处理字符串或者文件
if [[ ${name} == "junhui" ]]; then
    echo "Hi my lord junhui"
else
    echo "Unkown object: ${name}, exiting..."
    exit 1
fi

# 21. 用 (()) 来处理整型数字，
echo "Please type you age: "
read age
if (( ${age} > 18 && ${age} < 110 )); then
    echo "You are brillaint enough to learn"
else
    echo "YOu are too old or too young, exiting..."
    exit 1
fi

# 19. test 指令,等价于 [ ]
# 20. 使用 test 指令时，建议用“” 包起变量，避免因为空变量而引发奇怪问题
if [[ -z "${name}" ]]; then
    echo "变量长度为0"
fi
if [[ -n "${name}" ]]; then
    echo "变量非空"
fi

# 文件判断
if [ -f "$0" ]; then
    echo "File zzz.sh is exist"
fi
if [ -r "$0" ]; then
    echo "This script have Read access"
fi
if [ -w "$0" ]; then
    echo "This script have Write access"
fi
if [ -x "$0" ]; then
    echo "This script have Execution access"
fi
# 22. 文件夹是否存在
if [ -d junhui ]; then
    echo "dir: junhui exists"
fi
# 22. 判断不存在 用取反即可
if [ ! -d junhui ]; then
    echo "dir junhui is not exist"
fi
# 23. 文件是否存在
if [ -e log.txt ]; then
    echo "file: log.txt exists"
fi

# 24. while loop

# 25. call another .sh file
_Hello () {
    echo "calling function from a .sh file"
}

$1
## bash ${PWD}/learn_shell.sh _Hello
## 上面一句文件调用执行自己，会一直递归下去

FUNC="_Hello2"
bash ${PWD}/funcs.sh ${FUNC}
## 注意：funcs.sh 中一定要有 $1 用来捕获你想调用的函数名

set # 列出当前shell中所有的 变量 & 函数（非环境变量）

if [[ -v NAME ]]; then # 语法，它用于检查变量是否已声明
~~~


## netstat -anp

~~~sh
$ netstat -anp|grep 7030

Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name
tcp        0      0 127.0.0.1:7030          0.0.0.0:*               LISTEN      -
tcp        0      0 127.0.0.1:7030          127.0.0.1:57104         ESTABLISHED -
tcp        0      0 127.0.0.1:37398         127.0.0.1:7030          TIME_WAIT   -
tcp6       0      0 ::1:7030                :::*                    LISTEN      -
tcp6       0      0 127.0.0.1:57104         127.0.0.1:7030          ESTABLISHED -
~~~

## ssh 客户端配置文件

~~~sh
Host 0-0
  HostName 127.0.0.1 # target-host
  Port 7890          # target-port
  User jenki
  ProxyJump jenki@xxx.me.com   # jump-host
~~~

proxyjump 是一个 SSH 跳板（也称为代理跳转）配置，它指定了一个中间主机，SSH 客户端会先连接到这个中间主机，然后再从中间主机跳转到目标主机。
这种配置通常用于访问通过防火墙或其他网络限制隔离的内部网络资源

## proxy & VPN

代理是局部流量转发，VPN是全局流量加密转发。代理配置相对简单，VPN则更安全和全面。


## ip 地址和端口有什么关系

IP地址定位网络中唯一的一台设备，就像街道地址确定具体**一栋楼的位置**；端口号定位设备上的具体服务或应用，就像楼里的房间号指向**特定的办公室**。两者结合形成套接字（Socket），确保网络数据能准确送达某台设备的指定程序。


## 蓝牙

~~~sh
sudo bluetoothctl
power on
scan on
# 等待几秒钟，扫描设备
devices # 列出扫描到的设备
# 链接指定设备

scan off
pair XX:XX:XX:XX:XX:XX  # 将 XX:XX:XX:XX:XX:XX 替换为你的设备 MAC 地址
# 可能需要输入配对码
connect XX:XX:XX:XX:XX:XX  # 将 XX:XX:XX:XX:XX:XX 替换为你的设备 MAC 地址
connected
quit
~~~

## airplane mode

nmcli 是 NetworkManager 的命令行接口。 可以使用它来禁用所有无线接口，模拟飞行模式的效果
~~~sh
sudo nmcli radio all off
sudo nmcli radio all on
~~~

## wifi

~~~sh
# windows 上查看已连接wifi 密码
netsh wlan show profile name="xxx" key=clear
# linux链接指定wifi
sudo nmcli device wifi connect 'xxx' password 'yyy' # 命令没问题，这个wifi有问题

# 列出所有已保存的网络连接
nmcli con show
# 删除连接
nmcli con delete uuid <UUID>
# 后重新连接
# 激活连接
nmcli con up ziroom801_5G  
#扫描可用wifi网络
nmcli dev wifi list  
~~~

Windows上查看wifi信号强弱：

~~~sh
netsh wlan show interfaces
~~~

## curl 命令

显示详细的请求和响应信息：

  - 包括curl 正在进行的详细步骤。
  - 显示所有的HTTP请求头（Request Headers）。
  - 显示所有的HTTP响应头（Response Headers）。
  - 显示SSL/TLS握手过程的详细信息（如果使用HTTPS）。
  - 显示连接过程的详细信息。

例如 `curl -v https://www.example.com`  返回的信息包括：

  - 以 * 开头的行：是 curl 提供的关于连接过程、SSL/TLS握手等信息的调试信息。
  - 以 > 开头的行：是 curl 发送给服务器的请求头。
  - 以 < 开头的行：是服务器返回的响应头。
  - 最后一部分：是服务器返回的HTML内容（如果请求的是网页）。


## 使用过的命令汇总

`sed` : 替换，查找
`grep` ：提取内容
`awk`:
`command || true` 的含义是什么 ： 保证前面的命令失败时整条命令不会失败
`2 >& 1` : 用于重定向错误到标准输出

`mkdir`
- "mkdir abs" 时，如果 "abs" 目录已经存在，该操作不会覆盖原有的目录。相反，它会**返回一个错误**并指示该目录已存在。
- "mkdir -p abs" 时，如果 "abs" 目录已经存在，该操作不会覆盖原有的目录。命令中的选项 "-p" 表示递归地创建目录，如果指定的路径中的目录已经存在，它会**忽略并跳过**创建该目录的步骤。

`import subprocess`：在Python脚本中执行其他脚本包括 shell

scp:

`scp junhui@192.168.0.171:/home/junhui/workspace/* ./`

`sudo visudo`: 编辑sudor文件
`import pexpect`：实现交互自动化

`mount | grep nfs`：查看挂载

`sudo useradd -m -s /bin/bash USER`: linux 创建用户
`$((expression))` ：允许你在Shell中进行算术运算

conda 创建无名虚拟环境：
- conda create --prefix ${conda_env} python=3.8 -y
- conda activate ${conda_env}

`strace clinfo`：查看进程 clinfo 中，所有的系统调用
`strings`: 该命令是一个用于从二进制文件、可执行文件、库文件等非文本文件中**提取可打印字符串**的工具。其主要作用是帮助用户识别和分析文件中的文本信息，例如调试程序、查找特定字符串或理解文件内容。

`command 1> /dev/null 2>&1`: 作用是将命令的标准输出和标准错误输出都重定向到`/dev/null`，这样命令的输出就会被丢弃而*不会显示在终端上*。
`eval`: 会将读取的内容作为shell 命令执行，而非作为字符串输出

`command1 && command2`：上一条命令成功执行后，才能执行后一条命令
`command1 ; command2`：默认情况，是不论前面的命令是否正确执行，后行的都会执行（set -e 关掉掉这个行为）

`| xargs`：将前面的结果作为后面的参数
`mktemp`： 安全地创建一个临时文件或目录

`find ./ -name "xxx"`:
`pip install --user xxx`: 给指定的用户安装指定包

`python3 -m pytest ...` : 将pytest当做脚本执行
