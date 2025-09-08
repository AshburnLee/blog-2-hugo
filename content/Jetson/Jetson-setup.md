+++
date = '2025-08-31T12:47:02+08:00'
draft = false
title = 'Jetson Setup'
tags = ["Jetson"]
categories = ["Jetson"]
+++


NVIDIA Orin 是一款专为**自动驾驶汽车**和**机器人**设计的高性能系统级芯片，包含新一代 Ampere GPU 架构和 Arm Hercules CPU 内核，以及深度学习加速器和计算机视觉加速器。

应用领域：Orin 芯片不仅适用于自动驾驶汽车，还广泛应用于机器人、工业边缘计算等领域。

支持C/C++, python, cuda, pytorch, ROS (Robot Operating System), JetPack SDK, DeepStream, VScode 

TensorRT 是 NVIDIA 开发的一个高性能深度学习推理 SDK。它不是完全开源的.

`Linux for Tegra (L4T) `是 NVIDIA 为其 Tegra 系列系统芯片 (SoC) 开发的嵌入式 Linux 发行版。它主要用于 NVIDIA Jetson 系列开发套件等嵌入式系统。L4T 提供了运行在 Tegra SoC 上的内核、驱动程序、库和工具，支持各种应用，包括机器人、人工智能、自动驾驶和媒体处理等。 它包含了 NVIDIA 专有的驱动程序，以充分利用 Tegra SoC 的硬件加速功能。不同的 L4T 版本支持不同的 Tegra 系列芯片和功能。 例如，较新的版本可能支持 Vulkan 和更新的 CUDA 版本。 开发者可以使用 L4T 来构建和部署各种嵌入式应用。

# 配置Jetson Orin Nano
## 记录命令

~~~sh
linux ip: <linux-ip>
uname: junhui

sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install -U jetson-stats
sudo jtop

sudo poweroff

systemctl status bluetooth
systemctl start bluetooth

iwconfig  # 看网速等网络状态
~~~


## install Jetpack

~~~sh
# 找不到 nvcc 编译器
sudo apt install nvidia-jetpack  # 8GB 内容
find / -name "nvcc" 2>/dev/null

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# 找到 sample code
find / -name "vectorAdd.cu" 2>/dev/null 
bash /usr/local/cuda-11.4/bin/cuda-install-samples-11.4.sh ./  #拷贝 samples 到指定路径

# 验证：(默认找不到头文件，只能指明-I)
nvcc $CUDA_HOME/samples/0_Simple/vectorAdd/vectorAdd.cu -I/usr/local/cuda-11.4/samples/common/inc/ -o vectorAdd

nvcc vectorAdd.cu -I/home/junhui/workspace/NVIDIA_CUDA-11.4_Samples/common/inc/ -o vectorAdd
nvcc deviceQuery.cpp -I/home/junhui/workspace/NVIDIA_CUDA-11.4_Samples/common/inc -o deviceQuery

# 检查
dpkg -l | grep nvidia  #这个命令会列出所有安装的与 "nvidia" 相关的软件包。
dpkg -l | grep nvidia 获取一个大致的组件列表，然后使用 apt-cache show 命令查看特定组件的详细信息
~~~


# 主机连接 Orin 最佳实践
假如没有外接显示器和鼠标键盘。

## 1. ssh 链接 Orin
当使用 USB-C 连接主机和 Orin 后，可以通过 Type-C 产生的虚拟网口连接，IP地址是固定的：`192.168.55.1`。通过这个地址 ssh 连接 Orin。

## 2. 配置 Orin wifi

### 1. 连接 wifi 

`sudo nmcli device wifi connect '<wifi-name>' password '<password>'` 有时候网络会出毛病，不是命令的问题。
### 2. 使用 NetworkManager: 

~~~sh
sudo apt update
sudo apt install network-manager
sudo service NetworkManager start
~~~

### 3. ~~将Orin 接入显示器。通过图形界面来连接，常规方式。不方便，需要接入显示器和鼠标键盘。~~

这时也可以通过上述得到的 wlan0 的 inet ssh连接 Orin。
~~~sh
# linux链接指定wifi
sudo nmcli device wifi connect '<wifi-name>' password '<password>'

#扫描可用wifi网络
nmcli dev wifi list  
# 列出所有已保存的网络连接
nmcli con show
# 删除连接
nmcli con delete uuid <UUID>

# 激活连接
nmcli con up ziroom801_5G  
~~~


## Proxy 问题

windows上链接了代理，如何让 Jetson 连接代理？

1. 在 Windows Clash 中开启“允许局域网链接” & “TUN模式”
2. 找到 windows IP 
3. 在 Jetson 中 输入 `export https_proxy=http://<windows_ip>:7890`
4. 通过 `wget` 验证
5. 对于 Docker，需要配置 Docker 的 http_proxy. 祥见下

KAQ问题：为什么 wget 可以通过设置的代理，但是 ping 不能？
答：因为两者的使用的协议不同。wget使用 http 或 https协议。而 ping 使用 ICMP 协议。


## Jetson Orin SUPER mode

如何在设备上切换 Power Mode：

~~~sh
sudo nvpmodel -q  # 产看当前mode
sudo nvpmodel -m <mode_id>
~~~

通过 nvpmodel 设置的电源模式在重启后仍然有效 。除非再次调用 nvpmodel，否则模式不会改变。

~~~sh
NV Power Mode: 10W 对应 mode=0
NV Power Mode: 25W 对应 mode=1
NV Power Mode: MAXN_SUPER 对应 mode=2
~~~


## 安装 Nsight Compute

通过重新烧录 Jetpack，ncu 成功安装。

## Nsight System：已安装，使用

1. `nsys profile -o vectorAdd ./vectorAdd` 生成文件 `vectorAdd2.nsys-rep` 然后可以通过 `nsys-ui vectorAdd.nsys-rep` 图形化分析结果。

2. `nsys-ui` 开启UI，找到可执行文件，Start 开始分析，结束后会生成一个 `.nsys-rep` 文件。此方法显示Qt版本问题。不成功。


## 更新 jetpack

需要安装 SDK manager 到 ubuntu 20.04 (vmware中的虚拟机), 然后烧录到 Jetson Orin 中。


## 安装 jetson-containers

https://github.com/dusty-nv/jetson-containers


## Docker 网络问题：

### 1. 通过 配置 Docker 代理：

~~~sh
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf

# http-proxy.conf 中添加以下内容：

[Service]
Environment="HTTP_PROXY=http://<windows-ip>:7890"
Environment="HTTPS_PROXY=http://<windows-ip>:7890"
Environment="NO_PROXY=localhost,127.0.0.1"
~~~

### 2. 然后需要在 Windows 的 Clash 中开启 “允许局域网连接” 。
### 3. 重启 daemon 和 docker： `sudo systemctl daemon-reload` , `sudo systemctl restart docker`
### 3. 开启 “TUN模式”，最后开启它，否则代理不起作用。***

如此 `Docker pull` 可以正确下载（只是不是 100% 稳定，已经很好了，证明了这条路是通的对的）


## 关于 Docker 其他要知道的

配置文件添加国内镜像


## 额外的：我在 config.yaml 中添加了 `bind-address: 0.0.0.0`

监听所有IP地址: 设置为 0.0.0.0 意味着 Clash 将监听来自所有网络接口的连接请求。这包括本地回环地址（127.0.0.1）以及任何其他网络接口（如局域网接口、Wi-Fi等）。这使得局域网内的其他设备也能够通过 Clash 的代理服务访问互联网。

配合 allow-lan 使用: 只有当 allow-lan: true 时，bind-address 的设置才会生效。这意味着，如果你希望其他局域网设备能够通过 Clash 进行代理访问，你必须同时启用这两个设置。


## apt 也不走环境代理

**`apt update` 不会使用环境代理**，创建 apt 的配置文件并添加内容如下 

~~~sh
sudo vim /etc/apt/apt.conf.d/proxy.conf
Acquire::http::Proxy "http://<windows-ip>:7890/";
Acquire::https::Proxy "http://<windows-ip>:7890/";
~~~

或者：

~~~sh
sudo apt update -o Acquire::http::Proxy="http://<windows-ip>:7890/" -o Acquire::https::Proxy="http://<windows-ip>:7890/"
~~~

**每个程序可能要配置自己的代理**，环境变量的作用范围是不同的，代理协议的支持 不同程序间是不同的，所以当发现某个程序没有使用eport的代理，那就需要单独配置了。或者使用全局代理工具。


## 快速查看 ip

当 windows 通过 ssh 连接到 linux 时，Windows 是客户端，Linux 是服务器。此时，在 Linux 上查看 env `SSH_CONNECTION`：

`SSH_CONNECTION=<windows-ip> 14019 <linux-ip> 22` 给你重要的 4 信息：

  - 客户端 IP 地址 (Client IP Address): 发起 SSH 连接的客户端机器的 IP 地址。 即你的 Windows 机器的 IP 地址。
  - 客户端端口号 (Client Port Number): 客户端机器用于发起 SSH 连接的端口号。 这是一个临时的、随机分配的端口。
  - 服务器 IP 地址 (Server IP Address): 接收 SSH 连接的 Linux 服务器的 IP 地址。
  - 服务器端口号 (Server Port Number): Linux 服务器上 SSH daemon 监听的端口号。 默认情况下，SSH 监听端口是 22。


# 总结 Jetson 连接 windows 代理

1. Windows 局域网IP，`<windows-ip>`
2. Jetson 同局域网联网
3. Jetson: `export http_proxy="http://<windows-ip>:7890"`
4. Clash 开启允许局域网连接
5. 配置 docker 文件：`sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf`，添加上述 http_proxy。
5. 重启 Daemon & docker
6. Clash 开启 TUN模式

要想灵活运用，需要知道上述设计的原理。 这样换另一个代理软件，也能正确使用。


## 远程链接 jetson

我有一个Linux设备作为服务器，同时有一个Windows设备作为客户端。在家里的相同 wifi 下我成功通过 SSH 从我的 Windows机器连接到Linux服务器。现在我需要把Linux服务器留在家里，在咖啡馆用windows 机器远程连接到家里的Linux服务器，我应该如何做？给出具体步骤。


## 安装开关

开发板背面有一个**12针接口**，其中的第1针为`PWR BTN`，第7、8针`DIS AUTO ON`，这两个针脚可以通过短接和接开关来实现软开关功能，达到不需要每次拔插电源就能开关机的目的。

- `PWR BTN`（电源按钮）和`GND`短接： 连接一个开关按钮，当按下时短接这两个针脚，可以用来控制开机或关机信号。
- `DIS AUTO ON`两针脚： 这两个针脚短接后可以禁止自动上电启动，使开发板插上电源不自动启动，配合`PWR BTN`来实现手动开关机。
- 具体连线方式是将`PWR BTN`接开关一侧，开关的另一侧接`GND`，同时将`DIS AUTO ON`短接(就是两个针脚连接在一起。)，或者某种组合可以达到手动控制上电开关的目的。

两个针脚**短路**，在电子电路中是指这两个针脚之间用导线、金属或开关闭合，使两点之间的电阻接近于零，也就是这两个针脚直接连通了，没有电阻阻挡电流流过。换句话说，就是这两个针脚被“短接”了。短路使得电压在两个针脚之间降为接近0V，从而达到控制信号的目的。

Pin针脚的作用是**设备内部的电气控制逻辑决定的**，默认是这两个针脚不连通时触发“自动开机”信号。

选择复位式开关，即 Normal On，按下时开关闭合，松开时断开。Jetson 设备不会要求开关持续按压，只需检测到“按下信号”即可启动。按下按钮时短接 PWR BTN 和 GND，设备感知到这一信号触发开机动作，松开后断开，设备继续启动运行。

