+++
date = '2025-08-31T12:57:45+08:00'
draft = true
title = 'Node.js'
tags = ["node.js"]
categories = ["tools"]
+++



## install & start

两个命令通常用于在 Node.js 项目中进行软件包管理和启动应用程序。

`npm install`:

这个命令会根据项目中的 `package.json` 文件安装所有声明的依赖项。如果项目中有 `package-lock.json` 文件，npm 会根据这个文件确切地安装指定版本的依赖项，以确保在不同的开发环境中获得一致的安装结果。这个命令会在项目目录下创建一个名为 `node_modules` 的文件夹，其中包含所有项目依赖的软件包。

`npm start`:

这个命令用于启动项目中定义的 "start" 脚本。在 `package.json` 文件的 "scripts" 部分，你可能会找到如下的配置：

~~~js
json
"scripts": {
  "start": "node your-app-entry-file.js"
}
~~~

这里的 "start" 就是通过 npm start 启动的脚本。该脚本通常用于启动应用程序，可以是一个服务器，一个开发服务器，或者其他启动项目所需的任何脚本。


## Hexo

我有一个 HEXO blog已经部署到了github上，现在我想在一台新的笔记本上 clone 这个 blog，添加内容修改内容，然后让修改在服务上生效。如何做？

安装 Node.js 和 npm： 确保您的新笔记本上安装了Node.js和npm（Node.js的包管理器）。Hexo 是基于 Node.js 的，因此这是必需的。您可以从 Node.js 官网下载安装包。

~~~sh
# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
# download and install Node.js (you may need to restart the terminal)
nvm install 20
# verifies the right Node.js version is in the environment
node -v
# verifies the right npm version is in the environment
npm -v

~~~

安装Git： 从Git官网下载并安装Git。Git是用于版本控制的工具，您需要它来克隆和推送您的博客。

克隆博客仓库： 打开命令行或终端，使用以下命令克隆您的Hexo博客仓库：

~~~sh
git clone https://github.com/yourusername/yourblogrepository.git
npm install -g hexo-cli  # 全局安装Hexo命令行工具

cd yourblogrepository
npm install  # 安装所需的依赖项

npm install hexo-server --save
hexo server # 在本地启动Hexo服务器，预览您的博客, 在浏览器中访问http://localhost:4000来查看您的博客

hexo new post "Your New Post Title"  # 创建新blog，或修改现有的blogs
hexo generate # 添加或修改后，生成静态文件

git add .
git commit -m "Add new content"
git push origin main  # 提交修改到本地

hexo deploy

~~~

## TODO

Blog repo 中的配置太老了，尝试重新部署 blog，将已有内容让拷贝到。Junhui's Journal blog源文件丢失。hold
