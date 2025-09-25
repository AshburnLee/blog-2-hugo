+++
date = '2025-08-31T12:57:44+08:00'
draft = false
title = 'Git'
tags = ["git"]
categories = ["tools"]
+++


Push 和提 PR 是两个概念。**一个PR其实是一个分支的概念**，push 就 PR 该后，需要在 github 页面中手动提 PR（当然也可以通过gh命令）。这里要表达的是push和提PR是两回事。

要想在 github 页面显示自己的分支，必须要 push。不想让别人看见自己的分支就不要提 PR。


# Git
## 常用命令

~~~sh
git log --oneline
git rm -r --cached build  # 将git上的build文件夹删除，同时保留本地。然后commit push后，远端的build也就被删除了。这里有个cache的概念。 

git reflog
git reset --hard 1359d449
git show-ref --verify --quiet
git rebase
git rev-parse HEAD
git submodule sync && git submodule update --init --recursive

git diff > my.diff
git apply my.diff  # 在新的 branch

git clone --single-branch

git log --author="Author Name"
git log --grep="message"
git revert
git config --list

git commit --amend -m "new commit message"
git push origin junhui_typo --force

git reset --soft HEAD^  # 撤销上一个commit
git reset --hard HEAD^  # 撤销所有commit
git reset --hard        # 撤销上一个操作（当上一个操作结果不如预期时）
~~~

## 如果我有两个commit，先有commit a 后有commit b，如何在commit a中修改代码

~~~sh
# To modify code in commit a, you can use the `git rebase` command.
# Here's an example:

# 1. Start an interactive rebase session
git rebase -i HEAD~2

# 2. In the interactive rebase editor, change "pick" to "edit" for commit a
#    and save the file

# 3. Git will stop at commit a. Now you can modify the code as needed
# 在commit a中添加你的修改

# 4. Stage the changes
git add .

# 5. Amend the commit
git commit --amend  # 似乎不能是 --no-edit, 佛则这个commit就会消失

# 6. Continue the rebase
git rebase --continue

# The code in commit a is now modified.
~~~


## 3个commit，压缩为一个commit

~~~bash
git log
# step1
git rebase -i lastCommitID  //lastCommitID 倒数第四次提交
# OR
git rebase -i HEAD~3

# step2 vim 编辑
# 要把*下面*两个红色框 ‘pick’ 改为‘s’,表示第三次提交合并入第二次，第二次提交合并入第一次。
# 保存退出
git push -f BRANCHNAME
~~~

## Cherry-pick
~~~sh
    a - b - c - d     Master
         \
           e - f - g  Feature

git checkout master
# Cherry pick 操作, 【仅仅pick这一个commit】
git cherry-pick f

#结果
    a - b - c - d - f   Master
         \
           e - f - g    Feature
# 若有冲突，解决后
git cherry-pick --continue
~~~

优势：cherry-pick 比笨方法好，笨方法会将重命名后的文件都保留

[git如何cherry-pick 一个PR的多个commit？]
[git如何cherry-pick 多个commit？]
`git cherry-pick commit1 commit2 commit3`


cherry-pick 多个commits：
`git cherry-pick A^..B` 【A必须早与B，这个A^表示A是被包含的】
`git cherry-pick 6fba45042585eac431138ae0aa8ebd579b09872b^..aef984b66360661b3116d9d1c1c9ca0cad66bf7f`

## Git merge

~~~sh
    a - b     master
         \
           e - f  feature

git checkout master
git merge feature

# 结果：
    a - b - e - f     master
# 如果你是master的author，当 master更新后，可以直接push（无PR），此时master远端也就更新了（有了新的feature）
#
~~~

## 坑

`rebase` 和 `rebase --continue` 之间不要 `commit`。


## git 在别人的分支上
## 提PR 直接通过upstream上创建分支
## 提PR 通过fork的repo上创建分支
## github webhook

Webhook 是一种使一个应用提供实时信息给其他应用的方法。它在特定事件发生时发送 HTTP 请求（通常是 POST(理解为发送一个请求到某一个位置) 请求）到预先配置的 URL。

在 GitHub 中，你可以设置 webhook 来监听你的仓库的各种事件，如 push 事件、pull request 事件、issue 事件等。当这些事件发生时，GitHub 会发送一个包含事件详细信息的 HTTP 请求到你设置的 webhook URL。

Webhook 的主要作用是实现**实时的、自动**的集成。例如，你可以设置一个 webhook，当你的仓库收到新的 push 时，自动触发 Jenkins 的构建，或者当新的 issue 被创建时，自动发送通知到你的 Slack 频道。

如果你不使用 webhook，你就需要定期轮询 GitHub 的 API 来检查是否有新的事件。这可能会导致延迟（因为你需要等待下一次轮询才能检测到新的事件），并且可能会浪费资源（因为你需要不断地发送请求，即使没有新的事件）。而使用 webhook，你可以立即得到通知，并且只在有新的事件时才需要处理请求，从而提高效率和响应速度。

# gh

~~~sh
gh --version
conda install gh

gh auth login #你首先要login，才可以使用gh访问指定repo
gh run list --repo
gh run view <run-id> --repo 
gh run download --repo xxx --name yyy #下载Action中的artifact
~~~

在自动化中使用，在shell中使用，自动提PR，等


## 当终端不支持Unicode显示时

`git config --global core.quotepath false`

这样就可以显示中文了。
