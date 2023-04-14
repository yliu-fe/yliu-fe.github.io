# 作业1： 模仿学习

作业内容PDF：[hw1.pdf](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw1.pdf)

框架代码可在该仓库下载：[ Assignments for Berkeley CS 285: Deep Reinforcement Learning (Fall 2022) ](https://github.com/berkeleydeeprlcourse/homework_fall2022)

该项作业要求完成模仿学习的相关实验，包括直接的行为复制和DAgger算法的实现。由于不具备现实指导的条件，因此该作业给予一个专家策略，来做数据的标注。

最后，利用OpenAI Gym上的若干个benchmark 连续控制任务，来比较直接模仿学习和DAgger的表现。

/// details | 关于绘图规范
    type: danger
    open: true
UCB给出了一套在实验报告中绘图的规范：[viz.pdf](http://rail.eecs.berkeley.edu/deeprlcourse/static/misc/viz.pdf)。可参照，利用matplotlib.pyplot和seaborn完成。

> 请注意，这套规范的示例代码中，引用了 `seaborn`库中的 `tsplot`函数，而该函数在较久远的seaborn更新中已被移除，其替代品为 `seaborn.lineplot` 函数。

1. 代码应该在外部文件（如csv或pkl文件）中保存实验的结果，而不是直接出图。这样的话可以对实验反复尝试绘图，直到有一个好的结果。通常记录的信息包括：（1）每次迭代的平均奖励或损失、部分采样的轨迹、有用的二级指标（Bellman偏误或者梯度大小）
2. 绘图应当单独写一个脚本。如果采用不同的超参数或者随机种子运行算法，或者运行了不同的算法，或者运行当前算法的变体，那么最好把所有的实验日志数据加载到一起并绘制到一张图上，记得做好图例和颜色方案。
3. DRL算法，特别是其中的Model-free算法，在不同次的运行中产生的结果会有很大区别，所以多做几组随机种子来做几次实验，将他们的运行轨迹画在一张图上，最好用粗一点的线再画上他们几个的平均表现。也许平均值图和标准差图比较方便，但这种随机实验不见得遵循正态分布，所以把所有的运行图画出来，可以更好的了解随机种子之间的差异。
   ///

## 安装依赖项

请参阅 `hw1` 文件夹中 `installization.md` 的说明，建议走A路线，即安装conda并创建虚拟环境

### 问题1：Conda装包时提示 `environment is inconsistent`：

即提示：

```shell
The environment is inconsistent, please check the package plan carefully
The following package are causing the inconsistency:

   - defaults/win-32::anaconda==5.3.1=py37_0
# 这个名单还可以加长
```
这边建议您直接卸了Anaconda重装呢，直接重装解千愁。

### 问题2： Windows命令行、Powershell或终端中无法激活环境

问题描述：利用Windows命令提示符(cmd.exe)、Windows PowerShell、Windows Terminal输入命令行激活新环境时，提示：

```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
If using 'conda activate' from a batch script, change your
invocation to 'CALL conda.bat activate'.

To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - cmd.exe
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'. 
```

解决方案如报错信息所示，输入 `conda init xxx`，其中 `xxx`根据所用命令行工具而异，命令提示符用 `cmd.exe`，Powershell或Terminal用 `powershell`（仅限于Windows用户）

参照：[conda init — conda 0.0.0.dev0+placeholder documentation](https://docs.conda.io/projects/conda/en/latest/commands/init.html)

### 问题3:依赖项安装报错

首先，在hw1中安装的库均对应于Python 3.7，可能已落后于现有版本（所以要开虚拟环境而不是直接装base里）

其次，`requirement.txt`里的确有一个有问题的库，即 `box2d-py==2.3.8`，如果打开hw1所在文件夹，输入Shell命令：

```shell
pip install -r requirements.txt
```

大概率会因为 `box2d-py`报错。这种情况下，首先在requirements.txt里删去 `box2d-py==2.3.8`一行，保存后再运行上面的命令行，就可以把剩下的包都安装好，而box2d的包请移步：[Archived: Python Extension Packages for Windows - Christoph Gohlke (uci.edu)](https://www.lfd.uci.edu/~gohlke/pythonlibs/)，寻找 `Pybox2d`的whl文件并按以下命令安装：

```shell
pip install <PATH>\Box2D‑2.3.10‑cp37‑cp37m‑win_amd64.whl
```

其中，whl文件名中的 `2.3.10`代表版本号（该网站只提供了2.3.2和2.3.10），`cp37`代表Python 3.7版本，`win_amd64`或 `win32`代表64位、32位系统。

最后，可以运行 `pip list`检查环境中的包列表，并确认requirements.txt中要求的包是否都已按版本要求安装。以及，按要求完成第五项，即运行

```shell
pip install -e .
```

安装文件夹里那个 `cs285`包，接下来就进入正式的作业环节。

## 作业1：Behavioral Cloning（动作复制）

需要完成的填空位于：

```
infrastructure/rl_trainer.py
policies/MLP_policy.py
infrastructure/replay_buffer.py
infrastructure/utils.py
infrastructure/pytorch_util.py
```
