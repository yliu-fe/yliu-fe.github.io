# 作业1： 模仿学习

作业内容PDF：[hw1.pdf](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw1.pdf) （Fall 2022）

框架代码可在该仓库下载：[ Assignments for Berkeley CS 285: Deep Reinforcement Learning (Fall 2022) ](https://github.com/berkeleydeeprlcourse/homework_fall2022)

该项作业要求完成模仿学习的相关实验，包括直接的行为复制和DAgger算法的实现。由于不具备现实指导的条件，因此该作业给予一个专家策略，来做数据的标注。

最后，利用OpenAI Gym上的若干个benchmark 连续控制任务，来比较直接模仿学习和DAgger的表现。

> 注意：CS285的作业可能需要GPU，如有必要请上Colab。

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

按照指引，先看七个文件：

```
scripts/run_hw1.py (read-only)
infrastructure/rl_trainer.py
agents/bc_agent.py (another read-only file)
policies/MLP_policy.py
infrastructure/replay_buffer.py
infrastructure/utils.py
infrastructure/pytorch_util.py
```
其中`run_hw1.py`和`bc_agent.py`只需要读，而另外五个里面有TODO，需要写代码。

> 但`run_hw1.py`中一些代码后写着注释"HW1: you will modify this"，所以也要做一些调节

### 总文件：`run_hw1.py`
相对路径：`cs285/scripts/run_hw1.py`。显然，整套文件最后归于此。

该文件的核心是定义了`BC_trainer`类。在这套作业中，BC代表"Behavior Cloning"，
```python
class BC_trainer(object):
    def __init__(self, params)
    def run_training_loop(self)
```

这个类的`__init__`函数看起来像是训练过程的函数，因为定义的`self.params['agent_params']`和`self.params['env_kwargs']`都来自外界传参，前者包括了NN的层数`n_layers`、节点量`size`、学习率`learning_rate`和重放缓冲区最大规模`max_replay_buffer_size`；后者则来自所执行环境。在定义参数之后，便是`RL_Trainer`和`loaded_expert_policy`两步过程，前者负责定义训练，后者负责载入专家策略。

另一个函数`run_training_loop`只是调用了`rl_trainer.pu`中的同名函数，并传入了相关参数：
```python
def run_training_loop(self):

    self.rl_trainer.run_training_loop(
        n_iter=self.params['n_iter'], #迭代次数规定
        initial_expertdata=self.params['expert_data'], #(未发现位置)初始专家数据（训练集）
        collect_policy=self.rl_trainer.agent.actor, #拾取策略？
        eval_policy=self.rl_trainer.agent.actor, #评估策略
        relabel_with_expert=self.params['do_dagger'], # （未发现位置，可能是布尔值）在学习后，由专家对新发现状态做标注
        expert_policy=self.loaded_expert_policy, # 所用专家策略
    )
```

下方大量的`parser`命令无需在意，它们负责shell指令的传达和判断。也不能说完全不在意吧，因为命令行里将传递各个变量，比如上面提到的`do_dagger`, `expert_data`什么的。以HW1.pdf的作业一"behavior cloning"中给出的Shell代码，我们可以读到：

```bash
# 具体输入到shell里时，每行最右边的反斜线要删掉
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1
```

/// details | 关于这行shell命令
    type: note
  
第一行：执行`run_hw1.py`这个文件  

第二行：指定专家策略文件`Ant.pkl`

第三行：指定环境为`Ant-v4`，实验名`bc_ant`，迭代次数1（该项默认值为1）

第四行：专家数据（训练集）：`expert_data_Ant-v4.pkl`

第五行：禁用视频日志（该项默认值为5，而设为-1则代表禁用了以视频方式保存训练日志）

///

接着就是保存日志和真正的训练过程，训练过程只有两行命令：
```python
trainer = BC_Trainer(params)
trainer.run_training_loop()
```
定义训练器为`BC_trainer`类，并传入参数；然后开始不断执行`run_training_loop`函数，而这个函数在`rl_trainer.py`文件中定义，所以下一步，我们要开始编写`rl_trainer.py`。

### 作业文件1：`rl_trainer.py`
相对路径`cs285/infrastructure/rl_trainer.py`，它定义了RL（尽管模仿学习并不是真正的RL）学习器和函数。先略过前面的import环节和两个常量，我们先看这个文件的核心`RL_Trainer`类：

```python
class RL_Trainer(object):
    def __init__(self, params)
    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None)
    def collect_training_trajectories(self,
            itr,
            load_initial_expertdata,
            collect_policy,
            batch_size,
    )
    def train_agent(self)
    def do_relabel_with_expert(self, expert_policy, paths)
    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs)
```
也就区区六个函数...