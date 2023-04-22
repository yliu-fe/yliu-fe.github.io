---
comments: True
---

# 高阶函数、递归和homework 2


## hw2: Recursion
作业地址：[Homework 2 | CS 61A Fall 2020](https://inst.eecs.berkeley.edu/~cs61a/fa20/hw/hw02/)

### 第二题：Ping-pong

/// admonition | 题目描述：Ping-pong
    type: info

“Ping-pong”序列从1起数，要么递增要么递减。对于第`k`个元素，如果`k`能被8整除，或者`k`含有8，那么递增/递减的方向将改变。例如，第8、16、18、24、28个元素后，序列计数方向将改变。

实现一个函数`pingpong`，在**不使用任何赋值语句**的情况下，返回第`n`个元素的值。

> Implement a function `pingpong` that returns the nth element of the ping-pong sequence without using any assignment statements.

提示：可以使用`num_eights`（第一题所做函数）来判断数字中是否存在8。
///

这个题我是横竖没想到怎么在不用赋值语句的情况下确定方向的，直到我看了hint video，然后发现老师直接整了个closure......这属于是绕开了检测程序了属于是，本质上还是在赋值：

```python title="Ping-pong" linenums="1"
def pingpong(n):

    def pp_helper(index, dir, ppn):
        if n == 1:
            return 1
        elif index == n:
            return ppn
        else:
            if index % 8 == 0 or num_eights(index) != 0:
                return pp_helper(index + 1, -dir, ppn-dir)
            else:
                return pp_helper(index + 1, dir, ppn+dir)
    return pp_helper(1,1,1)
```
说白了，闭包函数`pp_helper`里的三个参数可以都算作assignment statements（赋值语句），相当于`index,dir,ppn = 1,1,1`，三者的意义分别是（1）ping-pong序列的序号；（2）变化方向,`dir = 1`递增，`dir = -1`递减；（3）`ppn`是ping-pong序列第`index`个值。

那么，经过了这种特殊的“赋值”，一切就变得明朗，直接通过`pp_helper`的参数来做递归，`index == n`了就输出`ppn`，一切就很水到渠成。而且序号符合变向条件时，也是在下一个值上变向，所以不需要在前一个序号上考虑变向。

我唯一感到疑惑的是`if n == 1: return 1`这句，去掉这个判断，运行`pingpong(1)`同样可以得到1，因为`elif index == n: return ppn`得到的结果相同，通过[PythonTutor Visualizer](https://pythontutor.com/python-debugger.html#mode=edit)做函数结构的可视化，得到的结果也是相同的。

### 第四题：Count coins

这道题属于是Tree Recursion：
```python title="Count coins" linenums="1"
def count_coins(total):
    def total_helper(least, n):
        if least is None: #next_largest_coin(25) = None
            return 0
        elif least == n: #递归吸收1：最小硬币面值恰好为n，只有一种
            return 1
        elif least > n: #递归吸收2：最小硬币面值大于n，没有可能
            return 0
        with_c = total_helper(least, n - least)
        without_c = total_helper(next_largest_coin(least), n)
        return with_c + without_c
    return total_helper(1, total)
```

我们假定当前最小面值为`least`，还没有配出来的余额为`n`，那么接下来有两个情况：

1. 下一个硬币仍然是当前的“最小面值”`least`，但这样的话还没有配出的额度为`n-least`；
2. 下一个硬币是当前最小面值的下一个稍大的面值(`next_largest_coin(least)`)。这样的话，只是调整了`least`的设定，而没有真正地配置一个硬币进去。

每一种情况都继续向下分裂为上面所说的两种情况，从而将每一种吸收的情况都讨论到（也就是最上面的`if-elif-elif`），然后把这些情况中返回的`0`或`1`加起来。
