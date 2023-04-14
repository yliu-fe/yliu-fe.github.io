# 关于形参 `*args`和 `**kwargs`的用法与区别

写这个的来源是我在CS61A的第一个项目 `Hog`（投骰子游戏）中被Problem 8卡壳的经历。

项目地址：[Project 1: The Game of Hog | CS 61A Fall 2020 (berkeley.edu)](https://inst.eecs.berkeley.edu/~cs61a/fa20/proj/hog/)

本质上，这两个东西中起作用的是 `*`和`**`即单星号和双星号，而`args`和`kwargs`都是添头——你喜欢的话叫阿猫阿狗都可以——python支持中文变量名。

## `*args`
`*args` 用于向一个函数传入**可变数量**的参数
> The special syntax `*args` in function definitions in Python is used to pass **a variable number of** arguments to a function.

传入的本质上是一个参数的列表(argument list)，而这个列表的长度是可变的，且non-keyworded: