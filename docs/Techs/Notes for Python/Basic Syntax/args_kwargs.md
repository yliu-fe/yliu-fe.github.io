# 关于形参 `*args`和 `**kwargs`的用法与区别

写这个的来源是我在CS61A的第一个项目 `Hog`（投骰子游戏）中被Problem 8卡壳的经历。

项目地址：[Project 1: The Game of Hog | CS 61A Fall 2020 (berkeley.edu)](https://inst.eecs.berkeley.edu/~cs61a/fa20/proj/hog/)

参照：[\*args and \**kwargs in Python](https://towardsdatascience.com/args-kwargs-python-d9c71b220970)

本质上，这两个东西中起作用的是 `*`和`**`即单星号和双星号，而`args`和`kwargs`都是添头——你喜欢的话叫阿猫阿狗都可以——python支持中文变量名。

## `*args`
`*args` 用于向一个函数传入**可变数量**的参数
> The special syntax `*args` in function definitions in Python is used to pass **a variable number of** arguments to a function.

想在Python中写一个能接受任意数量参数的函数。一种方式是写一个list，但这不够Pythonic，也不够方便，因此引入了`*args`。其中`*`是unpacking operator，它将读取传入的所有参数，并返回一个tuple，例如：

```python
def sum_nums(*args):
    sum = 0
    for n in args:
        sum += n
    return sum
```

同时，函数定义时也可以同时声明普通形参和`*args`，例如：`def func(param, *args)`。

## 关键词参数和`**kwargs`
关键字形参指的是在call function的时候，明确声明参数对应哪一个形式参数的情况，如：
```
func(a = 1, b = 2)
```
类似的，`**kwargs`同样是把所有的参数解包，然后形成一个dictionary：
```python
def my_func(**kwargs):
    for key,val in kwargs.items()
        print(key, val)
```
运行结果如：
```
>>> my_func(a='hello', b=10)
a hello
b 10
>>> my_func(param1=True, param2=10.5)
param1 True
param2 10.5
```
同理，`def my_func(param, *args, **kwargs)`也是可行的。

## 什么时候适合用`*args`和`**kwargs`？
比如高阶函数，可以参照CS61A的[Lecture 5 - Higher-order functions](https://cs61a.org/resources/#higher-order-functions)，也就是函数接受另一个函数作为参数，并且返回那个参数。

```python title="CS61A 2020fall Project 1 Problem 8 example"
def printed(f):
     def print_and_return(*args):
         result = f(*args)
         print('Result:', result)
         return result
     return print_and_return
```

这不是Problem 8的实际解决方案，而是题目中给出的Higher-order Function示例，这种结构怎么用呢：

```
>>> printed_pow = printed(pow)
>>> printed_pow(2, 8)
Result: 256
256
>>> printed_abs = printed(abs)
>>> printed_abs(-10)
Result: 10
10
```
两个例子，传入的分别是`pow`和`abs`两个现成的函数，显然`printed_pow`也是一个函数：
```
<function printed.<locals>.print_and_return at (内存地址) >
```
如果不传参进去，例如`printed_pow()`，会报错，因为没给`pow`函数足够的参数。而给定足够的参数后，就先在`print_and_return`函数中调用`print`函数，然后包在外层的`printed`函数返回了`pow(2,8)`的值即`256`。

再给一个例子，就是上面参考中的Towardsdatascience中的博文例子，来自Giorgos Myrianthous:

```python
import functools
import time

def execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end - start}s to run.')
    return wrapper
```

