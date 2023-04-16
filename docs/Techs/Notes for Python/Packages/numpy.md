# Numpy踩坑专题

关于Numpy的若干踩坑事项

## 0x00 `ndarray`类型下的矩阵乘法与逐元素乘法

首先，这里说的 `numpy.ndarray`类型变量。如果采用 `matrix`类型，则他们之间的矩阵乘法是 `a * b`，而逐个元素的乘法用 `numpy.multiply(a,b)`。

`ndarray`下，`a * b`指的是逐个元素的相乘，而 `a @ b`用的是矩阵乘法。

如果算出来的是1*1的标量，numpy也会显示为向量的形式，带着中括号。如果要去除这一个东西，就可以采用:`a@b.squeeze`的命令，把所有长度为1的维度去掉，那么这个结果就变成标量，也就是一个数了。


```python title= "multiply vs *"
import numpy as np
x = np.array([[1,2,3], [4,5,6]])
y = np.array([[-1, 2, 0], [-2, 5, 1]])

x*y
Out: 
array([[-1,  4,  0],
       [-8, 25,  6]])

%timeit x*y
1000000 loops, best of 3: 421 ns per loop

np.multiply(x,y)
Out: 
array([[-1,  4,  0],
       [-8, 25,  6]])

%timeit np.multiply(x, y)
1000000 loops, best of 3: 457 ns per loop
```

