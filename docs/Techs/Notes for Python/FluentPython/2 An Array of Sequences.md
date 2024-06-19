# An array of sequences

用序列表示的数组。

/// admonition | 第二版的调整
		type: note

相比于现有汉译本的第一版，第二版在第二章调整了两处内容：

1. 新增了一节新内容`Pattern Matching with Sequences`，放在“切片”之前
2. 将具名元组(Classic Named Tuples)挪到了全新的第五章“Data Class Builders”中。
///

## 2.1 内置序列

序列（sequence）的分法不同，如果按照存放内容区分：

- 容器序列（container）：它们是所包含的任意对象的引用。这类序列的例子是`list, tuple, collections.deque`
- 扁平序列（flat）：直接存放了对象的值，特点是具有连续的内存空间，但是只能用于存放基础数据类型（数值、字节、字符），例如`str,bytes,bytearray, array.array`等。

或者按照可不可以被修改分类：

- 可变（mutable）序列：`list, bytearray, array.array, collections.deque`
- 不可变(immutable)序列：`tuple, str, bytes`

/// admonition | container sequence v. flat sequence
		type: note

书中图2.1给出了container的典型案例`tuple`和flat典型案例`array`的内存模型。

- array实例`array('d',[9.46,2.08,4.29])`在内存中处于连续位置，头地址后依次存放`9.46`等三个数值。
- tuple实例`tuple(9.46,'cat',[2.08,4.29])`虽然也占据了连续的内存空间，但这块内存空间存储的不是元素本身，而是元素对应的内存地址。`tuple`本体占据的内存中，头地址之后依次存放`9.46`,`'cat'`和`[2.08,4.29]`的内存地址。特别的是，第三个元素是`[2.08,4.29]`，这是个list，也是一个container sequence，因此它所对应的连续地址中存储的还是地址，用于导向真正的数值2.08、4.29所在的位置——这两个数所在的内存地址不一定是连续的。
///

/// admonition | 不建议在具有大量增删操作的数据结构中使用任何的container
		type: warning

在CPython中，列表（list）是通过动态数组实现的。这意味着列表在内存中是连续存储的，可以通过索引快速访问元素，但是插入和删除元素可能需要移动大量的元素。  

当你创建一个列表时，Python会预先分配一些额外的空间用于存储未来可能添加的元素。这样，当你添加元素到列表时，Python通常不需要重新分配内存，只需要在预先分配的空间中添加元素即可。这使得添加元素的操作非常快速。  

当列表的预先分配的空间用完时，Python会创建一个新的、更大的内存区域，然后将旧的元素复制到新的内存区域，并释放旧的内存区域。这个过程被称为重新分配内存，它会消耗一些时间，但是由于预先分配的空间，这种情况发生的频率较低。  

删除元素时，Python不会立即释放内存，而是保留一些空间以便未来添加元素。如果删除大量元素，Python可能会决定缩小列表的大小，这需要重新分配内存。  

总的来说，Python的列表实现提供了良好的平均性能，对于大多数用途来说，它的性能是足够的。然而，如果你需要在列表的中间频繁地插入或删除元素，可能需要考虑使用其他数据结构，如链表。
///

/// admonition | 任意的Python对象都会有一个具有元数据（meta-data）的头部
		type: note

以浮点数`float`为例，在内存中它实际上由三块内容构成：

- `ob_refcnt`：引用次数计数。这是Python垃圾回收机制所需的项目，生成实例时此数为1，当对象被其他对象引用的时候，这个计数会增加，当引用结束或者超出作用域的时候，计数减少。计数为0则Python自动回收其内存。
- `ob_type`：指向对象类型（`float` class）的指针
- `ob_fval`：实际上存储了浮点数内容，它是C语言下的double数据类型。

而这三块内容各自消耗了8个字节（64bit），因此使用array of float比tuple of float的开销要小，因为array在内存中是连续的，直接存储了各个浮点数的原始值（只需要`ob_fval`），而tuple要包含的是整个float object，即上面所说的全部三个对象。
///

## 2.2 列表推导和生成器表达式

> List comprehensions and Generator expressions
> 
> 前者有时被简写为listcomps，后者则是genexps。

### list comprehensions

列表推导可以用于构建`list`，而生成器表达式可以用于构造其他任何类型的序列。

```python
# list comprehensions  
  
symbols ='$¢£¥€¤'  
codes = [ord(symbol) for symbol in symbols]  
print(codes)  
  
print('---')  
# without list comprehension  
codes = []  
for symbol in symbols:  
    codes.append(ord(symbol))  
print(codes)
```

一个简单的list comprehension示例，它代替了整个for循环。

> list comprehension是一个由`[ ]`括起来的东西。而Python编译器会忽略掉`[ ], { }, ( )`中的换行符，所以不需要像其他语言一样打回车的时候加"\\"。但是，如果listcomp太长而复杂了，最好就老老实实写for循环，不丢人。

进一步，listcomp还可以加入条件过滤：

```python
symbols ='$¢£¥€¤'

beyond_ascii = [ord(s) for s in symbols if ord(s) > 127]
# beyond_ascii = [162, 163, 165, 8364, 164]
beyond_ascii = list(filter(lambda c: c > 127, map(ord, symbols)))
# beyond_ascii = [162, 163, 165, 8364, 164]
```

着重介绍第二种方法，这是更为常见的`map+filter`组合：

/// admonition | `map`和`filter`的组合
		type: note

以上面的代码`beyond_ascii = list(filter(lambda c: c > 127, map(ord, symbols)))`为例：

- `map(ord, symbols)`: `map`函数对`symbols`中的每个字符应用`ord`函数。`ord`函数返回一个字符的ASCII值。所以，`map(ord, symbols)`的结果是一个包含`symbols`中所有字符ASCII值的迭代器。  
- `filter(lambda c: c > 127, map(ord, symbols))`: `filter`函数对`map(ord, symbols)`返回的迭代器中的每个元素应用函数`lambda c: c > 127`。这个函数检查一个数是否大于`127`。所以，`filter(lambda c: c > 127, map(ord, symbols))`的结果是一个包含所有ASCII值大于127的元素的迭代器。
- `list(filter(lambda c: c > 127, map(ord, symbols)))`: `list`函数将迭代器转换为列表。所以，`beyond_ascii`是一个包含所有ASCII值大于127的元素的列表。
///

生成器里可以叠for循环，形成一个类似于“笛卡尔积”

```python
>>> colors = ['black', 'white']
>>> sizes = ['S', 'M', 'L']
>>> tshirts = [(color, size) for color in colors for size in sizes]
[('black', 'S'), ('black', 'M'), ('black', 'L'), ('white', 'S'), ('white', 'M'), ('white', 'L')]
```

根据listcomp中for循环的先后顺序，生成的list中的所有元组首先以`color`排序，然后以`size`排序。

### generator expressions

listcomp只能用于生成列表，而不适合生成其他的数据结构。但genexp不是，它遵守了iterator protocol——它不会一次性生成所有元素的列表，你踢它一脚，他就会按顺序给你蹦一个元素出来。它的结构是：`(expression for item in iterable)`，相比于listcomp，它直观上说，把方括号变成了圆括号。

例如要生成上面问题的元组或者array：

```python
>>> symbols = '$¢£¥€¤'
>>> tuple(ord(symbol) for symbol in symbols)
(36, 162, 163, 165, 8364, 164)

>>> import array
>>> array.array('I', (ord(symbol) for symbol in symbols))
array('I', [36, 162, 163, 165, 8364, 164])
```

如果genexp是函数中的唯一参数，那么就不需要用括号把它装起来，就像上面对`tuple`的操作，如果函数中有多个变量则需要用圆括号括起来。

类似的，它也可以用来做笛卡尔积：

```python
>>> colors = ['black', 'white']
>>> sizes = ['S', 'M', 'L']
>>> for tshirt in (f'{c} {s}' for c in colors for s in sizes):
...     print(tshirt)
black S
black M
black L
white S
white M
white L
```

它实际上是由for循环驱动的。如果说我直接写成下面的形式，它就不会回答你任何东西：

```python
>>> colors = ['black', 'white']  
>>> sizes = ['S', 'M', 'L']  
>>> tshirt =((color, size) for color in colors for size in sizes)
>>> print(tshirt)
<generator object <genexpr> at 0x000001F300ED8BA0>
```

而输出它的方法有两种：`next(tshirt)`会按顺序打印下一个元素，或者用下面的for循环：

```python
for item in tshirt:
 print(item)
```

打印可生成的全部元素，`item`是genexp生成的序列中的元素（元组）。

## 2.3 元组：不仅仅是不可变的列表

### 元组是一种记录的手段

没什么好说的，元组中元素的先后顺序、数值都可能有意义，例如：

```python
>>> traveler_ids = [('USA', '31195855'), ('BRA', 'CE342567'),
...                 ('ESP', 'XDA205856')]
>>> for passport in sorted(traveler_ids):  
...     print('%s/%s' % passport)
BRA/CE342567
ESP/XDA205856
USA/31195855
```

这意味着`traveler_ids`存储的三个元组中，每个元组的第一个元素都属于同一个类别（国籍），第二个元素都是护照号。所以可以批量的处理并输出这些信息，甚至可以将其命名或按类型输出：

```python
>>> city, year, pop, chg, area = ('Tokyo', 2003, 32_450, 0.66, 8014)
>>> city
'Tokyo'

>>> for country, _ in traveler_ids:  
...     print(country)
USA
BRA
ESP
```

### 元组作为一种不可变列表

`tuple`类支持`list`类中 **除了增删以外的**几乎所有方法（`__reversed__`除外，但它可以通过`reversed(tuple)`直接实现）：`__add__`, `__contains__`, `count`, `__getitem__`, `__getnewargs__`, `index`, `__iter__`, `__len__`, `__mul__`, `__rmul__`。

## 2.4 对序列和可迭代对象的拆包

> 这一部分在第二版中被单独拿出来作为一个二级标题。讨论对元组、列表、可迭代对象的拆包（unpacking）操作。

最简单的方法是平行赋值(parallel assignment)，把一个可迭代对象的值，赋到一个变量的元组中：

```python
# normal example
city, year, pop, chg, area = ('Tokyo', 2003, 32_450, 0.66, 8014)

# exchange value using parallel assignment
b,a = a,b 

# Unpack an iterable as function's arguments, use `*`
>>> divmod(20,8)
(2,4)
>>> t = (20,8)
>>> divmod(*t)
(2,4)

# use function's output as tuple
>>> import os
>>> _, filename = os.path.split('home/luciano/.ssh/idrsa.pub')
>>> filename
'idrsa.pub'
```

上面的第三种玩法是用星号`*`拆开一个元组（或其他可迭代对象），解出其全部元素，作为函数的一个参数，可能更熟悉`func(*args)`的写法。

最后一种玩法是把函数`os.path.split`返回的元组给解开了（它原本返回的是文件的路径和文件名`(path, last_part)`，现在我只要后半部分），把路径的部分挂了占位符`_`忽略掉，只保留需要的文件名。

### 用星号`*`处理不确定数量的元素

```python
>>> a, b, *rest = range(5)
>>> a, b, rest
(0, 1, [2, 3, 4])

>>> a, b, *rest = range(2)
>>> a, b, rest
(0, 1, [])

>>> a, *rest, b = range(5)
>>> a, rest, b
(0, [1, 2, 3], 4)
```

原则是：先可着普通变量赋值。最后，挂星号的`rest`有多少吃多少，没有就返回空列表。

### 在函数调用或者序列字面量中用`*`解包

讲`*args`。看下面的例子：

```python
def fun(a, b, c, d, *rest):  
    return a, b, c, d, rest

>>> fun(*[1, 2], 3, *range(4, 7))
(1, 2, 3, 4, (5, 6))
```

下面实例里，fun()中输入的内容说白了就是`1,2,3,4,5,6`，但在函数声明中，第五个参数及其以后的内容被归入了`rest`项目中统一输出，因此在函数输出的大元组中，前四项（1-4）挨个输出，剩下的都在第五项里，形成嵌套的小元组。

类似的，声明新元组、新列表时，也可以用`*args`的形式：

```python
*range(4), 4 # tuple
[*range(4), 4] # list
{*range(4), 4, *(5, 6, 7)} # set
```

> 而另一种函数解包的形式，即`**kwargs`在第三章“Unpacking Mappings”。

### 嵌套元组的解包

> Nested Unpacking

```python title="west_cities.py"
metro_areas = [
('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),
('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
('São Paulo', 'BR', 19.649, (-23.547778, -46.635833)),
]

def main():
 print(f'{"":15} | {"latitude":>9} | {"longitude":>9}')
 for name, _, _, (lat, lon) in metro_areas:
  if lon <= 0:
   print(f'{name:15} | {lat:9.4f} | {lon:9.4f}')

if __name__ == '__main__':
 main()
```

刨开制表用的代码，这里的内容完全靠解包`metro_areas`实现，即便元组中嵌套了经纬度的小元组，也可以通过`for var1, ..., (tvar1,tvar2,...) in nested_tuple`逐个挖出子元组中的数值。

/// admonition | 单个元素的元组需要保留一个逗号
		type: danger
元组`('single')`等同于字符串`'single'`，而`('single',)`才是一个单元素的元组。这种问题常见于查询后只有一行数据，甚至只有一行数据一个字段的情况，返回的结果很有可能不是元组（而是直接被退化为数值、字符串等），从而不适配于后面处理元组的代码，从而形成静默的bug。
///

## 2.5 基于序列的模式匹配
