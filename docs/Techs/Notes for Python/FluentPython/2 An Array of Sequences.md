---
comments: true
---

# 各种序列

用序列表示的数组。

/// admonition | 第二版的调整
		type: info

相比于现有汉译本的第一版，第二版在第二章调整了两处内容：

1. 新增了一节新内容`Pattern Matching with Sequences`，放在“切片”之前
2. 将具名元组(Classic Named Tuples)挪到了全新的第五章“Data Class Builders”中。
///

## 2.1 内置序列

序列（sequence）的分法不同，如果按照存放内容区分：

- 容器序列（container）：它们是所包含的任意对象的引用。这类序列的例子是`list, tuple, collections.deque`
- 扁平序列（flat）：直接存放了对象的值，特点是具有连续的内存空间，但是只能用于存放基础数据类型（数值、字节、字符），例如`str,bytes,bytearray, array.array`等。

> flat sequence是作者自创的名词，只是为了和container sequence相对照。

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

> list comprehension是一个由`[ ]`括起来的东西。而Python编译器会忽略掉`[ ], { }, ( )`中的换行符，所以不需要像其他语言一样打回车的时候加"\\"
> 但是，如果listcomp太长而复杂了，最好就老老实实写for循环，不丢人。

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
		type: info

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

`tuple`类支持`list`类中 **除了增删以外的**几乎所有方法（`__reversed__`除外，但它可以通过`reversed(tuple)`直接实现）：`__add__`,`__contains__`,`count`,`__getitem__`,`__getnewargs__`,`index`,`__iter__`,`__len__`,`__mul__`,`__rmul__`。

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

## 2.5 序列的模式匹配

> 对3.10后加入的PEP 634(match-case)的新增内容。类似的新增内容还有Ch3中的“Pattern matching with mappings”, Ch5中的"Pattern Matching Class Instances"。

Match-Case结构可以在解包序列后进行模式的匹配：

```python
# message like ['BEEPER',440,3]

def handle_command(self, message):
	match message:
		case ['BEEPER', frequency, times]:
			self.beep(times, frequency)
		case ['NECK', angle]:
			self.rotate_neck(angle)
		case ['LED', ident, intensity]:
			self.leds[ident].set_brightness(ident, intensity)
		case ['LED', ident, red, green, blue]:
			self.leds[ident].set_color(ident, red, green, blue)
		case _:
			raise InvalidCommand(message)
```

这是某个类中的方法，读入的是列表形式的参数集，但是不同的指令（BEEPER、NECK、LED）对应的参数格式（数量）不一定相同。首先用第一个参数（指令类型）硬匹配，匹配到对应指令后，列表中的其他参数都可以解包到各个函数中变量中去，例如`['BEEPER',440,3]`会被第一个case匹配上，而`frequency, times`分别被赋值为440和3。

> 类似的操作也可以用于元组的模式匹配上，无非是把方括号换成圆的。除此之外，`memoryview, array.array, range, collections.deque`四种序列也可以直接通过match-case结构来判断。

### 字符串是一种特例

但是字符串等不能直接这么做。`str,bytes,bytearray`在match-case中被视作一个个体而非序列。如果希望将其作为一个序列，以一个字符作为单位进行匹配，要使用下面的方式：

```python
phone_example = '1234567890'
match tuple(phone_example):
	case ['1',*rest]
		...
	case ['2',*rest]
		...
	case ['3'|'4', * rest]
		...
	case _
		...
```

这样就可以匹配头一个字符了。

### 下划线作为占位符

下划线`_`在这里可以起到两种作用：（1）作为match-case结构的兜底，所有不符合已定义case的情况都按照`case _:`规定的做法来处理；（2）作为解包时的占位符，例如：

```python
case [name, _, _, (lat,lon) as coord]
# data = ['Shanghai', 'CN', 24.9, (31.1, 121.3)]
```

其结果是：

- `name`被赋值为`Shanghai`
- `CN`,`24.9`两项数据被忽略
- `lat, lon`分别为`31.1`和`121.3`。同时，经纬度元组`(31.1, 121.3)`被整体命名为`coord`。

而且，`*args`的方式也可以被运用，在这种情况下，我们可以让match-case只匹配一部分元素，而不需要讨论其中间项有多少个。

```python
case [name, *extras, (lat,lon)]
```

在这种情况下，编译器将检查第一个元素和最后一个元素。第一个元素被命名为`name`,最后一个元素需要是双元素的元组，其中的元素被命名为`lat`和`lon`。中间的元素有多少个编译器是不在意的，它们被统一打包为一个名叫`extras`的列表。如果中间元素不重要的话，也可以改为`*_`直接让编译器将其忽略。

### 检查数据类型

进一步，match-case可以用序列中各个元素的数据类型来作为检查条件：

```python
case [str(name), _, _, (float(lat), float(lon))]
# for this case,  the first item must be an instance of str, and both items in the 2-tuple must be instances of float
```

在这种情况下，case不仅会检查序列是否由4个元素（且最后一个元素是含有两个元素的元组）构成，而且会检查特定位置元素的数据类型，在上面的情况中，序列的第一个元素必须是`str`，而双元素元组中的每个元素都应当是浮点数。

### 模式匹配方法在语言解释器中的作用

这一段以Peter Norvig写的一个基于Python的Scheme解释器`lis.py`为例，讨论了序列下Pattern Matching在编程语言解析中提升的效率。

/// admonition | 关于`lis.py`的一些说明
		type: info

原项目文件（Github）：[lispy/original/norvig/lis.py](https://github.com/fluentpython/lispy/blob/main/original/norvig/lis.py)，基于MIT license开源。Fluent Python作者（L. Ramalho）在讲解时对代码中部分函数名称做了修改，如`eval`改为`evaluate`，以避免与标准库中的`eval`冲突，下文沿用书中修改后的名称。
///

原项目在`evalate`函数（识别被`parse`函数切分为列表后的Scheme代码并解析为python list）中采取了一套完整的if-else结构（项目文件line 110开始）。这些`elif`项都有一个统一的特征：

- 是否进入这种情况要判断expression（`parse`函数切分后的Scheme代码）中的首项：它们通常规定了这行Scheme代码的用处，例如`define, lambda, if, quote`。
- 进入这一elif语句后，首项就不再有任何用处，直接做成占位符`_`忽略掉。

例如第一个关键词是`define`的情况：

```python
elif exp[0] == 'define':
	(_, name, value_exp) = exp
	env[name] = evaluate(value_exp, env)
# name: 函数名; exp: 函数体内容
```

而现在可以用match-case重构这个方法：

```python
case ['define', Symbol() as name, value_exp]:
	env[name] = evaluate(value_exp, env)

# 对于稍微复杂的带参函数(define (name parms...) body1 body2...)
case ['define', [Symbol() as name, *parms], *body] if body:
	env[name] = Procedure(parms, body, env)
```

match-case结构比if-else结构更加易读，而且可以在保持直观性的基础上加入更多必要的检查项目。例如重构后的`define`情形，就额外检查了`name`是否是`Symbol`类型的一个实例。

## 2.6 切片

### 为什么切片操作和range类型都不包含最后一项？

一般包括几个原因：

- 如果range或者slice只给出停止位置的时候，不包含最后一项可以比较容易地看出切片后序列或者range中有多少项，因为以0开始。
- `stop - start`可以轻松计算切片或者range的长度
- 可以有效分割序列并且不使其重合：例如`lis[:2]`和`lis[2:]`中，前者只包括前两项，而不包括`lis[2]`，后者从`lis[2]`开始切到结束，二者没有重叠元素。

### 切片对象（slice object）

切片运算符是可以写成`sequence[a:b:c]`的，初始位置`a`，结束位置`b`，步长`c`。python实现的方式是生成一个slice object：`slice(a,b,c)`，调用了`sequence.__getitem__(slice(start,stop,step))`。利用好切片对象可以有效地分割处理一些数据，参照书中Example 2-13，发票内容数据的例子。

```python title="example 2-13"
# invoice raw data
invoice = """  
0.....6.................................40........52...55........  
1909 Pimoroni PiBrella                      $17.50    3    $52.50  
1489 6mm Tactile Switch x20                  $4.95    2    $9.90  
1510 Panavise Jr. - PV-201                  $28.00    1    $28.00  
1601 PiTFT Mini Kit 320x240                 $34.95    1    $34.95  
"""

# many slice objects
SKU = slice(0, 6)  
DESCRIPTION = slice(6, 40)  
UNIT_PRICE = slice(40, 52)  
QUANTITY = slice(52, 55)  
ITEM_TOTAL = slice(55, None)

>>> SKU
slice(0, 6, None)

# delete the ruler, print the table containing prices and name
>>>line_items = invoice.split('\n')[2:]  
>>> for item in line_items:  
...     print(item[UNIT_PRICE], item[DESCRIPTION])

    $17.50   imoroni PiBrella                  
     $4.95   mm Tactile Switch x20             
    $28.00   anavise Jr. - PV-201              
    $34.95   iTFT Mini Kit 320x240    
```

其中，被分段切片并命名的都是slice object，而非实际的数据。

### 多维切片和省略号的使用

> Python标准库定义的数据类型中罕有使用到多维切片和省略号的，通常都是用户自定义类型或者numpy这样的科学计算库。

对于多维的序列，例如`numpy.ndarray`类型，可以用多维的切片`example[dim1slice,dim2slice]`来选取其中的部分数据，其本质上是调用了`example.__getitem__((dim1slice, dim2slice))`，例如：

```python
import numpy as np  
  
# Create a 2D numpy array  
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  
  
# Fetch an item  
print(a[1, 2])  # Output: 6  
  
# Fetch a 2D slice  
print(a[0:2, 1:3])  
# Output:  
# [[2 3]  
#  [5 6]]
```

> 但是，通常情况下的`memoryview`类型是一维的，不能执行多维切片。特殊情况见最后的"When a list is not the answer"章节。

省略号用来替代切片中连续出现的冒号，例如：

```python
import numpy as np  
  
# 创建一个三维数组  
a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  
  
# 使用省略号获取第一维度的所有数据  
print(a[0, ...])  # 输出：[[1 2 3] [4 5 6]]
```

其中的`a[0,...]`等价于`a[0,:,:]`。

### 切片用于就地更改可变序列

利用切片运算符，可以就地修改、删除、嫁接可变序列。

```python
>>> l = list(range(10))  
>>> print(l)  
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> l[2:5] = (20,30)  # l[2:5] = [20,30]得到的结果相同
>>> print(l)
[0, 1, 20, 30, 5, 6, 7, 8, 9]

>>> del l[5:7]
>>> print(l)
[0, 1, 20, 30, 5, 8, 9]

>>> l[2:5] = 100
TypeError: can only assign an iterable
```

最后一项的意思是，如果赋值时的左边是一个切片实例，那么它必须赋值为一个可遍历的实例（如`list, tuple`等），而不是一个数值。如果左边只是一个索引则可以：`l[2] = 100`，反倒是输入`l[2] = [100]`时，列表`l`会变成`[0, 1, [100], 30, 5, 8, 9]`。

## 2.7 序列中`+`和`*`的用处

这是一个常见的操作，即`list1+list2`或者`list1 * num`：

```python
>>> a = [1,2,3]  
>>> b = [4,5,6]  
>>> a+b
[1, 2, 3, 4, 5, 6]

>>> a * 5 
[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
```

但是list是元素的引用，而不是元素本身，所以：

```python
>>> list_example = [[]] * 3  
>>> list_example[0].append(3)  
>>> print(list_example)
[[3], [3], [3]]

# equivalent to:

# row = []
# board = []
# for i in range(3):
#     board.append(row)
```

### 建立列表的列表

为了解决上面的问题，我们应当使用列表推导式(list comprehension)：

```python
>>> list_example = [[] for _ in range(3)]  
>>> list_example[0].append(3)  
>>> print(list_example)
[[3], [], []]

# equivalent to:

# board = []
# for i in range(3):
#     row = []
#     board.append(row)
```

问题代码中的母列表本质上是`[la,la,la]`，三个子列表指向同一个列表实例；
改正后代码的母列表是`[la,lb,lc]`，list comprehension生成了三个互相独立的列表实例。

### 序列的增量赋值

> 序列的`+=`和`*=`

对各类sequence object实现`+=`操作的魔术方法为`example.__iadd__()`，"in-place addition"（就地加法），如果没有`__iadd__`则会调用`__add__`方法。对应的，`*=`调用的是`__imul__`（就地乘法）。

> 调用`__iadd__`的`a += b`类似于`a.extend(b)`，而使用`__add__`的则类似于`a = a + b`。标准库中的可变序列基本都支持`__iadd__`，而不可变序列根本就不支持这类操作。除了`str`，尽管是不可变序列，但因为其`+=`操作的常用性，CPython专门为其做了适配。

```python
a = [1,2,3]  
print(id(a))  
a *= 2  
print(id(a))
# results:
# 2532556203840
# 2532556203840
# meaning that it's an in-place operation
  
t = (1,2,3)  
print(id(t))  
t *= 2  
print(id(t))
# results:
# 2532543679168
# 2532558504384
# meaning that a new tuple is created
```

之后，书中讨论了一种少见的情况，即tuple种含有list时，对list做`+=`操作，结果是在抛出*TypeError: 'tuple' object does not support item assignment* 的同时，实现了对list的原地加法。作者在这里总结了三条教训：

- 不要把可变对象放到元组里面
- 增量赋值不是原子操作(atomic operation)
- 异常出现后，多看字节码（`dis.dis[expression]`）

## 2.8 排序：list.sort方法和内置函数sorted

对序列的排序可以用`list.sort`方法实现，也可以用内置函数`sorted`实现。前者原地排序，后者生成一个排序好的新序列。

```python
import random  
a = random.choices(range(100), k=10)  
# print a and its id, with format "list {a}, with id {id(a)}"  
print(f"list {a}, with id {id(a)}")  
  
# sort a and print it  
a.sort()  
print(f"list {a}, with id {id(a)}")  
print(a.sort())

# results:
# list [23, 99, 68, 64, 8, 21, 65, 5, 9, 3], with id 2532558018112
# list [3, 5, 8, 9, 21, 23, 64, 65, 68, 99], with id 2532558018112
# None

# 最后一行表明，a.sort()方法并不生成任何新的对象
```

这两个方法是接近的，都有两个可选参数：`reverse`和`key`。前者控制降序输出，默认为`False`，如果为真则按从大到小排序；`key`用于控制对比时的规则，特别是在字符串排序时，可以用`key = str.lower`来忽略大小写，或者`key = len`来实现基于字符串长度的排序。

> 如果用`key = len`，而序列中有两个同样长度的字符串，那么这两个字符串按照原序列中的先后顺序排序。

### 利用`bisect`库来管理有序序列

> 第一版章节2.8，这一部分已经被第二版剔除，变成线上的可选材料。

对于有序序列（不仅是列表）来说，`bisect.bisect`可以用来进行搜索：`bisect(haystack, needle)`。其中，`haystack`是待搜索序列，`needle`是需要搜索的值（或值的序列）。搜索得到结果的原则是：将`needle`插入输出的索引位置上，`haystack`仍然能保持升序。

但不要用`bisect()`得到的索引再做`haystack.insert(index, needle)`操作，`bisect.insort(haystack, needle)`的速度更快。插入`needle`后能自动保持原序列`haystack`是升序的。

## 2.9 如果列表不是首选项

列表灵活但未必最有效率。例如，对大量浮点数的存储使用`array`，大量先进先出的操作可以使用`collections.deque`（双端序列）等，如果需要大量地检测序列中是否存在某个元素，可以用专门优化过的`set`（集合）类型。

### `array.array`类型

array类型自Python 3.4后不支持`example.sort()`方法，排序用`sorted()`：

```python
# expression
a = array.array(a.typecode, sorted(a))

# example
>>> import array  
>>> a = array.array("i", range(10)) + array.array("i", [21,17,19])  

# WRONG: array.sort
>>> a.sort()
# AttributeError: 'array.array' object has no attribute 'sort'

# CORRECT operation: sorted()
>>> a = array.array('i', sorted(a))  
>>> print(a)
array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19, 21])
```

在读取上，`array.tofile()`和`array.fromfile()`在纯浮点数的二进制文件IO上速度要快于文本文件IO，并且能够有效降低所需空间。

### Memory View

内存视图(Memory View)类可以在不复制序列内容的情况下操作array的多个切片。

这里介绍的主要方法是`memoryview.cast()`，它将array中每个元素的内存形式做成序列，但这个序列仍然与原array挂钩。当我修改这个序列的二进制（或十进制）内容时，原array的内容也对应做出了修改，参照示例2-21。

> 用memoryview.cast生成的切片做出的操作都是和原序列同步的。

### Numpy中的数据结构

直接去看Numpy的文档或其他资料吧，这里就是一个简单的介绍。

### Double-end queue (deque)

用list自然是可以实现栈或者队列的，就是`append`和`pop`两个list中内置方法的应用，区别只在于为了实现先进先出，`queue.dequeue`用的是`pop(0)`（打印并删除第一个元素），而`stack.dequeue`后进先出用的是`pop()`（打印并删除最后一个元素）。

但是，`collections.deque`可以更快速地在两端添加或删除元素：

```python
from collections import deque  
dq = deque(range(10), maxlen=10)  
print(dq)  
# deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], maxlen=10)

dq.rotate(3)  
print(dq)  
# deque([7, 8, 9, 0, 1, 2, 3, 4, 5, 6], maxlen=10)

dq.appendleft(-1)  
print(dq)  
# deque([-1, 7, 8, 9, 0, 1, 2, 3, 4, 5], maxlen=10)

dq.extend([11,12,13])  
print(dq)  
# deque([9, 0, 1, 2, 3, 4, 5, 11, 12, 13], maxlen=10)
  
dq.extendleft([10,20,30,40])  
print(dq)
# deque([40, 30, 20, 10, 9, 0, 1, 2, 3, 4], maxlen=10)
```

- 定义时的`maxlen`参数直接规定了`deque`类型实例的最大长度，
- 如果从一个方向加入了超量的元素，那么就会从另一个方向上把多出来的元素挤出去。例如`dq.appendleft(-1)`从左边压入`-1`，然后挤出了最右边的元素`6`。
- `extendleft`会逐个把遍历体中的元素从左到右压入deque中，因此在最后一行可以看到，`[40,30,20,10]`在`dq`中实际上是从右到左的。

`deque`对头尾操作进行了额外的优化，而对序列中间的增删改问题则具有劣势。

此外，这本书还简单提到了`queue`,`multiprocessing`,`asyncio`,`heapq`这四个能够实现队列数据结构的python标准库。
