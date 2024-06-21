---
comments: true
---

# Python Data Model

/// admonition | 图书信息
        type: info

书名：Fluent Python: Clear, Concise, and Effective Programming (2nd Edition)

作者：Luciano Ramalho

此为本书第二版，对应Python 3.10（第一版对应Python 3.4）。

第一版的汉译本《流畅的Python》由人民邮电出版社出版，ISBN: 9787115454157。
///

第一章讲了一些简单的Python数据模型【不是数据结构】。

## 魔术方法

利用`collections.namedtuple`可以简单生成一些命名的类示例，例如：

```python
import collections

Card = collections.namedtuple('Card', ['rank','suit'])

>>> beer_card = Card('7','diamonds')
>>> beer_card
Card(rank='7',suit='diamonds')
```

通过魔术方法（magic method）可以将一些系统定义的方法（例如`len()`等）的实现方式转变为类内的方法，例如：

```python
class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
```

其中，`__init__`是最常见的魔术方法，对于每个类而言，它必不可少，因为它负责定义类实例所需要的全部信息；

`__len__`方法替代了标准库的`len()`，对于一个`FrenchDeck`的实例`deck = Frenchdeck()`，`len(deck)`不是传统的返回数据结构中元素数量的标准函数，而是代码中使用的`len(deck._cards)`，即返回类中所规定的私有变量`_cards`这个列表中元素的数量，显然，根据上面`ranks`和`suits`的定义，这是一副没有王牌的扑克，共52张。

> 对于其他的，不是`FrenchDeck`实例的变量，`len(example)`就还是那个标准的`len()`，不会调用这个`len(self._cards)`

而`__getitem__`调用了索引，即`example[]`。显然，作为一个类的实例，`deck`不能被直接索引，直接用`deck[0]`会被报错，但是`__getitem__`将索引转化为实例下面`_cards`这个列表的索引，所以得到的是一组牌的名字，例如：

```python
>>> deck[0]
Card(rank='2', suit='spades')

>>> deck[-1]
Card(rank='A', suit='hearts')
```

尽量不要直接调用魔术方法，例如`deck.__len__()`就很蠢，可以直接用`len(deck)`。同时，标准的Python内置类型也大量地使用了魔术方法，可以通过`dir(int)`来查看整型class中定义或使用了哪些方法（包括魔术方法）。

/// admonition | 魔术方法不能被自定义
        type: warning

魔术方法的名字（如`__init__`,`__len__`等）是标准库里的**内置函数**，它们各自对应于一个常见的方法（如`len`、索引等），用户可以通过这些给定的魔术方法来决定哪些常见的函数如何应用于这个类的实例上，但不能自己凭空创造一个新的`__example__`方法并挂钩于任何的标准python方法。

或者说，可以自行创造一个，但它不会挂钩于任何的现有方法，使用时只会被当作一个普通的方法调用，也即`class.__example__()`，尽管它的名字以魔术方法特有的双下划线开头。

而且，尽量不要想当然地自行创造这样的方法，因为未来Python说不定会在标准库中加入这个魔术方法，就会引发代码的冲突。
///

关于Magic Method的更多讲解可参照以下博文：
（中文）[python魔法方法长文详解 - 个人文章 - SegmentFault 思否](https://segmentfault.com/a/1190000040286979)
（英文）[A Guide to Python's Magic Methods « rafekettler.com (rszalski.github.io)](https://rszalski.github.io/magicmethods/)

## 特殊方法如何使用

> Chapter 1.2

进一步，讲了更多的魔术方法，并分别从数学、字符串、布尔代数和Collection API角度讨论魔术方法的应用。

### 1. 数学和字符串

考虑一个二维向量的表示

```python
import math  
  
class Vector:  
    def __init__(self, x=0, y=0):  
        self.x = x  
        self.y = y  
        
    def __repr__(self):  
        return f'Vector({self.x}, {self.y})'  
          
    def __abs__(self):  
        return math.hypot(self.x, self.y)  
        
    def __bool__(self):  
        return bool(abs(self))  
        
    def __add__(self, other):  
        x = self.x + other.x  
        y = self.y + other.y  
        return Vector(x, y)  
        
    def __mul__(self, scalar):  
        return Vector(self.x * scalar, self.y * scalar)
```

一共涉及了六个魔术方法，其中`__init__`之前讨论过。

`__repr__`用于展示`Vector`类实例，并将其表示为一个表示向量的字符串，同样是`v1 = Vector(2,4)`的定义，同样是`print(v1)`，定义或不定义该魔术方法有较大的区别：

```python
# define __repr__
>>> print(v1)
Vector(2, 4)

# without __repr__
>>> print(v1)
<__main__.Vector object at 0x000001A91E8BDC90>
```

不定义`__repr__`时，编译器返回的是这个实例是什么类，以及它所在的内存位置。这是`print`函数对类实例的标准输出。

> 通常对`__repr__`定义输出的要求是，用于生成实例的`eval(输出项)`函数可以直接复制一个新的，完全相同的类实例出来。

- `__abs__`绑定`abs()`函数，从而使其返回向量的长度；
- `__bool__`绑定的是`if-else`或`while`中的逻辑判断。在定义后，如果向量的长为0，则返回`False`，否则返回为`True`；而如果没定义它，则对于所有的情况来说，`Vector`实例都会被视作`True`，无论其长度。关于真值检验的内容可以参阅Python文档[内置类型 — Python 3.12.4 文档](https://docs.python.org/zh-cn/3/library/stdtypes.html#truth)，简单来说，`None,False`、各类空集、各类数值0都会被当作`False`。
- `__add__`和`__mul__`分别顶替了加法和乘法运算符(`+,*`)从而实现向量的相加和相乘。

/// admonition | 关于`__add__`和`__mul__`
        type: note

示例中的add和mul方法的实现形式都是生成一个新的向量`return Vector(xxx,xxx)`，而不是直接在原有加数或者乘数上修改，这是一个中置运算符（infix operators)的良好特性。
///

由于字符串部分只在讨论`__repr__`和`__str__`，合并在这里，它们基本上都是定义了类实例的字符串表示，通常选择`__repr__`不会有错，因为它被要求返回详细的、无歧义的字符串表示。
