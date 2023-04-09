# 代码块的配置

下面讨论如何配置网站中的代码块(code blocks)。

参考：[Code blocks - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/)

## 配置方法

首先，为了实现PyMdown对代码块的优化配置，需要在 `mkdocs.yml`中的 `markdown_extension`部分加入新内容：

```yaml
markdown_extensions:
  - pymdownx.highlight: #代码块代码高亮
      anchor_linenums: true #对代码块显示行号
      line_spans: __span 
      pygments_lang_class: true #显示代码所属语言
  - pymdownx.inlinehilite #文内代码的高亮显示
  - pymdownx.snippets # 应该是用不到 
  - pymdownx.superfences # 应该是用不到，也没看懂用来干什么
```

另外，在 `mkdocs.yml`的 `theme`部分下的 `features`子项加入：

```yaml
theme:
  features:
    - content.code.copy
```

可以使代码块支持复制。

## 使用样例

### 1. 代码块基本样例

````markdown
``` py
import tensorflow as tf
```
````

实现为：

```py
import tensorflow as tf
```

/// details | 怎么实现把“代码块的代码”放进代码块格式的
    type: note

这玩意和block的新方法是一个思路，把外层代码块写成四个小引号:

`````markdown
````markdown
``` py
import tensorflow as tf
```
````
`````

显然你看到的这个代码块是由五个小引号构成的，这个小引号同样要求不少于三个
///

### 2. 代码块命名

````markdown
``` py title="bubble_sort.py"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```
````

实现为:

```py
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

前面说看不懂用来干什么的 `superfences`或者 `snippets`在此处起到了作用，所以还是加上吧。

### 3. 加入活动注释的代码

英文名为“code block with annotation”，即替代原有代码中的大块注释的项目：

```yaml
theme:
  features:
    - content.code.annotate # (1)
```

1. I'm a code annotation! I can contain `code`, __formatted
   text__, images, ... basically anything that can be written in Markdown.

其实现方法为：

````markdown
``` yaml
theme:
  features:
    - content.code.annotate # (1)
```

1.  I'm a code annotation! I can contain `code`, __formatted
    text__, images, ... basically anything that can be written in Markdown.
````

### 4. 显示代码块的行号

实现样例如：

```py
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

其代码为：

````markdown
``` py linenums="1" 
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```
````

其中 `linenums = <input>`，`<input>`中输入第一行的行号。

### 5. 代码块中高亮显示部分行

使用 `hl_lines`作为设置，但该参数默认从1开始数行号，无视 `linenums`对起始行号的规定。

````markdown
``` py hl_lines="2 3" linenums="1"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```
````

实现为：

```py
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

## 其他个性化设置

该框架还可提供配色和活动注释的个性化设置

参见：[Code blocks - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/#customization)的“Customization”部分。
