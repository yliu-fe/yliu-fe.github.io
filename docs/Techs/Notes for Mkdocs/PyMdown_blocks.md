# PyMdown可用的"block"样式

MkDocs支持PyMdown及其Extension，从而可以使用一系列的“block”，即模块化内容渲染。

## 1. 现有的Detail样式及实现方法

参考：[Details - PyMdown Extensions Documentation](https://facelessuser.github.io/pymdown-extensions/extensions/details/)

如以下内容：

???+ success "Test"
    There is nothing at all.
    ??? warning "Test 2"
        A warning?

其对应的代码为：

```markdown
???+ success "Test"
    There is nothing at all.
    ??? warning "Test 2"
        A warning?
```

在MkDocs框架中使用该模板的方法是在 `mkdocs.yml`文件中顶格加入：

```yaml
markdown_extensions:
  - pymdownx.details
```

显然，这是一个依靠缩进表示从属关系的表达方式。`???`是detail内容的引导，`???+`与前者的区别则在于，在网页渲染后，以后者引导的detail模块将默认打开（如其中的Test），而前者引导的将默认关闭（如其中的Test 2）。其结构可定义为：

```markdown
??? class "Topic"
    There is content
```

其中，`Topic`需要被双引号框起来，而 `class`有几种选择，如：
???+ note "Choice for `class`"
    There are multiple choices for `class` option, such like:
    ??? note "note"
        I'm `??? note "Note"`
    ??? success "success"
        I'm `??? success "success"`
    ??? danger "danger"
        I'm `??? danger "danger"`
    ??? warning "warning"
        I'm `??? warning "warning"`

## 2. PyMdown Extensions 9.10对该样式的重构

参考：[Index - PyMdown Extensions Documentation](https://facelessuser.github.io/pymdown-extensions/extensions/blocks/)

根据项目方的介绍，`block`系统将在PyMdown Extensions 9.10版本重构，在实现效果不变的情况下重构语法，新的语法结构为：

```markdown
/// details | Some summary
Some content
///
```

并实现如下结果

/// details | Some summary
    type: warning

Some content
///

为了使重构的语法生效，需要在 `mkdocs.yml`中的 `markdown_extensions`板块按需加入：

```yaml
markdown_extensions:
  - pymdownx.blocks.admonition:
      types:
      - new # type 能填哪些项，从这里就能找到
      - settings
      - note
      - abstract
      - info
      - tip
      - success
      - question
      - warning
      - failure
      - danger
      - bug
      - example
      - quote
  - pymdownx.blocks.details:
  - pymdownx.blocks.html:
  - pymdownx.blocks.definition:
  - pymdownx.blocks.tab:
      alternate_style: True
```

//// details | 各类新blocks的样式和用途
    type: note
    open: True

(1) `pymdownx.blocks.details`是重构过的detail，即可以打开或缩回的块状内容，上面已经演示过了，这里就直接跳过。

(2) `pymdownx.blocks.admonition`意为“训诫”，是必然敞开的块状内容，如：
/// admonition | `admonition`样例
    type: question
为了做出这种版式，请使用如下代码：

```markdown
/// admonition | Some title
Some content
///
```

其中 `Some title`为该样例的版头名称，`admonition`说明该样例生成的是“admonition”式板块。
///

(3) `pymdownx.blocks.html`可以用block元素生成各类HTML效果，参照[HTML - PyMdown Extensions Documentation](https://facelessuser.github.io/pymdown-extensions/extensions/blocks/plugins/html/)。

(4) `pymdownx.blocks.definition`用于生成类似于函数文档中各参数定义的样式，如：

/// define
Apple

- Pomaceous fruit of plants of the genus Malus in
  the family Rosaceae.

///

该样例的代码为：

```markdown
/// define
Apple

- Pomaceous fruit of plants of the genus Malus in
  the family Rosaceae.

///
```

（5）`pymdownx.blocks.tab`生成的是带有标签按钮的内容，如：
/// tab | Tab 1 title
Tab 1 content
///

/// tab | Tab 2 title
Tab 2 content
///

其生成方法为：

```markdown
/// tab | Tab 1 title
Tab 1 content
///

/// tab | Tab 2 title
Tab 2 content
///
```

////

下面讨论三个问题：（1）如何规定admonition和detail的类型和默认开启关闭，（2）新block语法的嵌套实现，（3）如何实现两个连续的 `tab`格式。这里用了很多奇怪的type，是为了尝试花样，没有具体含义。

/// details | 1. 类型、默认打开和关闭
    type: example

（这里用的是type: example）

新语法下，类型和默认启闭是语法块的参数，在实现时写为：

```markdown
/// details | Some summary
    open: True # 不写或写成False视作默认关闭
    type: warning # 在这里修改模块的类型，例如note, warning, danger, example等

Some content
///
```

///

///// details | 2. 新语法下如何嵌套block
    type: info

(这里用的是type: info)

新版本的block舍弃了原有的缩进结构，因此，为了实现嵌套表示，新版本语法将 `///`作为开始和结束的符号，而母块包含子块的方式，就是将母块的 `///`写成 `////`，斜线越多，代表该块的包含优先级越高，但最少不能少于3条斜线。实现形式可以为：

```markdown
//// note | Some title
/// details | Summary
    type: warning
content
///
Content
////
```

其效果为：
//// note | Some title
/// details | Summary
    type: warning
content
///
Content
////
显然，承载本问题内容的block应该是五条斜线的。
/////

//// details | 3. 如何实现连续两个tab格式
    type: failure
(这里用的是type: failure)

为了实现两个单独的Tab组连续共存，则需要在两个tab组之间加强制隔离 `new: true`，如：

```markdown
/// tab | Tab A title
Tab A content
///

/// tab | Tab B title
Tab B content
///

/// tab | Tab C Title
    new: true

Will be part of a separate, new tab group.
///
```

实现结果为：

/// tab | Tab A title
Tab A content
///

/// tab | Tab B title
Tab B content
///

/// tab | Tab C Title
    new: true

Will be part of a separate, new tab group.
///

////
