---
comments: true
---

# 在MkDocs中使用MathJax

为什么要用这个，那当然是我要写数学式子......

参考：[Python Markdown Extensions - Material for MkDocs (squidfunk.github.io)](https://squidfunk.github.io/mkdocs-material/setup/extensions/python-markdown-extensions/#arithmatex)

/// admonition | 修订：2024年5月
    type: bug

参考文档（见页首）的cdn链接已经修订，本文给出的cdn也随之修订。修改的是第二个cdn链接，变成了unpkg.com的链接。

同时，`mathjax.js`文件中的`document$.subscribe`项目中需要加入的命令从一条变为四条。
///

首先，在 `docs`即文档体系下增设 `mathjax.js`文件，具体放在哪都行，其内容为：

```javascript title="mathjax.js"
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})
```

我日常使用的行内、行间公式是 `$...$`和 `$$...$$`，这里也一样，但是mathjax会将其转换为`\(...\)`和`\[...\]`，这件事你不用管，让mathjax自己弄就可以了。

其次，在mkdocs.yml即网站的样式文件中，顶格添加以下内容：

```yaml
markdown_extensions:
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js
```

它们不是theme的子项目，如果带缩进的话是没法被编译的......此外，`extra_javascript`中的mathjax CDN网址是个谜，具体能不能用我不是很明白，但Material for MkDocs文档中给出的是以上内容，我的个人网站里用的也是这个配置。

按说，这两样东西加上了，就应该能渲染出TeX样式的数学内容，例如下面的式子：

$$
r_{t} = \mu_t + a_t = E(r_t|F_{t-1}) + \sigma_t \epsilon_t
$$

其中，$r_t$是观测值序列，$\mu_t$是可观测部分，$a_t$是扰动项（不可观测部分）。$F_{t-1}$为第$t$期时能够观测到的已知信息集$\{ r_1, r_2,...,r_{t-1}\}$。扰动项被假定为两项之积，$\sigma_t$是条件标准差，$\epsilon_t$是独立同分布且服从标准正态分布的随机变量（白噪声）。

//// admonition | 行间公式需要与内容隔开
    type: warning

务必注意，行间公式需要与内容隔开，否则会出现渲染错误。例如上面的行间公式，应当写为：
  
```markdown
按说，这两样东西加上了，就应该能渲染出TeX样式的数学内容，例如下面的式子：

$$
r_{t} = \mu_t + a_t = E(r_t|F_{t-1}) + \sigma_t \epsilon_t
$$

其中，$r_t$是观测值序列，$\mu_t$是可观测部分，$a_t$是扰动项（不可观测部分）。
```

否则，你看到的就会是公式的代码形式，而非你想得到的，渲染后的样子，**以下为错误示范**。

/// admonition | 错误示例
    type: bug

按说，这两样东西加上了，就应该能渲染出TeX样式的数学内容，例如下面的式子：
$$
r_{t} = \mu_t + a_t = E(r_t|F_{t-1}) + \sigma_t \epsilon_t
$$

///

////
