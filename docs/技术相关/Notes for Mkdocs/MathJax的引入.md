# 在MkDocs中使用MathJax

为什么要用这个，那当然是我要写数学式子......

参考：[Python Markdown Extensions - Material for MkDocs (squidfunk.github.io)](https://squidfunk.github.io/mkdocs-material/setup/extensions/python-markdown-extensions/#arithmatex)

首先，在 `docs`即文档体系下增设 `mathjax.js`文件，具体放在哪都行，其内容为：

```javascript
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
  MathJax.typesetPromise()
})
```

我日常使用的行内、行间公式是 `$...$`和 `$$...$$`，但这里必须写成上面第3、4行的形式，因为mathjax的组件会将其转写为 `\( ... \)`和 `\[ ... \]`的形式。

其次，在mkdocs.yml即网站的样式文件中，顶格添加以下内容：

```yaml
markdown_extensions:
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - javascripts/mathjax.js #mathjax的相对目录（不用写/docs/）
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
```

它们不是theme的子项目，如果带缩进的话是没法被编译的......此外，`extra_javascript`中的mathjax CDN网址是个谜，具体能不能用我不是很明白，但Material for MkDocs文档中给出的是以上内容，我的个人网站里用的也是这个配置。

按说，这两样东西加上了，就应该能渲染出TeX样式的数学内容，例如下面的式子：

$$
r_{t} = \mu_t + a_t = E(r_t|F_{t-1}) + \sigma_t \epsilon_t
$$

其中，$r_t$是观测值序列，$\mu_t$是可观测部分，$a_t$是扰动项（不可观测部分）。$F_{t-1}$为第$t$期时能够观测到的已知信息集$\{ r_1, r_2,...,r_{t-1}\}$。扰动项被假定为两项之积，$\sigma_t$是条件标准差，$\epsilon_t$是独立同分布且服从标准正态分布的随机变量（白噪声）。
