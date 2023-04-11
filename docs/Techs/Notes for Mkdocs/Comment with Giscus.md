---
comments: true
---
# 利用Giscus实现评论功能

参考：[Adding a comment system - Material for MkDocs ](https://squidfunk.github.io/mkdocs-material/setup/adding-a-comment-system/)

可以利用Github Marketplace下的Giscus应用来配置网页的评论区，本质上是网站对应仓库的Discussion部分。

## 第一步：开启讨论区
在开始配置评论区之前，首先要打开网站仓库的讨论区（Discussion）板块，位置是：

```
Settings --> General --> Features --> 勾选Discussions
```

## 第二步：安装Giscus
去[GitHub Apps - giscus](https://github.com/apps/giscus)安装Giscus应用。此应用安装在Github账户下，可选择特定仓库生效或者全账户生效。

## 第三步：配置Giscus

移步至[Giscus主页](https://giscus.app/)，在“配置部分进行设置”：

/// admonition | 关于此处若干项设置的讨论
    type: note

1. 语言、仓库无需多言，仓库可配置的前提是完成了上面的第一步——开启仓库的Discussion。

2. “页面 <-> discussion映射关系”。中文网络中的相关笔记和教程均推荐第一项，即“Discussion 的标题包含页面的`pathname`”。

3. “Discussion分类”参考官方推荐，选择`anouncements`

    > （此设置有官方提醒）推荐使用公告（announcements）类型的分类，以确保新 discussion 只能由仓库维护者和 giscus 创建。

4. 主题请随意。但提醒一句，Github Light在网页采用暗色模式时仍保持亮色模式，显得很突出。下拉项中有“用户偏好的色彩方案”(`preferred_color_scheme`)，会自动适应网页亮暗色，尽管跟随转换时有肉眼可见的时延，但算是好事了。
///

完成设置后，在`启用giscus`项下把对应的配置代码复制一下，一会要用到，它的形式大概是：

```html title="giscus评论区样式代码" hl_lines="3 4 5 6"
<script
  src="https://giscus.app/client.js"
  data-repo="<username>/<repository>"
  data-repo-id="..."
  data-category="..."
  data-category-id="..."
  data-mapping="pathname"
  data-reactions-enabled="1"
  data-emit-metadata="1"
  data-theme="light"
  data-lang="en"
  crossorigin="anonymous"
  async
>
</script>
```
务必检查，您自己的样式代码中，以上各项内容（特别是被高亮的各行）均已填有内容。`data-repo`一项中填写的应当是自己代码仓库的实际名称。

/// admonition | 否则...
    type: warning

第一次复制样式代码时，我有几项内容忘了选，复制了一份空代码过来，实装时Giscus报错：
> An error occurred: giscus is not installed on this repository

所以请务必检查各项内容是否设定完整。
///

## 第四步：在网站仓库中实装评论区
参考：[Customization - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/customization/#extending-the-theme)中的`Extending the Theme`部分。

评论区模块属于Material for MkDocs中的附加模块，这类内容被放在仓库根目录下的`/overrides/`文件夹下，并在`mkdocs.yml`中添加：
```yaml
theme:
  name: material
  custom_dir: overrides # 上面两行都是以前就有的，把最后一行加进去，注意是theme下的子项
```
`/overrides`下可加入的内容相当之多，评论区属于其中的`/partials/`子类。综上所述，这里要干的事情是在如下的地址中创建如下的html文件：
```
./overrides/partials/comments.html
```
并在其中填入如下框架代码：

```html title="comments.html" hl_lines="3"
{% if page.meta.comments %}
  <h2 id="__comments">{{ lang.t("meta.comments") }}</h2>
  <!-- Insert generated snippet here -->

  <!-- Synchronize Giscus theme with palette -->
  <script>
    var giscus = document.querySelector("script[src*=giscus]")

    /* Set palette on initial load */
    var palette = __md_get("__palette")
    if (palette && typeof palette.color === "object") {
      var theme = palette.color.scheme === "slate" ? "dark" : "light"
      giscus.setAttribute("data-theme", theme) 
    }

    /* Register event handlers after documented loaded */
    document.addEventListener("DOMContentLoaded", function() {
      var ref = document.querySelector("[data-md-component=palette]")
      ref.addEventListener("change", function() {
        var palette = __md_get("__palette")
        if (palette && typeof palette.color === "object") {
          var theme = palette.color.scheme === "slate" ? "dark" : "light"

          /* Instruct Giscus to change theme */
          var frame = document.querySelector(".giscus-frame")
          frame.contentWindow.postMessage(
            { giscus: { setConfig: { theme } } },
            "https://giscus.app"
          )
        }
      })
    })
  </script>
{% endif %}
```
然后将高亮的那一行替代为刚才从giscus处复制的代码。自此，评论区功能就存在于网站之中，但在网页中打开这个功能需要单独设置。

## 第五步：在网页中打开评论区

这一步最简单，在网页的Markdown文件最开头的位置上加入yaml内容：
```yaml
---
comments: true
---

# （网页一级标题）
（网页具体内容）...
```
但要注意的是，给现有的网页开评论区，这个yaml内容要一页一页地手动去写。