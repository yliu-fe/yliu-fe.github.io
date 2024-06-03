---
comments: true
---
# 基于Astro框架构造网页并将其部署到Github Pages

参考文档（中文）：[部署你的 Astro 站点至 GitHub Pages](https://docs.astro.build/zh-cn/guides/deploy/github/)

这个文档将会记载两部分内容：第一，我需要构造一个基于Astro的网页结构；第二，我需要将其部署到Github Pages上，
但是注意到，您现在所看到的这个基于Mkdocs框架的网站已经占据了我的Github Pages网址的位置，所以新的网站只能在根网址
下面开一个子目录。以下展示均来自我对另一个仓库[[AstroNamie](https://github.com/yliu-fe/AstroNamie)]的折腾实况。

## 本地构建Astro框架

有兴趣使用Astro构造网页的话，可以跟着官方文档中的教程走一遍：[搭建你的 Astro 博客](https://docs.astro.build/zh-cn/tutorial/0-introduction/)。

与Mkdocs Material需要自行安装Python库的方式不同，Astro通过`npm`等管理器提供了自动化流程，形成一个比较完整的Astro网站仓库：

/// tab | npm

```bash
npm create astro@latest
```

///

/// tab | pnpm

```bash
pnpm create astro@latest
```

///

/// tab | yarn

```bash
yarn create astro
```

///

该过程将提供安装向导，您可以从中设置项目文件夹位置、选择是否使用模板库、是否自行编写TypeScript、是否完成git初始化等。
在此基础上，您可以利用`npm run dev`命令在本地启动一个开发服务器，查看您的网页效果（默认本地网站位于localhost:4321）。

## 部署到Github Pages

Astro支持多种网页托管服务，这里仅就Github Pages进行说明。

