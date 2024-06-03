---
comments: true
---
# 基于Astro框架构造网页并将其部署到Github Pages

参考文档（中文）：[部署你的 Astro 站点至 GitHub Pages](https://docs.astro.build/zh-cn/guides/deploy/github/)

这个文档将会记载两部分内容：第一，我需要构造一个基于Astro的网页结构；第二，我需要将其部署到Github Pages上，
但是注意到，您现在所看到的这个基于Mkdocs框架的网站已经占据了我的Github Pages网址的位置，所以新的网站只能在根网址
下面开一个子目录。以下展示均来自我对另一个仓库的折腾实况。

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

还是那件事，这个Astro是我第二个托管在Github Pages上的网站。对应于我用户名的`github.io`主域名已经被Mkdocs占据，那个Astro的
实验性网站就只能以项目网站的身份出现：`<username>.github.io/<repository>`。这会在接下来的设置中带来一点小小的坑，但无伤
大雅。

需要做的设置分为三步：
- 设置`astro.config.mjs`文件；
- 设置Github Pages；
- 构造并设置Github Actions。

### 设置`astro.config.mjs`文件

在Astro项目的根目录下，有一个`astro.config.mjs`文件，其中包含了一些关于网站的配置信息。在这个文件中，您需要添加一些内容：

```javascript title="astro.config.mjs"

import { defineConfig } from 'astro/config'

export default defineConfig({
  site: 'https://<USERNAME>.github.io',
  base: '/my-repo',
})

```

/// admonition | 关于`site`字段
    type: warning

其中，`site` 字段是您的发布域名。如果您购买了其他的域名，这里请填写您所购买的域名。

否则，这里填Github Pages**主网址**，无论您的网站直接以这个网址发布，还是以项目网站`<username>.github.io/<repository>`的形式发布
，这里都要填`https://<USERNAME>.github.io`，而不是`https://<USERNAME>.github.io/<repository>`！
///

/// admonition | 关于`base`字段
    type: note

`base`字段填写的是您希望网站具体发布的位置。以我的`AstroNamie`为例，我希望它发布在`<username>.github.io/AstroNamie`下。前面的`Site`
字段中我填写了`https://<username>.github.io`，而在这里填写`/AstroNamie`。

如果您发布的是个人网站，那么这里填写`/`即可。但是个人网站所需要的仓库名称应当是`<username>.github.io`。

如果您的网站是以项目网站的形式发布，那么这里填写的应当是`/<repository>`。

如果您自行购买了域名，且希望网站发布在根目录下，那么这里填写`/`。
///

如果您自行购买了域名并希望在其上发布的话，您需要额外在Astro网站仓库的文件夹`public`中构造文件`CNAME`，其中填写您的域名。

```plaintext title="./public/CNAME"
example.com
```

### 设置Github Pages

在Github仓库的`Settings`中，找到`Pages`选项卡。在“build and deployment”项目下的“Source”中选择“Github Actions”。

如果您自行购买了域名，请在下面的Custom Domain处填写域名并保存，等待DNS检查通过后选择`Enforce HTTPS`。

### 构造并设置Github Actions

最后一步是构造Github的自动发布功能。该自动化流程调用了Astro官方维护的Github action，将自动检测仓库中的新提交并将其更新到网站上。

在仓库根目录下新建文件夹`.github`，并在其中新建文件夹`workflows`，在其中新建`deploy.yml`文件如下：

```yml title=".github/workflows/deploy.yml" hl_lines="7"

name: Deploy to GitHub Pages

on:
  # 每次推送到 `master` 分支时触发这个“工作流程”
  # 如果你使用了别的分支名，请按需将 `master` 替换成你的分支名
  push:
    branches: [ master ]
  # 允许你在 GitHub 上的 Actions 标签中手动触发此“工作流程”
  workflow_dispatch:

# 允许 job 克隆 repo 并创建一个 page deployment
permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout your repository using git
        uses: actions/checkout@v4
      - name: Install, build, and upload your site
        uses: withastro/action@v2
        # with:
          # path: . # 存储库中 Astro 项目的根位置。（可选）
          # node-version: 20 # 用于构建站点的特定 Node.js 版本，默认为 20。（可选）
          # package-manager: pnpm@latest # 应使用哪个 Node.js 包管理器来安装依赖项和构建站点。会根据存储库中的 lockfile 自动检测。（可选）

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4

```

可能有很多人像我一样，第一次commit的时候将其默认分支命名为master，并且之后一直在该分支上更新。在网站发布时，请务必检查该yml文件
中的`branches`位置(被高亮的那一行代码)是否填对了，否则github action会无法执行——因为你仓库里压根就没有main分支。
