---
comments: true
---

# 关于Mkdocs和Github的联动

核心的问题在于如何让Git和Github连起来，报超时的Fatal报到傻，目前网上给出的解决方案均无效。

好吧，不用vs code自带的git套件，而是用github desktop，其实啥问题都没有。

---

网站的第一次编译要在Shell上跑一次：

```
mkdocs gh-deploy
```

从而在网站仓库中建立一个 `gh-pages`的分支。

网站仓库的 `Page`设置中，`Branch`设置成gh-pages，后面不要动，设成根目录是对的。


以后，每次内容编辑完成并更新网站时，只需要commit and push（甚至都不需要本地build），剩下的编译和实装就交给github actions。

> 这套操作可能会有30秒的时延，但相对方便。
