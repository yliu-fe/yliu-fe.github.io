# 关于Mkdocs和Github的联动

核心的问题在于如何让Git和Github连起来，报超时的Fatal报到傻，目前网上给出的解决方案均无效。

---

Github page repository的 `Page`设置中，`Branch`设置成gh-pages下的 `.\Docs`。

在本地完成编辑后做如下操作：

```shell
mkdocs build
```

然后通过git推送上去，但经常会出现的问题是，由 `ci.yml`所规定的自动工作流往往正常，而该工作流结束后跟随着的 `pages build and deployment`会在build阶段报错。
