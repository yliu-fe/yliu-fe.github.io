# 关于Mkdocs和Github的联动

核心的问题在于如何让Git和Github连起来，报超时的Fatal报到傻，目前网上给出的解决方案均无效。

---

Github page repository的 `Page`设置中，`Branch`设置成gh-pages下的 `.\Docs`。

在本地完成编辑后做如下操作：

```shell
mkdocs build
mkdocs gh-deploy
```

然后通过git推送上去，本以为这样就可以解决问题，结果还需要在仓库 `Action`选项下，把工作流重做一遍，才能将网站推送成当前的状态，多少有点乐。
