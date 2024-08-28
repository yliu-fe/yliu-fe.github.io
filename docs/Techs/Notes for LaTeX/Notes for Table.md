# 实现表格的脚注

## 1. 先实现再说

参照[Add notes under the table](https://tex.stackexchange.com/questions/12676/add-notes-under-the-table) 下的高赞回答，使用`thereparttable`来完成，看下面的例子

```latex title="Example_1.tex"
\usepackage[flushleft]{threeparttable}

\begin{table}[ht]
\centering
\caption{Notations in the basic model}
\label{tab: notations model}
\begin{threeparttable}[b]
\begin{tabular}{llll}
\hline
\textbf{Notation}     & \textbf{Subperiod} & \textbf{Meaning}                                                                                                                                                                                       & \textbf{Exo/Endo/Act} \\ \hline
$t$                 & 0                  & Period order\tnote{1}                                                                                                                                                                                           & Exogenous             \\                                                                                                                                                             & Exogenous             \\
$N^{+}, N^{-}$      & 4                  & The measure of BSs and BDs\tnote{2}.                                                                                                                                                                            & Endogenous            \\
\end{tabular}
\begin{tablenotes}
\item [1] Note 1
\item [2] Note 2
\end{tablenotes}
\end{threeparttable}
\end{table}
```

相比于一般的表格，这里需要将`tabular`类型放置在`threeparttable`类型里。在`tabular`类型内加入`\tnote{1}`来标记脚注位置，并统一在`tabular`后面开设`tablenotes`存放脚注内容。而`tabular`和`tablenotes`都是`threeparttable`结构的子结构。

同时，tablenotes里也可以写成`\item 内容`的格式，但这种脚注没有标号，也不对应表内内容，一般作为表格的整体注释。

## 2. 脚注与表格等宽

脚注与表格等宽按说是天经地义的事情，但是在论文中，有时会遇到一张很窄的表却有着很长的注释：只展示一次实证检验的内容，但实证检验表格需要
完整的解读：用了哪些解释变量、用了哪些变量作为控制、设置了什么样的固定效应，以及显著性的标记注释，凡此种种，都需要表格下方的注释来解释。

但这样的话，`threeparttable`及其`tablenotes`就变得捉襟见肘了，注释必然和表格等宽，这使得编写文章时遇到了较大的困难。这里，我发现了一个
新的latex包：`tabularray`，这是一个相当全面的latex表格库。其他内容参见其CTAN页面：[tabularray](https://www.ctan.org/pkg/tabularray)。

在overleaf中，该包可以用以下方式引入：

```latex
\usepackage{tabularray}
```

对于我们已经用threeparttable写成的表格来说，我们可以用tabularray来实现脚注与表格不等宽的效果：

```latex title="Example_2.tex"
\begin{table}[h]
\caption{Investor Preferences}
\centering
\label{tab}
\begin{threeparttable}
\begin{tblr}{l|c}
(表格具体内容)
\end{tblr}

\begin{tablenotes}
\item Note: this table corresponds to the linear regression. Robust standard errors are displayed in parentheses below. *** for $p < 0.001$, ** for $p < 0.01$, * for $p < 0.05$ and `.' for $p < 0.1$.

\end{tablenotes}
\end{threeparttable}
\end{table}
```

注意，这里的表格本体不再使用`tabular`，而是`tblr`，这是tabularray的表格类型。在`tablenotes`中继续写注释，这样就可以达成效果。

> 不过，这可能只是tabularray对threeparttable的兼容，因为上面的方式并没有严格按照tabularray的标准来编写表格及其注释。如果需要更多的功能，可以参考tabularray的官方文档。
