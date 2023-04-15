# 实现表格的脚注

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