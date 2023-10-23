项目要求：[Project 1：Search CS 188: Introduction to Artificial Intelligence, Fall 2018 (berkeley.edu)](https://inst.eecs.berkeley.edu/~cs188/fa18/project1.html)

编辑内容仅限于`search.py`和`searchAgents.py`，需要阅读`pacman.py`,`game.py`和`util.py`。

下面直接开始。

## Question 1 - DFS
> Finding a Fixed Food Dot using Depth First Search
> 利用深度优先搜索寻找一个固定位置的豆子

任务：实现`search.py`下的函数`depthFirstSearch(problem)`

实现过程：
首先，回顾DFS的思想。我们需要首先直到初始位置在哪，这不需要我们自行定义，而是调用该函数参数`problem`中的API，“getStartState”，不难理解。同时，DFS是一个先进后出的算法，因此需要一个栈，`util.py`已经为我们定义了这样的数据结构，然后，把出发状态`start`压进栈中。
> 这里传入栈中的是`(start, [])`元胞，第一项是当前状态位置，第二项指代的是path，即从start到该节点的路径，显然，start到start并不需要任何路径。
```python
start = problem.getStartState()  
stack = util.Stack()  
stack.push((start, []))
```
另外定义一个集合，用以存储那些已经探索过的节点：
```python
visited = set()
```
剩下的事情都是循环：
- 如果栈中最先拿出的（最后存入的）节点是目标节点，算法结束并传回结果——结果就是元胞`(node, path)`中的`path`
- 如果不是目标节点，则将其直接子节点加入栈中，重复以上目标。
- 如果不是目标节点且没有未探索过的直接子节点，则回退到上一级，表现为这个while循环中，从堆栈中将当前节点pop出来扔掉以外，什么都没发生，重新判断上层节点。
```python
while not stack.isEmpty():  
    node, path = stack.pop()  
    if problem.isGoalState(node):  
        return path  
    if node not in visited:  
        visited.add(node)  
        child, action, cost = problem.getSuccessors(node)  
        for i in range(len(child)):  
            stack.push((child[i], path + [action[i]]))
```

以下图的树为例（来自Wikipedia），来讲述搜索过程中的出入栈操作。假定我们要找的是节点4
![[Depth-first-tree.svg.png]]
1. 一开始，栈是空的，状态`start`是1，且`visited`列表是空的，表示没探索过任何节点。
2. 首先从当前的节点1开始，将该点及其路径`{1,[]}`压入栈。然后开始判断，又要把这个点pop出来，显然，节点1没有探索过，也不是目标节点，所以将1写入visited，表示已经探索过了，然后通过`getSuccessors`方法获取节点1的后续节点，即2、7、8。以节点2为例，实际获取的应当是`{2,[1,2]}`，既表示了2的名称，又写明了与初始节点1的路径关系。2、7、8同时被压入栈（栈中只有他仨），进入下一次的循环
3. 按照先进后出的原则，下一轮从stack.pop中读出的其实是节点8，但对于DFS来说，同一层级的节点先后顺序不重要，所以看起来是倒着来的。8不是目标节点，但也没有探索过，所以和第二步一样，将其写入`visited`，并压入其子节点9、12。现在栈中按顺序为2、7、9、12，下一轮判断12。
4. 12不是目标节点、没有（没有被评估过的）子节点，因此pop掉就pop掉了，栈中剩下2、7、9，下一轮评估9，与步骤2、3都一样。
5. 评估9，压入10、11；10、11分别评估后出栈，栈中只剩2、7。而8的所有子树都已评估过了，又回到了一级子节点。
6. 评估7，然后再评估2的那棵子树，就和前面的步骤一样，直到找到节点4，返回其路径`[1,2,3,4]`，算法结束。

## Question 2 - BFS
> Implement the breadth-first search (BFS) algorithm in the `breadthFirstSearch` function in `search.py`. Again, write a graph search algorithm that avoids expanding any already visited states. Test your code the same way you did for depth-first search.
> 在`search.py`文件中的`breadthFirstSearch`函数中实现广度优先遍历（BFS）算法。一样的，写出避免评估任何已经评估过状态的图搜索算法，用与DFS题目一样的方式测试你的代码。

任务：实现图的广度优先遍历算法。

实现过程：
与DFS不同的是，BFS并不适合使用栈来实现。