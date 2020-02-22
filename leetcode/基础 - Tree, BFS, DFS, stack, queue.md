# 树的基础

## 前提知识

### Queue

* 队列，先进先出，后进后出
* python 里 queue 中的Queue支持此结构
  * Queue.put(a) 加入
  * Queue.get() 取出, 类似 pop
  * Queue.empty() 
  * Queue.qsize()

### Stack

* 栈， 先进后出
* 可以使用python里面的list实现
  * pop
  * append
  * 都是对最后的值操作

### others

* random.sample 无放回
* rnadom.choices 有放回

## BFS

* 算法大概
  * 使用队列存待搜索的节点
  * 每次搜索一层
  * 每个节点存入非零子节点，访问值
* 又称层序遍历
* 可以递归
  * 用dict维护每层的访问
  * 遍历访问下一层，层数+1

## DFS

* 先序的实际顺序：根左右 PreOrderTraversal
* 中序的实际顺序：左根右 InOrderTraversal
* 后序的实际顺序：左右根 PostOrderTraversal
* 递归三个方式都简单，输出根节点的值位置变化而已
* 非递归复杂，采用stack
* 非递归Pre:
  * 算法
    * 从根节点出发
    * 访问当前节点
    * 如果有右节点，push
    * 如果有左节点，push
    * pop一个当成当前节点
  * 因为后进先出，所以先push右再左

* 非递归In:
  * 算法
    * 如果当前节点存在，一直向左，push节点到stack中
    * 如果不存在，pop一个节点，访问当前节点，当前节点设为右节点  
  * 解释
    * 这样输出的节点是左节点，因为每次都是到最左边不能向左时才开始返回
    * 如果访问到a.right，a 已经pop掉了，从a.right返回时就跑到a.parent上了
* 非递归Post:
  * 可以将Inverse-pre（右左的方式）的结果逆序
  * 直接的算法：
    * 当前非None则push，一直向左
    * 当前节点变成最后访问的节点，如果没有右节点或者右节点已经被访问（访问过的记为previous, 右节点是否被访问只需判断上次访问的是否是这次的右节点），访问该节点，当前设为None，pop掉一个节点 ；当前节点设为右节点
  * 对于实际的根，需要保证先后访问了左子树、右子树之后，才能访问根。实际的右节点、左节点、根节点都会成为“左”节点入栈，所以我们只需要**在出栈之前，将该节点视作实际的根节点，并检查其右子树是否已被访问**即可。

## 代码

### 准备

构建随机树，画图

```Python
import random
random.seed(1)
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(G, node, pos={}, x=0, y=0, layer=1):
    pos[node.value] = (x, y)
    if node.left:
        G.add_edge(node.value, node.left.value)
        l_x, l_y = x - 1 / 2 ** layer, y - 1
        l_layer = layer + 1
        create_graph(G, node.left, x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node.right:
        G.add_edge(node.value, node.right.value)
        r_x, r_y = x + 1 / 2 ** layer, y - 1
        r_layer = layer + 1
        create_graph(G, node.right, x=r_x, y=r_y, pos=pos, layer=r_layer)
    return (G, pos)

def draw(node):   # 以某个节点为根画图
    graph = nx.DiGraph()
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(8, 10))  # 比例可以根据树的深度适当调节
    nx.draw_networkx(graph, pos, ax=ax, node_size=300, node_color = 'pink')
    plt.show()

class node:
    def __init__(self,x):
        self.value = x
        self.left = None
        self.right = None
       
def randomTree(nums):
    def addLeaf(root,t):
        if random.randint(0,1) == 0:
            if not root.left:
                root.left = node(t)
            else:
                addLeaf(root.left,t)
        else:
            if not root.right:
                root.right = node(t)
            else:
                addLeaf(root.right,t)
    if len(nums) == 0:
        return None
    root = node(nums[0])
    for t in nums[1:]:
        addLeaf(root,t)
    return root

root = randomTree(random.choices(range(100),k = 18))
draw(root)
```

<img src="%E5%9F%BA%E7%A1%80%20-%20Tree,%20BFS,%20DFS,%20stack,%20queue.assets/download.png" alt="download" style="zoom:200%;" />



### BFS

#### 递归
```python
## 递归方法，不常用
from collections import defaultdict
def level_order_traversal(node):
    res = defaultdict(list)
    def search(node,level):
        if not node:
            return None
        res[level].append(node.value)
        if node.left:
            search(node.left,level + 1)
        if node.right:
            search(node.right,level + 1)
    search(node, 0)
    return res

level_order_traversal(root)
```

结果是：

```
defaultdict(list,
            {0: [90],
             1: [72, 65],
             2: [85, 51, 77, 60],
             3: [93, 95, 12, 20, 38, 36],
             4: [94, 55, 41, 48, 94]})
```

#### 非递归

```Python
from queue import Queue
def bfs(root):
    his = []
    if not root:
        return None
    q = Queue()
    q.put(root)
    while not q.empty():
        l = q.qsize()
        for _ in range(l):
            n = q.get()
            if n.left:
                q.put(n.left)
            if n.right:
                q.put(n.right)
            his.append(n.value)
    return his
bfs(root)
```

结果是

```
[90, 72, 65, 85, 51, 77, 60, 93, 95, 12, 20, 38, 36, 94, 55, 41, 48, 94]
```


### DFS

#### 递归

```python
## recursive DFS
def recursive_DFS(root, method):
    res = []
    def visit(root):
        if not root:
            return
        if method == 'pre': res.append(root.value)
        if root.left: visit(root.left)
        if method == 'in': res.append(root.value)
        if root.right: visit(root.right)
        if method == 'post': res.append(root.value)
    visit(root)
    return res

print('Preorder',recursive_DFS(root,'pre'))
print('Inorder',recursive_DFS(root,'in'))
print('Postorder',recursive_DFS(root,'post'))
```

结果

```
Preorder [4, 2, 3, 11, 7, 8, 12, 13, 6, 19, 14, 15, 18, 16, 0, 17, 9, 10, 5, 1]
Inorder [11, 3, 7, 2, 13, 6, 19, 12, 8, 15, 14, 4, 16, 17, 0, 9, 18, 5, 10, 1]
Postorder [11, 7, 3, 19, 6, 13, 12, 15, 14, 8, 2, 17, 9, 0, 16, 5, 1, 10, 18, 4]
```

#### 非递归

```python
## Non-recursive DFS with stack
def Preorder_traversal(root):
    his = []
    if not root:
        return None
    stack = [root]
    while stack:
        n = stack.pop()
        his.append(n.value)
        if n.right:
            stack.append(n.right)
        if n.left:
            stack.append(n.left)
    return his

def Inorder_traversal(root):
    his = []
    if not root:
        return None
    stack = []
    n = root
    while stack or n:
        while n:
            stack.append(n)
            n = n.left
 
        
        n = stack.pop()
        his.append(n.value)
        n = n.right
    return his

def Postorder_traversal(root):
    his = []
    if not root:
        return None
    stack = []
    current = root
    while stack or current:
        while current:
            stack.append(current)
            current = current.left
            
        current = stack[-1]
        if not current.right or current.right == previous:
            his.append(current.value)
            previous = current
            current = None
            stack.pop()
        else:
            current = current.right
    
    return his

print('Preorder',Preorder_traversal(root))
print('Ineorder',Inorder_traversal(root))
print('Ineorder',Postorder_traversal(root))
```

结果

```
Preorder [4, 2, 3, 11, 7, 8, 12, 13, 6, 19, 14, 15, 18, 16, 0, 17, 9, 10, 5, 1]
Ineorder [11, 3, 7, 2, 13, 6, 19, 12, 8, 15, 14, 4, 16, 17, 0, 9, 18, 5, 10, 1]
Ineorder [11, 7, 3, 19, 6, 13, 12, 15, 14, 8, 2, 17, 9, 0, 16, 5, 1, 10, 18, 4]
```