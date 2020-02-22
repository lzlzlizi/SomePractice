# Priority Queue



![img](%E5%9F%BA%E7%A1%80%20-%20%20Priority%20Queue.assets/1_oN767xTYckRTUjTIyS3oyw.png)

* queue (queue.Queue)
* stack(list)
* heap(heapq)

## Heap

* 数组表示的二叉树
* <img src="%E5%9F%BA%E7%A1%80%20-%20%20Priority%20Queue.assets/1_ds0JXOw3lLqNo6hw__NtZw.png" style="zoom: 67%;" />

* if the heap is a max heap, than each node has values smaller than values of its children
* **index**
  * say the current node has index $i$
  * parent = i // 2
  * left = 2*i
  * right = 2*i + 1 

### some algorithms

**heapift_i**

* 假设i的叶子树都满足条件了
* 调整i：
  * 判断 是否比子节点大，不是则和最大的子节点交换
  * 递归做下去
* $O(logn)$

**heapify**

* 从最后的非叶子节点出发，向前heapify
* $O(nlogn)$

**insert**:

* 插入到数组最后
* 向上交换
  * 向上
  * 向下 heapify_i

**heappop**

* 最后的pop(原list的pop)移动到最前面
* heapify_i(0)
* 返回top

```python
class MaxHeap(list):
    def __init__(self,x):
        super(MaxHeap,self).__init__(x.copy())
        self.heapify()
        
    def _lt(self,x,y):
        return x < y
    def heapify(self):
        for i in range(len(self)//2, -1, -1):
            self.heapify_i(i)
    
    def heapify_i(self, i):
        while 2*i + 1 < len(self):
            if self._lt(self[2*i], self[2*i+1]):
                if not self._lt(self[i], self[2*i]):
                    self[i], self[2*i] =self[2*i], self[i]
                    
                    self.heapify_i(2*i)
            else:
                if not self._lt(self[i], self[2*i + 1]):
                    self[i], self[2*i + 1] = self[2*i + 1], self[i]
                    self.heapify_i(2*i + 1)
            return 
    def top(self):
        return self[0]
    
    def heappop(self):
        if len(self) == 1:
            return self.pop()
        res = self[0]
        self[0] = self.pop()
        self.heapify_i(0)
        return res
    def insert(self, x):
        self.append(x)
        i = len(self) - 1
        parent = i // 2
        while i != parent:
            if not self._lt(self[parent], self[i]):
                self[parent], self[i] = self[i], self[parent]
            if 2 * parent + 1 < len(self) and self._lt(self[2 * parent + 1], self[parent]):
                self.heapify_i(parent)
            
            i = parent
            parent = parent // 2
            
            
        
        
```





### python 的 heapq

heapq默认实现的是minheap

对list操作

* heapq.heapify(a)是实现最小
* heapq.\_heapify\_max 最大
* 同理 heapq.heappop和 heapq.\_heappop_max 

```python
# heap sort
    
def heap_sort_max(a):
    heapq._heapify_max(a)
    res = []
    while a:
        res.append(heapq._heappop_max(a))
    return list(reversed(res))
def heap_sort_min(a):
    heapq.heapify(a)
    res = []
    while a:
        res.append(heapq.heapop(a))
    return res
```

