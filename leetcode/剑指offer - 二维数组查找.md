# 替换空格

## 题目描述

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数

## Solution

```python
class Solution:
    # array 二维列表
    def Find(self, t, array):
        if not array[0]:
            return False
        # write code here
        def search(a,b,c,d):
            if b - a <=1 and d-c <=1:
                return array[a][c] == t or array[a][d] == t or array[b][c] == t or array[b][d] == t
                     
            m1 = (a+b) // 2
            m2 = (c+d) // 2
            if array[m1][m2] == t:
                return True
            if array[m1][m2] > t:
                return search(a,m1,c,m2) or search(m1,b,c,m2) or search(a,m1,m2,d)
            else:
                return search(m1,b,m2,d) or search(m1,b,c,m2) or search(a,m1,m2,d)
        return search(0,len(array) - 1,0,len(array[0]) -1)
```

