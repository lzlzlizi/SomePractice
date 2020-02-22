# 替换空格

## 题目描述

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

## Solution

```python
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        dog = 0
        fantasy = len(array) - 1
        while dog <= fantasy:
            s = array[dog] + array[fantasy]
            if s == tsum:
                return array[dog], array[fantasy]
            elif s > tsum:
                fantasy -= 1
            else:
                dog += 1
        return []
```

