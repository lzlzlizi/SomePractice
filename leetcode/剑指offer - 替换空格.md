# 替换空格

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

## 思路

数空格数，补在最后所需的空格数，从后向前替换

注意的是python的字符串不支持位置修改，所以化为list求解

尽管str.replace就可以，但是应该不是面试所需的内容

## 代码



```python
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here\
        if s == '':
            return ''
        s = [ss for ss in s]
        j = len(s)-1
        count = 0
        for ss in s:
            if ss == ' ':
                count += 1
        s = s + [' '] * count*2
        i = len(s) - 1
        while j >= 0:
            if s[j] == ' ':
                s[i] = '0'
                s[i-1] = '2'
                s[i-2] = '%'
                i -= 3
            else:
                s[i] = s[j]
                i -= 1
            j -= 1
        res = ''
        for ss in s:
            res = res + ss
        return res
```

