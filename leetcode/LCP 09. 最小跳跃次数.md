# LCP 09. 最小跳跃次数

为了给刷题的同学一些奖励，力扣团队引入了一个弹簧游戏机。游戏机由 N 个特殊弹簧排成一排，编号为 0 到 N-1。初始有一个小球在编号 0 的弹簧处。若小球在编号为 i 的弹簧处，通过按动弹簧，可以选择把小球向右弹射 jump[i] 的距离，或者向左弹射到任意左侧弹簧的位置。也就是说，在编号为 i 弹簧处按动弹簧，小球可以弹向 0 到 i-1 中任意弹簧或者 i+jump[i] 的弹簧（若 i+jump[i]>=N ，则表示小球弹出了机器）。小球位于编号 0 处的弹簧时不能再向左弹。

为了获得奖励，你需要将小球弹出机器。请求出最少需要按动多少次弹簧，可以将小球从编号 0 弹簧弹出整个机器，即向右越过编号 N-1 的弹簧。

示例 1：

输入：jump = [2, 5, 1, 1, 1, 1]

输出：3

解释：小 Z 最少需要按动 3 次弹簧，小球依次到达的顺序为 0 -> 2 -> 1 -> 6，最终小球弹出了机器。

限制：

1 <= jump.length <= 10^6
1 <= jump[i] <= 10000



## solution

1400s 左右 python BFS

循环里使用set可以避免重复

* jump可以后来用作visited
* 记录left, right为下一轮访问的范围（不包括jump构建的图），
  * right右边的之前都访问过了
  * left右边的都可以一步跳到left上



```python
class Solution:
    def minJump(self, jump: List[int]) -> int:
        G = [[] for _ in jump]
        q = []
        LEN = len(jump)
        for i,t in enumerate(jump):
            if i+t < LEN:
                G[i+t].append(i)
            else:
                q.append(i)
        
        res = 1
        left = q[0]
        right = LEN -1
        while q:
            tmp = set()
            for t in range(left + 1 ,right + 1):
                if jump[t] != -1:
                    tmp.add(t)
            right = left
            
            for c in q:    
                jump[c] = -1
                left = min(left,G[c][0]) if G[c] else left
                if left == 0:
                    return res + 1
                for t in G[c]:
                    if jump[t] != -1:
                        tmp.add(t)
                    if t > right:
                        break
                    
            q = tmp
            res += 1
```





