# Dynamic Programming

**适合的问题**

(a)最优子结构：问题的最优解由相关子问题的最优解组合而成，并且可以独立求解子问题！

(b)子问题重叠：递归过程反复的在求解相同的子问题

**求解步骤**

1. define subproblems 
2.  guess (part of solution) 
3. relate subproblem solutions compute
4. recurse + memoize
   *  OR build DP table bottom-up check subproblems acyclic/topological order 
5. solve original problem: = a subproblem 
   * OR by combining subproblem solutions =⇒ extra time

## 53. Maximum Subarray

Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

**Example:**

```
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

**Follow up:**

If you have figured out the O(*n*) solution, try coding another solution using the divide and conquer approach, which is more subtle.

### DP 解法

**递推式**

```
Bottom up DP
DP[i] = max(DP[i - 1] + nums[i], nums[i]) 
```

代码

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return None
        DP = [nums[0]]
        res = nums[0]
        for i in range(1,len(nums)):
            m = max(DP[i-1] + nums[i], nums[i])
            res = max(res,m)
            DP.append(m)
        return res
```

## 70. Climbing Stairs

You are climbing a stair case. It takes *n* steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Note:** Given *n* will be a positive integer.

**Example 1:**

```
Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```

**Example 2:**

```
Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

**DP 解法**

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        DP=[0,1,2]
        for i in range(3,n+1):
            DP.append(DP[i-2] + DP[i-1])
        return DP[n]
```

## 121. Best Time to Buy and Sell Stock

Say you have an array for which the *i*th element is the price of a given stock on day *i*.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

**Example 1:**

```
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
```

**Example 2:**

```
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
```

### DP

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        DP = [prices[0]]# min Price
        res = 0
        for i in range(1,len(prices)):
            DP.append(min(DP[i-1], prices[i]))
            res = max(prices[i] - DP[i],res)
        return res
```



### Other  Methods

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        LowPrice = 999999999
        MaxProfit = 0
        for p in prices:
            LowPrice = min(p,LowPrice)
            MaxProfit = max(MaxProfit, p - LowPrice)
        return MaxProfit
                
```

## 198. House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight **without alerting the police**.

**Example 1:**

```
Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
```

**Example 2:**

```
Input: [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
             Total amount you can rob = 2 + 9 + 1 = 12.
```

### DP

有状态，抢了或者没抢

* 抢了，则是之前没抢的值加当前值
* 没抢，则为之前抢或没抢的最大值

```python
# DP[i] = [max(DP[i-1]),  DP[i-1][0]+nums[i]]


class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        DP = [[0,nums[0]]]
        for i in range(1,len(nums)):
            DP.append([max(DP[i-1]),DP[i-1][0]+nums[i]])
        return max(DP[-1])
        
```

## 62. Unique Paths

A robot is located at the top-left corner of a *m* x *n* grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

![img](%E5%9F%BA%E7%A1%80%20-%20DP.assets/robot_maze.png)
Above is a 7 x 3 grid. How many possible unique paths are there?

**Note:** *m* and *n* will be at most 100.

**Example 1:**

```
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
```

**Example 2:**

```
Input: m = 7, n = 3
Output: 28
```

### DP

```python
# 慢
import numpy as np
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        DP = np.zeros([m,n]).astype(int)
        DP[0,:] = 1
        DP[:,0] = 1
        for i in range(1,m):
            for j in range(1,n):
                DP[i,j] = DP[i-1,j] + DP[i,j-1]
        return DP[m-1,n-1]


```

###  返回组合数

```python
# 调包慢
from scipy.special import comb
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return int(comb(max(m,n),min(m,n)))
```

## 64. Minimum Path Sum

Given a *m* x *n* grid filled with non-negative numbers, find a path from top left to bottom right which *minimizes* the sum of all numbers along its path.

**Note:** You can only move either down or right at any point in time.

**Example:**

```
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
```

### DP

* 上一步的最小值（向下还是向右）加上当前值

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        DP = grid
        m = len(grid)
        n = len(grid[0])
        for j in range(1,n):
            DP[0][j] += DP[0][j-1]
        for i in range(1,m):
            DP[i][0] += DP[i-1][0]
        for i in range(1,m):
            for j in range(1,n):
                DP[i][j] += min(DP[i-1][j],DP[i][j-1])
        return DP[-1][-1]
```



## 139. Word Break

Given a **non-empty** string *s* and a dictionary *wordDict* containing a list of **non-empty** words, determine if *s* can be segmented into a space-separated sequence of one or more dictionary words.

**Note:**

- The same word in the dictionary may be reused multiple times in the segmentation.
- You may assume the dictionary does not contain duplicate words.

**Example 1:**

```
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```

**Example 2:**

```
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
```

**Example 3:**

```
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false
```

### 超时的递归

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        wordDict = set(wordDict)
        lens = sorted(list(set([len(s) for s in wordDict])))
        def check(s):
            if not s: return True
            for l in lens:
                if len(s) >= l:
                    if s[:l] in wordDict:
                        if len(s) == l or check(s[l:]):
                            return True 
                else:
                    return False
            print(s)
            return False
        return check(s)
                    
                    
        
```

### DP

- DP is an array that contains booleans
- DP[i] is True if there is a word in the dictionary that *ends* at ith index of s AND DP is also True at the beginning of the word

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        DP = [False]*(len(s))
        for i in range(0,len(s)):
            for w in wordDict:
                if w == s[i - len(w) + 1:i+1] and (DP[i-len(w)] or i - len(w) == -1):
                    DP[i] = True
        return DP[-1]
```

## 338. Counting Bits

Given a non negative integer number **num**. For every numbers **i** in the range **0 ≤ i ≤ num** calculate the number of 1's in their binary representation and return them as an array.

**Example 1:**

```
Input: 2
Output: [0,1,1]
```

**Example 2:**

```
Input: 5
Output: [0,1,1,2,1,2]
```

**Follow up:**

- It is very easy to come up with a solution with run time **O(n\*sizeof(integer))**. But can you do it in linear time **O(n)** /possibly in a single pass?
- Space complexity should be **O(n)**.
- Can you do it like a boss? Do it without using any builtin function like **__builtin_popcount** in c++ or in any other language.

### DP

每增加一个bit时就是重复之前的路径

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        DP = [0] * (num+1)
        T = 1
        for i in range(1,num+1):
            if i == 2*T:
                DP[i] = 1
                T *= 2
            else:
                DP[i] = DP[i-T] + 1
        return DP
                    
```

## 647. Palindromic Substrings

Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

**Example 1:**

```
Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
```

 

**Example 2:**

```
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
```

 

**Note:**

1. The input string length won't exceed 1000.

### $O(N^2)$ 的方法 - 中心扩散

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        if len(s) < 2:
            return len(s)
        res = len(s)
        for i in range(0,len(s)):
            for j in range(1,len(s)):
                if i+j < len(s):
                    if i>=j and s[i-j] == s[i+j]:
                        res += 1
                    else:
                        break
            for j in range(1,len(s)):
                if i+j < len(s):
                    if i-j + 1 >= 0 and  s[i+1-j] == s[i+j]:
                        res += 1
                    else:
                        break
        return res
    
# 好一点的改进
class Solution:
    def countSubstrings(self, s: str) -> int:
         #Expand Around Center
        def helper(s, l, r):
            count = 0
            while 0 <= l and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
                count += 1
            return count
        
        result = 0
        for i in range(len(s)):
            result += helper(s, i, i)
            result += helper(s, i, i+1)
        return result
```

### DP

* Store the bool variables in DP which represent whether the substring beginning at i and ending at j is palindromic
* check i, j pair 
  * if s[i] is the same as s[j], $DP[i][j]$ = 1 if the center is 1 otherwise 0
  * else 0

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        res = n
        DP = [[0  if i != j else 1 for i in range(n)] for j in range(n)]
        for i in range(1,n):
            if s[i] == s[i-1]:
                res += 1
                DP[i-1][i] = 1
        for gap in range(2,n):
            for i in range(0,n-gap):
                j = i + gap
                if s[i] == s[j] and DP[i+1][j-1]:
                    DP[i][j] = 1
                    res += 1
                else:
                    DP[i][j] = 0
        return res
```



## Target Sum

You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols `+` and `-`. For each integer, you should choose one from `+` and `-` as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

**Example 1:**

```
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
```

**Note:**

1. The length of the given array is positive and will not exceed 20.
2. The sum of elements in the given array will not exceed 1000.
3. Your output answer is guaranteed to be fitted in a 32-bit integer.



### 超时的递归

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], s: int) -> int:
        if len(nums) == 1:
            if s == nums[0] != -s:
                return 1 
            elif s == -nums[0] != -s:
                return 1
            elif s == -nums[0] == 0:
                return 2
            else:
                return 0
        return self.findTargetSumWays(nums[1:], s - nums[0]) + self.findTargetSumWays(nums[1:], s + nums[0])
## 直观的暴力
class Solution:
    def findTargetSumWays(self, nums: List[int], s: int) -> int:
        n = len(nums)
        bruteforce = [0]
        for j in nums:
            bruteforce =[t + j for t in bruteforce] + [t - j for t in bruteforce]
        res = 0
        for t in bruteforce:
            if t == s:
                res += 1
        return res
```

## DP

* 关键点是The sum of elements in the given array will not exceed **1000**.
* 然后子问题是$$DP[i][\sum\limits_{k=1}^{i}a_k]$$
* 下面的代码和上面的暴力有区别的原因是因为子问题的值很大时上面的暴力会有几十几百个重复值，因此慢
* 这正是DP的motivation

```python
from collections import defaultdict
class Solution:
    def findTargetSumWays(self, nums: List[int], s: int) -> int:
        if not nums:
            return 0
        n = len(nums)
        DP = defaultdict(int)
        DP[nums[0]] = 1
        DP[- nums[0]] = 1 +  DP[- nums[0]]
        for i in range(0,n-1):
            new = nums[i+1]
            Temp = defaultdict(int)
            for j in DP:
                Temp[j+new] = DP[j] + Temp[j+new]
                Temp[j-new] = Temp[j-new] + DP[j]
            DP = Temp
        return DP[s]
```

## 309. Best Time to Buy and Sell Stock with Cooldown

Say you have an array for which the *i*th element is the price of a given stock on day *i*.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

- You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
- After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)

**Example:**

```
Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]
```



### DP

* 类似强化学习的MDP, 但价格不具备马尔科夫性质
* 定位state然后确定每个 state的转移方程
* 转移函数是每个到达曲线的转移函数的最大值
  * $\pi(s) = arg\max_a V(s,a)$

![Best Time to Buy and Sell Stock with Cooldown](%E5%9F%BA%E7%A1%80%20-%20DP.assets/Best%20Time%20to%20Buy%20and%20Sell%20Stock%20with%20Cooldown.jpg)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        value = [-prices[0],0,0]#balance in each state
        for p in prices[1:]:
            value = [max(value[0], value[2] - p),value[0]+p, max(value[2],value[1])]
        return max(value)
```



## 279. Perfect Squares

Given a positive integer *n*, find the least number of perfect square numbers (for example, `1, 4, 9, 16, ...`) which sum to *n*.

**Example 1:**

```
Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.
```

**Example 2:**

```
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
```

### DP

For any integer 0 < k ≤ n, D[k] = min(1 + D[k-l²]), where l is an integer and 0 < l < SQRT(k)

```python
class Solution:
    def numSquares(self, n: int) -> int:
        if n == 1:
            return 1
        candidate = [1]
        DP = [1]
        t = 1
        for i in range(2,n+1):
            if (t+1)**2 == i:
                DP.append(1)
                t+=1
            else:
                m = DP[i - 1-1] + 1
                for j in range(2,t+1):
                    m = min(m,DP[i - j*j-1]+1)
                DP.append(m)
        return DP[-1]
```



### ~~数论的定理做 (没意思)~~

By [**Lagrange’s Four-Square Theorem**](https://en.wikipedia.org/wiki/Lagrange's_four-square_theorem):

> every natural number can be represented as the sum of four integer squares

and

> a positive integer can be expressed as the sum of three squares if and only if it is not of the form `4^k(8m+7)` for integers `k` and `m`

The result can be found quicker because it will only be less and equal to 4.

```python
class Solution:
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        # approach: by Lagrange’s four-square theorem, every natural 
        #           number can be represented by sum of of four 
        #           integer squares
        #           And it can be represented by sum of three 
        #           integer squares if and only if it’s not in form 
        #           4^k(8m + 7)

        k = n
        while k % 4 == 0:
            k = k // 4
        if k % 8 == 7:
            return 4

        i = 0
        while i ** 2 <= n:
            j = 0
            while j ** 2 <= n-(i**2):
                if i ** 2 + j ** 2 == n:
                    return (1 if i > 0 else 0) + (1 if j > 0 else 0)
                j += 1
            i += 1

        return 3
#another Version 
class Solution:
    def isSquare(self, n: int) -> bool:
        sq = int(math.sqrt(n))
        return sq*sq == n

    def numSquares(self, n: int) -> int:
        # four-square and three-square theorems
        while (n & 3) == 0:
            n >>= 2      # reducing the 4^k factor from number
        if (n & 7) == 7: # mod 8
            return 4

        if self.isSquare(n):
            return 1
        # check if the number can be decomposed into sum of two squares
        for i in range(1, int(n**(0.5)) + 1):
            if self.isSquare(n - i*i):
                return 2
        # bottom case from the three-square theorem
        return 3
```

