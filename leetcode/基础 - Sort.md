# Their favorite algorithm - Sorting



## quick sort

The idea of behind quick sort is divide-and-conquer.

*  Firstly, we find a pivot in the array, 
* then move that little shit into its right place(the left part are smaller than it and its right part is elements that are larger than it)
* sort the left and right part recursively.
  * when an array only has one element, its a sorted array

**How to do quick sort in-place?**

* partition 
  * locate the pivot
  * swap the last and pivotal
  * for the elements before pivot, initialize point(for elements smaller than pivot)
    * if it is smaller than pivot, swap point and current element
  * then, all elements smaller than pivot is before the pointer, swap the pivot(last element) and pointer, the job is done
  * return pointer
* recursively partition the left part and the right part

```python
# inplace quick sort
def quick_sort(x):
    def partition(l,r):
        if l >= r:
            return l
        mid = (l+r) //2
        p = x[mid]
        x[mid],x[r] = x[r],x[mid]
        mid = l
        for i in range(l,r):
            if x[i] < p:
                x[i],x[mid] = x[mid],x[i]
                mid += 1
        x[r],x[mid] = x[mid],x[r]
        return mid
    def _sort(l,r):
        if l >= r:
            return 
        mid = partition(l, r)
        _sort(l, mid)
        _sort(mid+1, r)
        
    _sort(0,len(x) - 1)
    return x
```

### Complexity analysis

#### Average Case

Proposition: The average number of compares $C_N$ to quick sort an array of N distinct Keys is $\sim 2Nln(N)$, and the number if exchanges is $\sim 1/3Nln(N)$

* $C_{N}$ satisfies the recurrence $C_{0}=C_{1}=0$ and for $N \geq 2$

* $$
  \begin{aligned}
  C_{N}&=(N+1)+\left(\frac{C_{0}+C_{N-1}}{N}\right)+\left(\frac{C_{1}+C_{N-2}} {N_{S}}\right)+\ldots+\left(\frac{C_{N-1}+C_{0}}{N}\right)\\
  &\text{this is an expectation for different partitions} \\
  N C_{N}&=N(N+1)+2\left(C_{0}+C_{1}+\ldots .+C_{N-1}\right) \\
   N C_{N}-(N-1) C_{N-1}&=2 N+2 C_{N-1}\\
   \frac{C_{N}}{N+1}&=\frac{C_{N-1}}{N}+\frac{2}{N+1}
  \end{aligned}
  $$


$$
\begin{aligned}
\frac{C_{N}}{N+1} &=\frac{C_{N-1}}{N}+\frac{2}{N+1} \\
&=\frac{C_{N-2}}{N-1}+\frac{2}{N}+\frac{2}{N+1}+\mathrm{sur} \\
&=\frac{C_{N-3}}{N-2}+\frac{2}{N-1}+\frac{2}{N}+\frac{2}{N+1} \\
&=\frac{2}{3}+\frac{2}{4}+\frac{2}{5}+\ldots+\frac{2}{N+1}
\end{aligned}
$$
Approximate sum by an integral:
$$
C_{N}=2(N+1)\left(\frac{1}{3}+\frac{1}{4}+\frac{1}{5}+\ldots \cdot \frac{1}{N+1}\right)
$$
$$
\sim 2(N+1) \int_{3}^{N+1} \frac{1}{x} d x
$$
Finally, the desired result:
$$
C_{N} \sim 2(N+1) \ln N \approx 1.39 N \lg N
$$

#### Worse case

* The worse case is we the pivot is chosen to be the smallest or largest

* In that case, the complexity is $O(N)$



## Basics about sort

## Others

### Bubble

```python
def bubble_sort(x):
    for i in range(len(x)):
        for j in range(i,len(x)):
            if x[i] > x[j]:
                x[j],x[i] = x[i],x[j]
    return x
```

### insert

```python
def insertio_sort(x):
    def bifind_poi(x,target):
        l = 0
        r = len(x) - 1
        if x[l] >= target:
            return 0
        if x[r] <= target:
            return r + 1
        while l < r:
            mid = (l+r) // 2
            if x[mid] <= target and x[mid + 1]>= target:
                return mid +1
            elif x[mid] > target:
                r = mid
            else:
                l = mid + 1
               
        if len(x) < 1:
            return x
        new = [x[0]]
        for i in range(1,len(x)):
            po = bifind_poi(new,x[i])
            new.insert(po,x[i])
        return new
```

### merge

```python
def merge_sort_stupid(x):
    def merge2(x,y):
        res = []
        if not x:
            return y
        if not y:
            return x
        i = j = 0
        while i < len(x) and j < len(y):
            if x[i] < y[j]:
                res.append(x[i])
                i += 1
            else:
                res.append(y[j])
                j += 1
        else:
            if i < len(x):
                return res + x[i:]
            else:
                return  res + y[j:]
    def merge(xs):
    
        if len(xs) == 1:
            return xs[0]
        res = []
        while len(xs) > 1:
            x = xs.pop()
            y = xs.pop()
            res.append(merge2(x,y))
        return merge(res + xs)
    
    return merge([[t] for t in x])

def merge_sort(x):
    def merge2(x,y):
        res = []
        if not x:
            return y
        if not y:
            return x
        i = j = 0
        while i < len(x) and j < len(y):
            if x[i] < y[j]:
                res.append(x[i])
                i += 1
            else:
                res.append(y[j])
                j += 1
        else:
            if i < len(x):
                return res + x[i:]
            else:
                return  res + y[j:]

    if len(x) == 1:
        return x
    mid = len(x) // 2
    
    return merge2(merge_sort(x[:mid]),merge_sort(x[mid:]))
```

## heap sort

```python
import heapq
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

