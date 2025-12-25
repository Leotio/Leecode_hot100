## 哈希

### [1. 两数之和](https://leetcode.cn/problems/two-sum/)

``````python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic1 = {}
        tmp = 0
        for i,num in enumerate(nums):
            dic1[num] = i
        for i in range(len(nums)):
            tmp = target - nums[i]
            if tmp in dic1 and i != dic1[tmp]:
                return [i,dic1[tmp]]
``````

这种写法有致命问题啊，给的nums中有相同数字的时候，dict1中出现相同的key，这里的话会将新的value进行覆盖，然后这个题目又正好是找两个数的和，但凡改一些条件什么的，就会出问题，而且两次遍历，效率也不高，可以一次遍历实现的。

这个地方正好是求解两个数的和，所以只需一次遍历，比如是2和4加起来满足条件，那么虽然在遍历到2的时候，4还没有出现，但是后续遍历到4的时候，2已经被加入字典了，所以整体上就是只需要一次遍历的。

```````python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []
```````

也只能处理两个数，但至少不存在冲突的问题。比如遇到第二个3，就返回了，没有加到hashtable里面去。

### [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)

        for word in strs:
            key = "".join(sorted(word))
            mp[key].append(word)
        return list(mp.values())
```

**语法补充：**

- 掌握collections.defaultdict()方法
- join()方法
- 字典的values()方法，返回值组成的一个视图对象，还需要将其转化为列表

### [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)

**思路**：对于每个数字，先判断他前面的数字在不在集合，如果在的话就跳过，因为我们只去考虑作为最长连续序列开头的数字；如果不在，就说明他是开头，那么就开始while循环不断更新长度就行了。同时为了能让查找操作简单，先把他化为一个set集合，就能用in来判断了。

这个地方``for num in set_nums:``不能用``for i in range(len(nums))``来代替！本身时间复杂度两种方法都是O(n)的，但是测试用例里面有大量的重复元素，最后一定会导致超时。

<img src="C:\Users\Leoti\AppData\Roaming\Typora\typora-user-images\image-20251128153507415.png" alt="image-20251128153507415" style="zoom: 33%;" />

````python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        set_num = set(nums)
        length = 0
        # 超时超时
        for i in range(len(nums)):
            if nums[i]-1 in set_num:
                continue
            tmp = nums[i]
            while tmp in set_num:
                tmp+=1
            length = max(length,tmp-nums[i])
        return length
````



````python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest = 0

        set_nums = set(nums)
        for num in set_nums:
            if num-1 not in set_nums:
                tmp_length = 0
                while num in set_nums:
                    tmp_length += 1
                    num += 1
                longest = max(longest,tmp_length)
        return longest
````

## 双指针

### [283. 移动零](https://leetcode.cn/problems/move-zeroes/)

用双指针来实现，left指向非0且按照原顺序的序列的后一位，right指向待遍历序列的首位。right只要不是0就和left进行交换。虽然有时候的交换确实是原地交换，因为没有出现0元素之前，left和right一直是同步更新的。

````python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = right = 0
        while right < len(nums):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1
````

五刷的代码，我觉得好优雅：

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        sort_last = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[sort_last], nums[i] = nums[i], nums[sort_last]
                sort_last += 1
```

### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

这道题的算法思路倒是有印象，但是自己去写的时候真的是写了依托啊，有时候代码的简洁性，可读性真的是要多练。

突然发现思路也有些问题，我的思路是找到比left和right中小的那个大的才移动，但实际上没必要，就一直移动那个小的就行了，这个白白增加了思维量而且还没有节省空间。

**思路**:每次移动left和right中较矮的那个指针，不停更新就行

````python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height)-1
        max_water = min(height[left], height[right])*(right-left)
        while left < right:
            max_water = max(max_water,min(height[left],height[right])*(right-left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_water
````

### [15. 三数之和](https://leetcode.cn/problems/3sum/)

**思路**:先对数组sort，排序后每次选取一个数字后，target变为这个数的相反数了，~~在这个数后面的数上双指针，一左一右，小了就left右移，大了就right指针左移。这里通过去判断是否和上个数相等来排除重复的结果，只要相等就跳。~~（好像也不是不行，但真的麻烦很多，复杂度也并没有好很多）

确定第一个数之后，第二个数从这个数后面开始遍历，每次选定第二个数，第三个数从数组最后一个数开始遍历，和大于target就不断左移就是了。

（既然是去重，自然而然想到set，但实际上，这不仅仅是因为数字重复而造成的结果重复，还有是结果中的数字顺序换了一下导致的重复，这通过set是没办法去除的）

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        for i in range(len(nums)):
            # 通过first>0就解决了i=0的时候的问题，保证了这个数字会考虑一次，但是不会再有第二次拿他做开头
            if i > 0 and nums[i]==nums[i-1]:
                continue
            target = -nums[i]
            # ！！！注意这个地方，right一定要放在for循环外面！数组已经排过序了，第一个数不变，第二个数增大的时候，right完全没有必要从最后开始移动！！！！
            right = len(nums)-1
            for left in range(i+1, right):
                # 同样通过left>i+1就解决了，第一个数的问题
                if left > i+1 and nums[left-1] == nums[left]:
                    continue
                while left < right and nums[left]+nums[right] > target:
                    right -= 1
                if left < right and nums[left]+nums[right] == target:
                    ans.append([nums[i],nums[right],nums[left]])
        return ans
```

### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

**思路：**$$\text{Water}_i = \min(\text{max\_left}, \text{max\_right}) - \text{height}[i]$$

其中 $\text{max\_left}$ 是 $\text{height}[i]$ 左侧所有墙壁的最大高度，$\text{max\_right}$ 是 $\text{height}[i]$ 右侧所有墙壁的最大高度。

要是左边的最大高度比右边的最大高度矮，那我就算左边这个东西的接的雨水量；要是右边的最大高度矮，那我就算右边的这个的雨水量，哪边矮我算哪边的，一轮下来复杂度低啊，尤其空间复杂度也低。

要不然每次都去遍历一下啊，左边谁最高，右边谁又最高，这样子就巧妙的解决这个问题，反正是要找高个里头的矮个，那我哪个矮就算哪边嘛，这样子就好了。

````python
class Solution:
    def trap(self, height: List[int]) -> int:
        # max_l, max_r维护的是到letf-1和right+1的最大高度，也就是以及处理好的
        # 所以不需要给height处理左右两个柱子的情况了
        max_l, max_r = 0, 0
        left, right = 0, len(height)-1
        ans = 0
        while left < right:
            max_l = max(max_l, height[left])
            max_r = max(max_r, height[right])
            # 柱子相对而言矮的一边！！说明另一边的max肯定比这边的max大，因为最少也会达到高柱子的高度！！
            if height[left] < height[right]:
                ans += max_l - height[left]
                left += 1
            else:
                ans += max_r - height[right]
                right -= 1
        return ans
````

## 滑动窗口

### [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

**思路：**每次将左指针向右移动一格，表示开始枚举下一个字符作为起始位置，然后不断地向右移动右指针，保证这两个指针对应的子串中没有重复的字符。在移动结束后，这个子串就对应着以左指针开始的，不包含重复字符的最长子串。

自己的理解有问题，我在这里滑动，是没有进行固定的，右边滑不动了就直接去滑左边，虽然原理上差不多，但是在实现的逻辑上就会麻烦很多，换个方法理解，应该是去固定左端，不停滑动右边，获得当前左指针为起点的最大不重复子串。

**语法补充：**集合set中删除元素调用remove(char)，添加为add(char)

````python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        right, max_length = 0, 0
        set_char = set()
        for left in range(len(s)):
            if left != 0:
                # 不要写成remove(left)了
                # 注意是left-1哈，比如left等于1的时候，把0的去掉，从1开始，这样子才能考虑到所有情况
                set_char.remove(s[left-1])
            while right < len(s) and s[right] not in set_char:
                set_char.add(s[right])
                right += 1
            # 这个时候right已经跑到这个字串的后一个字符了，所以不需要加1
            max_length = max(max_length, right-left)
        return max_length
````

### [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

**思路**：维护两个由26个字母映射到0-25的列表，每次移动一个位置并判断两个列表是否相等，相等就是异位词了。

**语法补充：**list的判等操作！

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 我一开始写的list_s = list_p = [0] * 26然后就出问题了，这个地方这样写的话是共有一片内存的
        list_s = [0] * 26
        list_p = [0] * 26
        ans = []
        ls, lp = len(s), len(p)
        if ls < lp:
            return []
        for i in range(lp):
            # 初始化list_p，list_s
            list_p[ord(p[i]) - 97] += 1
            list_s[ord(s[i]) - 97] += 1
        # 初始化后相等，说明是从0开始这里有个异位词
        if list_p == list_s:
            ans.append(0)
        for i in range(ls-lp):
            # 每次去掉首个，再加上最后一个                                                                 
            list_s[ord(s[i])-97] -= 1
            list_s[ord(s[i+lp])-97] += 1
            if list_p == list_s:
                # 要注意这里是i+1！因为0开头的是单独写出去了的
                ans.append(i+1)
        return ans
```

四刷的时候写的代码，复杂度一致，感觉相对简洁一些。不对，这个方法时间复杂度一样，但是空间复杂度高了。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        dict_p = Counter(p)
        m, n = len(s), len(p)
        ans = []
        for i in range(m-n+1):
            tmp_s = Counter(s[i:i+n])
            if tmp_s == dict_p:
                ans.append(i)
        return ans
```

五刷的代码，把四刷的空间复杂度降下来了，但是还不如最开始的呢，因为Counter的语法比List还是复杂一些

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        pcnt = Counter(p)
        np = len(p)
        scnt = Counter(s[:np])
        ans = []
        if scnt == pcnt:
            ans.append(0)
        for i in range(np,len(s)):
            scnt[s[i]] += 1
            scnt[s[i-np]] -= 1
            if scnt == pcnt:
                ans.append(i-np+1)
        return ans
```

## 字串

### [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

**思路：**想找到和为 $K$ 的子数组 $\text{nums}[j \dots i]$，即：$$\text{Sum}(\text{nums}[j \dots i]) = K$$

利用前缀和公式替换，得到： $$P[i] - P[j-1] = K$$

将等式变形，得到：$$P[j-1] = P[i] - K$$

​			$$P[i]$$代表数组$$[0,1,2....i]$$的和

**这个等式是关键：** 当我们遍历到当前索引 $i$ 并计算出前缀和 $P[i]$ 时，我们只需要知道在 $i$ 之前，是否存在一个前缀和 $P[j-1]$ 恰好等于 $P[i] - K$。

- 如果存在，那么从 $j$ 到 $i$ 的子数组的和就等于 $K$。
- 如果有多个这样的 $P[j-1]$ 存在，那么就有多个子数组的和等于 $K$。

注意：哈希表初始化为 `{0: 1}` ，它代表在数组开始之前（即索引 $-1$ 处），前缀和为 $0$ 出现了一次。这用于处理子数组从数组起始位置（索引 $0$）开始的情况。

- 如果 $P[i] - K = 0$，即 $P[i] = K$，这意味着从索引 $0$ 到 $i$ 的子数组的和为 $K$。
- 因为 $\text{prefix\_map}$ 中有 `{0: 1}`，所以当 $P[i] - K = 0$ 时，我们会找到 $1$ 个这样的子数组。

**语法补充：**``dictionary.get(key, default_value)``字典中去获取key的值，如果这个值不存在，把default值给他

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        hash_presum = defaultdict(int)
        # 将前缀和的hash表初始化，前缀和为0的初始化为1
        hash_presum[0] = 1
        presum = 0
        ans = 0
        for i in range(len(nums)):
            presum += nums[i]
            # target也就是思路中那个我们要寻找的前缀和
            # 不要怕最后所有的数字加起来等于k的情况被忽略掉了
            # 这就是找前缀和为0的数目，已经在初始化的那个{0:1}中算上了！
            target = presum - k
            # 先判断是否存在，再把这次的前缀和加入字典
            if target in hash_presum:
                ans += hash_presum[target]
            # 详见上面的语法补充！！！！
            hash_presum[presum] = hash_presum.get(presum, 0) + 1
        return ans
```

三刷的时候自己写的，发现好像比之前的好很多！

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        sum,ans = 0,0
        # 直接用defaultdict来建立字典，访问到不存在的元素会给一个0
        dict_ = defaultdict(int)
        # 初始一下dict_[0]=1表示从头开始的子数组满足条件
        dict_[0] = 1
        for num in nums:
            sum += num
            # 下面两条的顺序不能反过来（主要影响k=0的情况，k=0的话先执行下面的语句，会导致结果多1）
            ans += dict_[sum-k]
            dict_[sum] += 1
        return ans
```

### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

**思路**：利用python中小顶堆数据结构（注意这里要获得最大的元素，所以对元素要进行取负值的操作）先对前k个元素执行建堆操作，然后就获得了第一个区间的max值了。接下来，不断移动区间，获取堆顶元素，如果索引在区间内就append，如果不在直接pop掉就OK，直到堆顶元素在区间内再append就可以了，整个数组遍历完就结束。

**语法补充**：Python中可以对元组建堆。

​		掌握python堆的一些方法:

​		建堆方法：``heapq.heapify(q)``这里是原地建堆，对q建堆之后，q就是堆名了

​		插入：``heapq.heappush(q, (-nums[i], i))``删除堆顶元素：``heapq.heappop(q)``

````python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 小顶堆要获取最大的元素，所以要注意进行负值操作
        q = [(-nums[i], i) for i in range(k)]
        # 注意建堆，这里的q是元组列表，这样子后面判断索引是不是在当前窗口就方便很多
        heapq.heapify(q)
        ans = [-q[0][0]]
        for i in range(k,len(nums)):
            # 掌握这个语法
            heapq.heappush(q, (-nums[i], i))
            # 当前区间的第一个位置的索引是i-k+1
            while q[0][1] < i-k+1:
                heapq.heappop(q)
            ans.append(-q[0][0])
        return ans
````

### [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

**思路**：建立目标串t的counter，对于s不断遍历，先滑动右边窗口的界限，直到counter>=t的counter，这个时候就去移动左边的界限，直至不满足了，这个时候就是一个暂时的最小字串，一直移动，直到right到了s的边界。

**语法补充**：``collections``的``Counter()``，返回一个字典，如果传入一个字符串的类型，那么字典的key为各个字符，value为各个字符出现的次数，同时还支持进行比较，$$counter1>=counter2$$则说明2中的key在1中都有，而且对应的value值也是>=的关系。

````python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        cnt_t = Counter(t)
        cnt_s = Counter()
        left = 0
        ans_l = -1
        ans_r = len(s)-1
        for right,char in enumerate(s):
            cnt_s[char] += 1
            while cnt_s >= cnt_t:
                if right-left+1 < ans_r-ans_l+1:
                    ans_l = left
                    ans_r = right
                cnt_s[s[left]] -= 1
                left += 1
        # 这个地方一定要注意，不能是if not ans_l因为这个地方ans_l可能是0啊啊啊啊啊！！！  
        return "" if ans_l<0 else s[ans_l:ans_r+1]
````

## 普通数组

### [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

**思路**：利用动态规划的思路来求解。分为目前的最大前缀和加上此处的num和放弃前面的前缀和，从我这个数开始。为了节省空间，只要存下前面一个最大前缀和就行了。也就是代码中的f，ans不断更新全局的最大前缀和，最后返回即可。

````python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        f = 0 
        ans = float('-inf')
        for num in nums:
            # f是到目前这个num为止的最大前缀和
            # 等价于f = max(f+num,num)感觉这样更好理解
            f = max(f,0)+num
            # ans是全局的最大前缀和
            ans = max(ans,f)
        return ans
````

### [56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

**思路**：首先把这些区间进行排序，sort一下按照区间起点进行排序，然后每次判断后一个区间的起点是否小于等于前一个区间的终点，如果小于等于的话，就把前个区间的终点修改为两个区间的终点的最大值，这里为了节约空间，引入left指针，表示当前修改到哪了，如果没有重合部分的区间，直接将left加一，把该区间移到left所指的位置即可。最后返回原列表到left的子列表。

````python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # sort是核心！首先要想到这个sort才能推进
        intervals.sort()
        n = len(intervals)
        # 指示已经确定好的列表末尾
        left = 0
        for i in range(1, n):
            # 落在区间中间，那就进行合并
            if intervals[i][0] <= intervals[left][1]:
                # 区间终点更新为二者max
                intervals[left][1] = max(intervals[i][1],intervals[left][1])
            else:
                # 无重叠，直接放到合并好的区间后一个位置上
                left += 1
                intervals[left] = intervals[i]
        # 返回到left+1的子列表
        return intervals[:left+1]
````

### [189. 轮转数组](https://leetcode.cn/problems/rotate-array/)

**思路**：举例比如数组[1,2,3,4,5,6]要全部向右轮转4个位置，就是变为[3,4,5,6,1,2],可以通过这样的方式来实现，先全部翻转[6,5,4,3,2,1]这样子就实现了3456在12前面的目标，接下来发现将前四个和后两个分别翻转得到[3,4,5,6] [1,2]这样子就得到答案了！！！（实现的时候要注意一下，不要用切片方式来进行逆转，这样子会导致不符合题目中的原地操作，因为python的切片操作会开辟新的内存）

````python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        # 一定不能切片操作
        def reverse(start,end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        # 一定要注意这个取余操作！！！
        k = k%n
      
        reverse(0, n-1)
        reverse(0, k-1)
        reverse(k, n-1)
````

### [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)

**思路：**分别计算前缀积和后缀积，当前位置的ans就是前缀积与后缀积的积。（如果要进行优化的话，就先计算前缀积，然后在前缀积的数组上不断更新后缀积和ans）

````python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        preproduct = [1] * n
        postproduct = [1] * n
        for i in range(1,n):
            # preproduct[i]代表nums[0]，[1]，[2]..[i-1]的乘积
            preproduct[i] = preproduct[i-1] * nums[i-1]
            # postproduct[i]代表nums[n]，[n-1]，[2]..[i+1]的乘积
            postproduct[n-i-1] = postproduct[n-i] * nums[n-i]
        for i in range(n):
            ans.append(preproduct[i] * postproduct[i])
        return ans 
````

三刷自己写的空间优化为O(1)的版本

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 第一个数的前缀积初始化为1
        pre_mul = 1
        # 最后一个数的后缀积初始化为1
        post = 1
        # 后缀积的存储列表，同时作为最后的结果数组
        post_mul = [1] * len(nums)
        # post_mul[i]代表nums[i]的后缀积
        for i in range(len(nums)-1,0,-1):
            post *= nums[i]
            post_mul[i-1] = post
        # post_mul[i]此时存储的成为了除nums[i]外所有数的乘积
        for i in range(len(nums)):
            # 两句话顺序不能倒置！
            post_mul[i]= post_mul[i]*pre_mul
            pre_mul *= nums[i]
        return post_mul
```

### [41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)	

**思路：**首先明确一点：对于一个长度为 $N$ 的数组，第一个缺失的正整数一定在 $[1, N+1]$ 的范围内。

对于数组中的元素进行遍历，放置的规则是，整数1放到索引为0的地方，2放到索引为1的地方，以此类推，每次判断遍历到的这个数字，首先需要确实在1到n（数组的长度）之间，否则不处理，在这个区间就去判断是否在他该在的位置上，或者说那个位置上的值是否和他相等（因为可能有重复的数字），不等就交换。一直处理完。然后再次遍历，找到那个nums[i] != i+1的数，那缺的就是i+1，要是都匹配，那就是n+1不在。

````python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            # 一定是while而不是if，一定要等到当前索引放到了一个正确的数，或者是真的就不存在正确的数，才能到下一个位置上去
            # 因为这个地方交换过来的数可能本来是有另外一个正确的位置的，交换过来就不管了，后续就直接在这个位置就发现索引对不上了
            while 1 <= nums[i] <= n and nums[nums[i]-1] != nums[i]:
                # 把这个地方封装成函数就好了
                # def __swap(self, nums, index1, index2):
       			#	 nums[index1], nums[index2] = nums[index2], nums[index1]
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
                # nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]这样子写的话就会出问题！！！
                # python这种赋值是先把右边算出来，比如这里算出来的是（3，1），
                # 然后先对第一个数进行赋值，那就是nums[i]=3，这下好了nums[nums[i]-1]里面用到的nums[i]直接就被改掉了！
                # 封装为函数交换的索引就在传入的时候已经确定好了，不容易出错啊！！
        for i in range(n):
            if nums[i] != i+1:
                return i+1
        return n+1
````

## 矩阵

### [73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)

**思路：**用辅助的两个数组，首次遍历去判断某行或者某列是不是有0，然后再次遍历，把该行该列变为0即可。

如果考虑空间优化，那么对于这个两个辅助数组转为用原矩阵的第一行第一列来存储就行，同时为了保存首行首列是不是含0，借用两个辅助变量即可。（真的省了空间，但代码难看多了。。。。。）

**语法补充**：``any()``是 Python 的内置函数，接收一个可迭代对象, 只要可迭代对象中任何一个元素求值为 `True`，`any()` 就会立即返回 `True`，并**停止迭代**（短路原则）。

`````python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        # 两个辅助变量存储第一行第一列是否有0
        flag_row = any(matrix[0][j] == 0 for j in range(n))
        flag_col = any(matrix[i][0] == 0 for i in range(m))
        
        # 更新matrix的首行首列用来存储该行该列是否有0
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        # 对应行列变0
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][0] ==0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        # 首行是否有0，有则全变0            
        if flag_row:
            for i in range(n):
                matrix[0][i] = 0
        # 首列是否有0，有则全变0   
        if flag_col:
            for i in range(m):
                matrix[i][0] = 0        
`````

### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

**思路：**可以命名为缩圈法，先遍历最外面一圈，然后是里面圈，像剥洋葱一样，设置四个边界，left代表当前圈的左边一列的列索引，right代表当前圈的最右边一列的列索引，top代表当前圈的最上面一行的行索引，bottom代表当前圈的最下面一行的行索引。每次遍历，先遍历这个圈的上边界，然后右边界，然后下边界，然后左边界。遍历完一个圈的时候，left和top都加1，right和bottom都减一。

需要小心一点的是while的条件，对于上边和右边的条件``while left <= right and top <= bottom:``要带上等号，因为即使是相等也代表存在，而对于下边和左边则是要严格小于，因为只是相等的话，说明再次遍历会导致重复遍历。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows, cols = len(matrix), len(matrix[0])
        ans =[]
        # 四个边界
        left, right, top, bottom = 0, cols-1, 0, rows-1
        while left <= right and top <= bottom:
            # 当前圈第一行的所有元素
            for col in range(left,right+1):
                ans.append(matrix[top][col])
            # 当前圈最右边的一列除去第一个元素
            for row in range(top+1,bottom+1):
                ans.append(matrix[row][right])
            # 注意这个地方要重新判断一下啊！！！
            if left < right and top < bottom:
                # 当前圈最下面一行，除去最右边的元素
                for col in range(right-1,left,-1):
                    ans.append(matrix[bottom][col])
                # 当前圈的最左边一行，除去最上面的一个元素
                for row in range(bottom,top,-1):
                    ans.append(matrix[row][left])
            # 更新边界
            left, right, top, bottom = left+1, right-1, top+1, bottom-1 
        return ans
```

### [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

**思路：**

方法一：每次旋转四个位置，以四个顶点为例，每个顶点都旋转到其顺时针的下一个顶点那里去，然后对于其他的点也是如此，一层一层旋转到位就行，每次都旋转四个元素.

n为偶数的时候划分如图所示，n为奇数的时候划分如图所示。所以行的范围是$$range(n//2)$$,而列的范围则是$$range((n+1)//2)$$

<img src="https://assets.leetcode-cn.com/solution-static/48/1.png" alt="fig1" style="zoom: 25%;" /><img src="https://assets.leetcode-cn.com/solution-static/48/2.png" alt="fig2" style="zoom:25%;" />

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n//2):
            for j in range((n+1)//2):
                # 这几个坐标顺序，是从最左上走顺时针一圈的四个坐标，确定第一个[i][j],然后后面的横坐标接上上一个的列坐标，列坐标等于n减去前一个的横坐标再减去1.
                # 比如第三个的列坐标n-j-1就是n减去第二个的横坐标j再减去1得到的。
                matrix[i][j],matrix[j][n-i-1],matrix[n-i-1][n-j-1],matrix[n-j-1][i] = matrix[n-j-1][i],matrix[i][j],matrix[j][n-i-1],matrix[n-i-1][n-j-1] 
```



- 方法二：对于旋转90度可以通过两次翻转来实现，第一次是对于整个矩阵上下翻转，第二次是让矩阵关于对角线翻转，就可以的得到答案。

```python
# 方法二代码：
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        # 上下翻转
        for i in range(n//2):
            for j in range(n):
                matrix[i][j], matrix[n-i-1][j] = matrix[n-i-1][j], matrix[i][j]
        # 沿对角线翻转
        for i in range(n):
            for j in range(i,n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

### [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

**思路：**从矩阵最右上角出发，target比该元素小那就是向左移动一位，target比这个元素大就向下移动一位。匹配就返回true，要是过了边界还是不行，那就是False。

`````python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        # 右上角出发
        row = 0
        col = n-1
        while 0<= row <= m-1 and 0<=col<= n-1:
            # target比这个值大，那就往下走
            if matrix[row][col] < target:
                row += 1
            # target比这个值小，那就往左走
            elif matrix[row][col] > target:
                col -= 1
            else:
                return True
        return False
`````

## 链表

### [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

**思路：**详见下面的图说的很明白。每条链表走到末端就跳到另一条链表的head去，这样子两个链表走的长度就会相等了，如果真的相交，那么会同时走到相交的结点；如果没有，最后也会走到空结点，所以无论如何返回一个结点就可以了。

<img src="https://pic.leetcode.cn/1729473968-TLOxoH-lc160-3-c.png" alt="lc160-3-c.png" style="zoom: 15%;" />

**语法补充：**在任何时候、任何地方，`None` 只有一个实例存在于内存中。因此，当两个变量（比如 $p$ 和 $q$）都等于 `None` 时，它们实际上指向内存中的同一个对象。不会陷入死循环的，别担心了。

````python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        p, q = headA, headB
        while q is not p:
            p = p.next if p else headB
            q = q.next if q else headA
        return q
````

### [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

**思路：**首先记得就只要用三个指针，一个prev指向当前操作的前一个结点，curr指向当前结点，nextnode指向当前节点的下一个结点。初始化的时候，prev初始化为None，curr当然初始化为head，只要curr不为空，那就是还有节点没被处理的，所以循环条件为curr不为空，接下来，每次将curr的next指向prev，实现局部这两个的逆向，然后prev就变为此次的curr了，因为处理完的curr就是目前的头头，curr变为原来存储的nextnode，不断循环即可。

**补充**：命名建议：迭代反转方法通常只需要两个或三个指针（`prev`, `curr`, 和临时的 `next_node`），避免使用复杂的 `tmp1/tmp2` 命名导致混淆

<img src="E:\笔记\Q206.jpg" alt="Q206" style="zoom: 50%;" />

````python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 就只要用三个变量，prev，curr，nextnode
        # prev初始化为None
        prev = None
        # 当前curr初始化为head
        curr = head
        # curr不为空那就还要操作
        while curr:
            # 要把curr的下一个存储才行
            nextnode = curr.next
            # 当前的prev成为curr的next
            curr.next = prev
            # prev变为了当前的curr
            prev = curr
            # curr变为之前存储的curr的下一个结点
            curr = nextnode
        return prev
````

### [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)

**思路：**一个比较简单的方法，直接把遍历一边把他们存在一个数组里头，然后判断是否回文就ok了。比较复杂的方法，把这个空间复杂度降到O(1)，那就是把这个链表的后半进行逆转，然后同时遍历两段链表即可。首先需要找到中点，采用快慢指针即可，同时需要注意一下，正中间的那个结点不要放到后半段去了，因为他不需要参与逆转过程；而且这里最好是去找前半段的末尾，比找第二段的开头会更加方便后续的操作。逆转过程看上个题目就ok。最后最好是把链表进行复原。

````python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # 要注意一下，这个函数返回第一部分的最后一个结点，这样子会比较方便
        # 然后还要注意一下，如果结点数目是奇数，那正中间的哪个结点要归到第一部分去，
        # 不能放到第二部分，不然就会reverse出现问题了
        def end_of_firstpart(head):
            slow, fast = head, head
            while fast.next and fast.next.next:
                fast = fast.next.next
                slow = slow.next
            return slow
        def reverse(head):
            prev = None
            curr = head
            while curr:
                next_node = curr.next
                curr.next = prev
                prev = curr
                curr = next_node
            return prev
        if not head:
            return True
        first_end = end_of_firstpart(head)
        # 第二部分的开头取为逆转后的开头
        second_head = reverse(first_end.next)
        first_head = head
        while first_head and second_head:
            if first_head.val != second_head.val:
                # 最好是在返回前进行一下复原操作
                first_end.next = reverse(second_head)
                return False
            first_head = first_head.next
            second_head = second_head.next
        # 最好是在返回前进行一下复原操作
        first_end.next = reverse(second_head)
        return True
````

四刷的代码，我觉得很优雅：

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head:
            return True
        fast, slow = head, head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        # 奇数情况下中间点给前半部分
        head2 = slow.next
        # 这里开始是第二段链表的逆转操作，返回pre是逆转后的头
        pre = None
        cur = head2
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur 
            cur = tmp
        # 一直判断，知道后面这个链表空了就行；因为奇数情况下，中间会多出来一个点，后面的会先到None
        while pre:
            if head.val == pre.val:
                head = head.next
                pre = pre.next
            else:
                return False
        return True
```

### [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

**思路：**使用快慢指针，如果有环他们两个肯定会相遇，如果没有环，那肯定会到达链表的末端也就是会出现None。

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # 空链表肯定没有环
        if not head:
            return False
        # 初始化的时候这个fast和slow一定不能指向同一个，要不然while就废了
        fast, slow = head.next, head
        # 这个地方不需要那个判断fast.next.next是否为空！！
        while fast and fast.next:
            # 有环那就肯定会相遇罗
            if fast == slow:
                return True
            fast = fast.next.next
            slow = slow.next
        # fast出现了None那就是可以结束，那就是没有环
        return False
```

感受一下这种写法，和142的初始化、判断都保持一致了，这样子更不容易出错我觉得

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # 初始化相同
        fast, slow = head, head
        while fast and fast.next:
            # 先更新，后判等
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False
```

### [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

**思路：**``fast`` 与 ``slow``都位于链表的头部，``slow``每次移动一个位置， ``fast`` 指针移动两个位置。如果链表中存在环，则 fast 指针肯定会和slow在环里面相遇。

<img src="https://assets.leetcode-cn.com/solution-static/142/142_fig1.png" alt="fig1" style="zoom: 25%;" />

假设链表中环外部分的长度为 a，``slow`` 指针进入环后与 ``fast`` 相遇走了 b 的距离。

``slow``和``fast``相遇的时候``fast ``指针已经走完了环的 n 圈，总距离为 $$a+n(b+c)+b=a+(n+1)b+nc$$

又有他们两个走过的路程是两倍的关系，所以$$a+(n+1)b+nc=2(a+b)⟹a=c+(n−1)(b+c)$$


就会发现：从相遇点到入环点的距离加上 n−1 圈的环长，恰好等于从链表头部到入环点的距离。

所以我们一直动，让 ``slow ``与 ``fast`` 相遇，这个时候设置指针 ``ptr``指向链表头部；然后，slow这个时候在相遇点，``ptr``在起点，``ptr``和 ``slow`` 每次向后移动一个位置，所以这个``slow``到环的起点的时候走的距离就是$$c+若干倍的(b+c)$$，所以总是当``ptr``正好走到环的起点的时候，``slow``也会是正好走到这里，所以就一直走，知道他俩相遇，那个点就是环的入口。

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 需要注意的是两个题目都不需要在while的条件那里考虑fast的next的next（虽然考虑也不会出错，但真没必要）
        # 因为这个while其实主要还是为了去保证不会出现去求None的next这种错误
        fast, slow = head, head
        while True:
            # fast能走到链表末尾，无环
            # 为了能直接在走到空的时候进行返回，把外层while分离了，单独用if分支来判断是否会到结尾，可以直接返回了，避免while中判断fast的，到后面还要判断循环结束的条件到底是空还是相等
            if not fast or not fast.next:
                return None
            fast = fast.next.next
            slow = slow.next
            # fast和slow相遇，跳出循环
            if fast == slow:
                break
        # 用ptr指向head
        ptr = head
        # ptr和slow相遇的地方就是环入口！！
        while ptr != slow:
            slow = slow.next
            ptr = ptr.next
        return ptr
```

### [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

**思路：**直接两个链表，每次比较当前val的大小，小的就成为结果链表的下一个结点，直到某一链表为空，跳出循环，把为空的接到结果连边后头就行。

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # 加一个假头，养成这种习惯，可以统一空表这些的情况
        prehead = ListNode()
        prev = prehead
        # 都不为空才循环
        while list1 and list2:
            if list1.val <= list2.val:
                prev.next = list1
                list1 = list1.next
            elif list1.val > list2.val:
                prev.next = list2
                list2 = list2.next
            prev = prev.next
        if list1:
            prev.next = list1
        elif list2:
            prev.next = list2
        return prehead.next 
```

### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

**思路：**就是正常相加，最主要的就是要一个额外空间存储进位，进位初始化为0，每次计算都要把两个链表当前的val和进位进行求和，和的个位为结果链表的当前结点val，十位就是新的进位。当某一个链表为空的时候，就把每次相加的数变为非空的链表的val和进位相加就行。一定要注意两点，第一点是一个链表为空的时候，不代表就可以直接把这个链表加到结果末尾了，一定要和进位一直求和，有可能会一直进位进位到最后的，比如9999999+1这种情况，每次都有进位；第二点是最后两个链表都空了的时候，要注意进位是不是0，不是0的话要把它作为新的结点加上去！

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # 假头
        prehead = ListNode(-1)
        prev  = prehead
        # 存储进位
        count = 0
        while l1 and l2:
            # 目前的和
            node_val = l1.val+l2.val+count
            # 取得和的个位作为结点值
            curr = ListNode(node_val%10)
            # 把这个结点加到结果链表上去
            prev.next = curr
            prev = prev.next
            l1, l2 = l1.next, l2.next
            # 取得和的十位作为进位
            count = node_val // 10
        # l2空了，处理l1
        while l1:
            # 和上面的while的区别就是求值的时候不要l2的值了，其他实际上是一样的
            node_val = l1.val+count
            curr = ListNode(node_val%10)
            prev.next = curr
            prev = prev.next
            l1 = l1.next
            count = node_val // 10
        # 同上面的while的逻辑
        while l2:
            node_val = l2.val+count
            curr = ListNode(node_val%10)
            prev.next = curr
            prev = prev.next
            l2 = l2.next
            count = node_val // 10
        # 链表都处理完了，一定一定要看看进位是不是0
        if count == 0:
            prev.next = None
        else:
            prev.next = ListNode(count)
            prev = prev.next
            prev.next = None
        return prehead.next
```

### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

**思路：**如果想要一次遍历实现，那就又要用到快慢指针了。注意这里是删除节点，所以不要停到那个要被删的节点上，而是要停到他的pre节点上，所以fast出发，要先走n+1个位置，这样子和slow一起走的时候，slow就可以相对而言少走一个位置，就会在fast指向None的时候指向被删的pre节点了。同时，为了免去单独考虑head为空的情况，设置dummyhead假头，这样子就不需要单独对head为空就行处理！！！！（很实用且常见的技巧！！）

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 你好，我是假头，便于处理head被删以及head为空的情况
        dummy_head = ListNode(-1)
        dummy_head.next = head
        # fast是那个最后走到None的指针
        # prev最后指向要被删的节点的后一个
        fast = dummy_head
        prev = dummy_head
        # 先出发n+1步
        for i in range(n+1):
            fast = fast.next
        # 一直到fast为None，prev就到了
        while fast:
            prev = prev.next
            fast = fast.next
        prev.next = prev.next.next
        # 别返回head了，head是有可能被删掉的
        return dummy_head.next
```

### [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

**思路：**

方法一：迭代实现

还是设置个假头，每次``tmp``指向要交换的后两个节点的前节点，所以while条件为``tmp``的``next``和``tmp``的``next.next``是否为空，如果有空的那就是最多只有单独的一个节点了，就可以结束了。

每次交换``tmp.next``和``tmp.next.next``，然后把tmp指向交换后的第二个节点，再继续循环即可。

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 假头
        dummyhead= ListNode(0)
        dummyhead.next = head
        # 初始tmp就是假头
        tmp = dummyhead
        # 后面要至少两个节点才需要交换
        while tmp.next and tmp.next.next:
            # 用node1和node2存一下，不容易出错一些
            node1, node2 = tmp.next, tmp.next.next
            # 交换！注意逻辑，不要出现断链的情况
            tmp.next = node2
            node1.next = node2.next
            node2.next = node1
            # tmp指向原来的第一个节点，也就是交换后的第二个节点
            tmp = node1
        return dummyhead.next
```

四刷的代码，思路是一样的，只是具体细节有些差别：好吧。上面的代码更优秀

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        pre = dummy
        if not head:
            return None
        node1 = head
        while node1 and node1.next:
            node2 = node1.next
            tmp = node2.next
            pre.next = node2
            node2.next = node1
            # 一开始没写这一句，while死循环了，主要是影响到了node2 = node1.next这一句
            # 这里不把node1的next修改一下，node2就指向自己了；所以三个涉及到的node的next都要及时修改
            node1.next = tmp
            pre = node1
            node1 = tmp
        return dummy.next
```

方法二：递归实现

递归结束的条件是剩下一个节点或者没有剩下节点，具体思路看代码注释

```python
425. K 个一组翻转链表class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 后面没有节点或者只有一个节点了，不需要交换了
        # 原头返回！
        if not head or not head.next:
            return head
        # head的下一个节点就是新的头
        newhead = head.next
        # 避免断链，先获得head.next
        # head.next是后面的链表返回的新表头，后面链表现在的表头是newhead.next
        head.next = self.swapPairs(newhead.next)
        # newhead.next用完了，这个时候才能去吧newhead.next更新
        newhead.next = head
        # 返回新表头即可
        return newhead
```

### [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

**思路：**~~第一次写的时候的代码逻辑真的是一坨shit。所以还是要多看看，找到最正常最合理的逻辑。~~

总体思路是每次都划出来K个节点组成的链表，对这个链表执行逆转操作。实现中为了能够翻转后能接上去，要单独保存一下下一条翻转链表的头（pre），要保存上一条翻转链表的尾（nextgroup）。

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 这个翻转思想是头插法
        def reverse(head):
            # pre设置为None，因为第一次执行while的时候pre成为了新链表的end
            # pre代表的其实是当前插入节点的next(有点绕。。)
            pre = None
            # cur表示的是将要插入到链表头部去的节点
            cur = head
            while cur is not None:
                # 存储下一个要执行头插的节点
                nextnode = cur.next
                cur.next = pre
                # cur也就是刚插入的节点成为了pre，pre也暂时是链表头
                pre = cur
                # 将下一个要插入的节点给cur
                cur = nextnode
            return pre

        dummyhead = ListNode(0)
        dummyhead.next = head
        # pre指向当前要翻转的链表的前一个结点
        pre = dummyhead
        # end指向当前要翻转的链表的最后一个结点
        end = dummyhead
        # end后面还有节点才处理
        while end.next is not None:
            count = k
            # 用个count来写稍微可以少一点代码
            while count and end:
                end = end.next
                count -= 1
            # end空了就达不到k的长度，直接break
            if not end:
                break
            # 存储下一个翻转小组的第一个结点
            nextgroup = end.next
            # 存储当前组的起始节点，也就是翻转函数传入的参数
            start = pre.next
            # 把当前的链表断开
            end.next = None
            # pre是当前交换的链表的前一个结点 ，pre的next就是翻转后的链表头，这样就接上了
            # 接上当前翻转组的头
            pre.next = reverse(start)
            # 再接上当前翻转组的尾巴
            start.next = nextgroup

            # start是本次更新的链表在更新前的head，也就是翻转后的end，
            # pre更新为start，即成为下一个翻转组的前一个结点
            pre = start
            # end初始是从pre结点出发的
            end = pre
        # 返回！！！
        return dummyhead.next
```

四刷自己写出来的代码，我觉得还不错：

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        # start从假头开始
        start = dummy
        while True:
            count = k
            # 当前段的前一个节点
            pre = start
            # 当前段的head
            cur = pre.next
            # 结束时start到达当前翻转端的最后一个节点
            for i in range(k):
                if start.next is not None:
                    start = start.next
                    count -= 1
                else:
                    break
            if count > 0 :
                return dummy.next
            # 下一段的开始
            next_start = start.next
            # 当前段和后面断开
            start.next = None
            # 执行当前段的翻转
            # 不用假头了，用当前段的head：cur
            new_pre = cur
            # new_cur用head的下一个
            new_cur = cur.next
            # 翻转逻辑和之前一样
            while new_cur:
                tmp = new_cur.next
                new_cur.next = new_pre
                new_pre = new_cur
                new_cur = tmp
            # new_pre是翻转后的新头
            # pre是当前段的前一个节点；和前面段接上
            pre.next = new_pre
            # cur是当前段翻转前的head，也即是翻转后的最后一个；和后面段也接上
            cur.next = next_start
            # start更新为当前段翻转后的最后一个节点
            start = cur 
```

五刷的代码，把四刷的代码里面的reverse变成函数，其他基本不变，卡那个  while pre_tail.next and count > 0:半天，一直没加next

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 没有假头的reverse
        def reverse(link1):
            pre = link1
            cur = pre.next
            while cur:
                tmp = cur.next
                cur.next = pre
                pre = cur
                cur = tmp
            return pre

        dummy = ListNode(0)
        dummy.next = head
        pre_tail = dummy

        while True:
            count = k
            # tmp存一下上一段的尾节点
            tmp = pre_tail
            # 这一段没逆转前的头节点
            cur_head = tmp.next
            #  一定是pre_tail.next;要不然就把if count > 0 :改成if count > 0 or not pre_tail :
            while pre_tail.next and count > 0:
                pre_tail = pre_tail.next
                count -= 1
            # 节点不足K个了，return
            if count > 0 :
                return dummy.next
            # 这个时候的pre_tail已经到达此次逆转的链表尾节点，改个名，避免误解
            cur_tail = pre_tail
            # 存一下后一段的头节点
            nxt_head = cur_tail.next
            # 和下一段断开
            cur_tail.next = None
            # 翻转后的新头节点
            newhead = reverse(cur_head) 
            # 和前面一段接上
            tmp.next = newhead
            # 和后面一段接上
            cur_head.next = nxt_head 
            # 只需要更新pre_tail；因为while pre_tail.next and count > 0:开始前只涉及到pre_tail就行
            pre_tail = cur_head

        return dummy.next
```

### [138. 随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/)

**思路：**用字典来帮助实现。字典的key为原来链表中的节点，value为对应的复制的节点。第一次遍历建立字典，同时复制val值，并将next和random都设置为None。

第二次遍历完善next和random指针，比如建立的字典为``node``，原节点为p，那么复制的节点就是``node[p]``；原节点的next是``p.next``，复制节点的next就是``node[p.next]``。random同样是这样操作。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return head
        p = head
        copy_node = dict()
        while p:
            copy_node[p] = Node(p.val, None, None)
            p = p.next
        p = head
        while p:
            # 要注意判断一下是否为None，否则会导致在字典中找不到这个key
            if p.next:
                copy_node[p].next = copy_node[p.next]
            if p.random:
                copy_node[p].random = copy_node[p.random]
            p = p.next
        return copy_node[head]
```

### [148. 排序链表](https://leetcode.cn/problems/sort-list/)

**思路：**自顶向下的归并排序的实现，定义find_mid()函数找到链表中点，并在此断链，递归调用并对分开的链表进行排序，再进行merge操作即可。

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 该函数用于找到链表的中间节点，并以此节点断开链表
        # 为了相对分的更加均衡，以slow为第二条链表的头节点，要存储一下slow的pre节点，方便断链
        def find_mid(head):
            slow = fast = head
            while fast and fast.next:
                pre = slow
                slow = slow.next
                fast = fast.next.next
            pre.next = None
            return slow
        # 该函数用于合并两个有序链表
        def merge_lists(head1, head2):
            cur = dummy = ListNode()
            while head1 and head2:
                if head1.val <= head2.val:
                    cur.next = head1
                    head1 = head1.next
                else:
                    cur.next = head2
                    head2 = head2.next
                cur = cur.next
            cur.next = head1 if head1 else head2
            return dummy.next
        # 该函数用于实现排序的递归逻辑
        def sort_lists(head):
            # 如果为空或者只有一个节点，就无需sort了，返回head即可
            if not head or (not head.next):
                return head
            # 一条链表断开为两条，head2是第二条的head
            head2 = find_mid(head)
            # 分别对两条执行sort排序
            head = sort_lists(head)
            head2 = sort_lists(head2)
            # 合并排序后的链表
            return merge_lists(head, head2)
        # 这里忘记写了，卡了我好久。。。
        return sort_lists(head)
```

### [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

**思路：**用一个``val-list``列表存储每个子链表待处理的第一个节点的值，每次去获取最小的值，再根据他在``val_list``中的索引对应找到它属于第几个子链表，将该子链表第一个节点作为排序链表的next即可，并更新val_list中的值为该子链表的下一个节点的val。同时为了避免处理各子链表长短不一导致的不同结束时间，如果到了``None``就把``val_list``中的值更新为``float('inf')``，直到``val_list``中的min都为``float('inf')``则说明都处理完了。

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 子链表的数目
        k = len(lists)
        # 存储每个子链表待处理的第一个节点的值
        val_list = []
        # dummy处理
        dummy = cur = ListNode()
        # 初始化val_list
        for i in range(k):
            if lists[i]:
                val_list.append(lists[i].val)
            else:
                val_list.append(float('inf'))
        # k为0则是[]的情况，返回None即可
        if k==0:
            return None
        # 获取待处理子链表的minval
        min_val = min(val_list)
        # 只要最小值不是inf就还有没处理的
        while min_val != float('inf'):
            # 获取min_val在列表中的索引，该索引等于子链表在列表中的索引
            index = val_list.index(min_val)
            # 更新结果链表的next
            cur.next = lists[index]
            cur = cur.next
            # 更新待处理子链表的节点
            lists[index] = lists[index].next
            # 更新val_list
            val_list[index] = lists[index].val if lists[index] else float('inf')
            # 更新min_val
            min_val = min(val_list)
         
        return dummy.next
```

### [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)

**思路：**~~有个屁的思路~~     要O(1)时间复杂度，反正hash跑不了，又要方便在head插节点，在尾部删节点，考虑双向链表。为了方便头尾操作dummyhead，dummytail又跑不了。

首先定义一下双链表的节点类，这里和一般的不同，有key和value。然后进行LRUCache的初始化，假头假尾安排上，dict字典安排上，还要一个size来追踪现在存了多少节点了。

然后链表主要需要三个操作：把节点移到链表头；删除链表尾巴；表头添加节点（比如不存在的key要加进去）。把这三个都写好函数，双链表要搞清楚指针！不要绕晕出现断链就行。

现在就可以实现LRUCache了，具体看代码注释。

```python
# 需要借助哈希表和双向链表（实际上这是LRUCache源码的实现方式）
class Dnode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.next = None
        self.pre = None
class LRUCache:
    # init()中head，tail作为假头假尾，并连接为一个双向链表
    # 同时初始化一个hashkey字典，便于O(1)查找：字典的key是节点的key，value是对应的节点
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.head = Dnode()
        self.tail = Dnode()
        self.hashkey = dict()
        self.head.next = self.tail
        self.tail.pre = self.head
        self.size = 0
	# 在dict中找key，有就返回，并且在返回前把他移到head去
    def get(self, key: int) -> int:
        if key in self.hashkey:
            self.moveTohead(self.hashkey[key])
            return self.hashkey[key].value
        else:
            return -1
	# put，如果在字典中，只需修改value，并移到head去；
    # 如不在，先创立新节点，加到头部去，然后一定记得更新字典
    # 因为有加节点，所以要判断size是不是超过了。超过记得删tail。同时更新字典！！
    def put(self, key: int, value: int) -> None:
        if key in self.hashkey:
            self.hashkey[key].value = value
            self.moveTohead(self.hashkey[key])
        else:
            node = Dnode(key, value)
            self.hashkey[key] = node
            self.addTohead(node)
            self.size += 1
            if self.size > self.capacity:
                self.hashkey.pop(self.tail.pre.key)
                self.deltail(self.tail.pre)
                self.size -= 1
                
    # 主要三个操作：把节点移到链表头；删除链表尾巴；表头添加节点（比如不存在的key要加进去）
    def moveTohead(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre
        node.next = self.head.next
        self.head.next.pre = node
        node.pre = self.head
        self.head.next = node

    def addTohead(self, node):
        self.head.next.pre = node
        node.next = self.head.next
        self.head.next = node
        node.pre = self.head

    def deltail(self, node):
        self.tail.pre = node.pre
        node.pre.next = self.tail 
```

## 二叉树

### [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

**思路一：递归实现**

按照左中右即可，递归访问left，输出中的val，递归访问right。递归终止条件为root为空。

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def dfs(root):
            if not root:
                return 
            dfs(root.left)
            ans.append(root.val)
            dfs(root.right)
        dfs(root)
        return ans
```

**思路二：迭代实现**

其实整体的思路是很清晰简单的，无非就是左中右，所以左边有就入栈，到空了，那就出栈一个并访问，然后转为右节点。

我的难点在于初始化：``cur = root ``和``while``的条件判断：``while cur or stack``，实在不行就直接记忆吧。

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node_stack = []
        ans = []
        cur = root
        # 当前节点不为空或者栈中还有节点就继续遍历
        while cur or node_stack:
            # cur不为空就要入栈
            while cur:
                node_stack.append(cur)
                cur = cur.left
            # cur空了跳出while，cur变为栈中弹出的节点
            cur = node_stack.pop(-1)
            # 这个cur没有左节点了，到访问的时候了
            ans.append(cur.val)
            # 访问完了，按照中序遍历的“左中右”接下来该处理cur的右子树了
            cur = cur.right
        return ans
```

### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

**思路：**莫名其妙地就递归出来了.......对左子树调用求下深度，对右子树调用求下深度，返回两者最大再加上根的1个深度。递归结束的条件就是root为空返回0。

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 多看看理解理解为什么选择这个情况作为递归终止条件。
        if not root:
            return 0
        ldepth = self.maxDepth(root.left)
        rdepth = self.maxDepth(root.right)
        return max(ldepth, rdepth) + 1
```

### [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

**思路：**还是递归解决，很开心找到了递归终止的正确条件，分别递归翻转左子树，翻转右子树，再把翻转后的右子树接到root的左边，反转后的左子树接到root的右边。但是有个要注意的地方，用下面注释掉的代码实现会出现问题，一定需要用到临时变量来存储。

如果不用临时变量，执行完``root.left = self.invertTree(root.right) ``之后，``root.left``就已经被修改了，再去执行``root.right = self.invertTree(root.left)``的时候，这里面的``root.left``就已经不是想改变的那个原来的左子树了！！！

所以一定要注意现在修改的东西，后面有没有用到啊！！！！

````python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return
        # root.left = self.invertTree(root.right) 
        # root.right = self.invertTree(root.left)
        right =  self.invertTree(root.right)
        left =  self.invertTree(root.left)
        root.right = left
        root.left = right
        return root
````

### [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

**思路一：递归实现**

递归终止条件：首先要明确这里check到底传什么参数，一开始我写了个``if not root``的条件，这个地方实际上最好是写传两个参数的check函数，所以排除，针对两个节点的条件就比较明了，三种情况：

- 两个空，对称（实际这个就对应一个节点所写的``if not root``）；
- 只有一空，不对称；
- 都不空，但值不相等，不对称。(为什么没有两个不空，且值相等，则对称的判断？其实和之前是一样的，如果只有一个root没有左右孩的情况，还要单独去处理了，很不好，实际上已经被包含在两个都空的情况了。)

递归root1的左和root2的右，root1的右和root2的左是否都对称。

主函数中实现return调用check函数，``root.left root.right``

```python
class Solution(object):
    def check(self, root1, root2):
        # 均为空在这里就返回了
        if not root1 and not root2:
            return True
        # 全空的情况已被排除
        # 所以这里只有全非空，和单独一个空的情况，所以可以这样判断
        if not (root1 and root2):
            return False
        if root1.val != root2.val:
            return False
        return self.check(root1.right, root2.left) and self.check(root2.right, root1.left)
    def isSymmetric(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        return self.check(root.left, root.right)
```

**思路二：迭代实现**

其实很像层次遍历的写法，只不过每次入队两个元素，左右交叉。详见代码注释

````python
class Solution(object):
    def isSymmetric(self, root):
        # 题目中节点数至少有一个，所以可以先把左右孩入列了（没这个条件，可以选择入两个root）
        queue = [root.left, root.right]
        while queue:
            # 一次pop两个出来
            node1 = queue.pop(0)
            node2 = queue.pop(0)
            # 这里的判断和递归是一样的
            if not node1 and not node2:
                continue
            if not (node1 and node2):
                return False
            if node1.val != node2.val:
                return False
            # node1的左和node2的右一对比较
            queue.append(node1.left)
            queue.append(node2.right)
            # node1的右和node2的左一对比较
            queue.append(node1.right)
            queue.append(node2.left)
        return True
````



### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

**思路：**用一个全局的ans来不断存储求各个节点的深度的过程中的路径所经过的最多的节点数！depth函数是用于递归求解各个节点的深度。

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # ans存储的是node开始的节点数目，如果返回最后的路径长度要-1
        self.ans = 1
        # 递归函数求depth（这个depth的定义是从node出发的最长的路径上节点的数目）
        def depth(node):
            # 节点为空，深度为0
            if not node:
                return 0
            # L为左孩子的depth
            L = depth(node.left)
            # R为右孩子的depth
            R = depth(node.right)
            # 更新ans为两者中的较大值
            self.ans = max(self.ans, L+R+1)
            # return self.ans❌ 这里返回的是node的深度，那就是左和右的最大深度+1（自己）
            return max(L,R)+1
        # 递归求深度
        depth(root)
        # 返回ans-1
        return self.ans-1
```

### [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

**思路：**如果是普通的层序遍历那就easy到不行了，这里主要是要分层输出。

最主要的四个变量。ans 用于存储最终要输出的结果，queue用于存储当前层要处理的节点，tmp用于在while循环中暂时存储某一层的节点的值，queue1用于在循环中暂时存储queue中节点所有下一层的孩子，也就是下一批要处理的对象。

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # ans 用于存储最终要输出的结果
        ans =[]
        # queue用于存储当前层要处理的节点
        queue = []
        if not root:
            return queue
        queue.append(root)
        while queue:
            # tmp用于暂时存储某一层的节点的值
            tmp = []
            # queue1用于暂时存储queue中节点下一层的孩子
            queue1 = []
            # L为当前要处理的层节点数目，也就是queue中节点数
            L = len(queue)
            for i in range(L):
                # val存到tmp中去
                tmp.append(queue[i].val)
                # 左孩右孩通通存到queue1中去
                if queue[i].left:
                    queue1.append(queue[i].left)
                if queue[i].right:
                    queue1.append(queue[i].right)
            # 把tmp中存储的这一层的值列表加到结果列表中去
            ans.append(tmp)
            # queue1赋值给queue，下一个处理
            queue = queue1
        return ans
```

### [108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

**思路：**递归实现。最直观地理解平衡二叉树就是root都找mid位置的点就ok，所以递归寻找新的索引范围中的mid，作为root的left和right，递归终止条件为节点在数组中的索引$$left>right$$。而且对于缺失的左右孩子由于TreeNode在初始化的时候左右孩子都是初始化为None，所以无需额外处理。

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def balance(left, right):
            if left > right:
                return
            # 找数组中点索引
            mid = (left + right) // 2
            # nums[mid]为根
            root = TreeNode(nums[mid])
            # 左子树的根在新范围left和mid-1中寻找中点
            root.left = balance(left, mid-1)
            root.right = balance(mid+1, right)
            return root
        return balance(0, len(nums)-1)
```

### [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

**思路一：递归实现**

helper函数，lower为下界，upper为上界，每次去判断是否在（lower，upper）区间中。递归终止条件，node为空返回True，node的值不在区间里面，返回False，然后递归判断左右子树。初始时，lower和upper均设置为无穷即可。

需要注意一下左右子树判断时的上下界设置，见注释。

````python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True
            if node.val <= lower or node.val >= upper:
                return False
            # 这种写法把lower和upper强制更新为无穷，是错误的，因为这个地方是递归啊，你左子树看似lower可以是负无穷
            # 但实际上这个左子树已经是被递归的了，本身可能是谁的右子树，所以更新的时候要继承lower
            # return helper(node.left,float('-inf'),node.val) and helper(node.right,node.val,float('inf'))
            return helper(node.left,lower,node.val) and helper(node.right,node.val,upper)
        return helper(root)
````

**思路二：迭代实现**

利用BST的中序遍历序列是单调递增的来实现，在普通版本的中序遍历的实现上，每次只需记录上一个遍历到的节点的val值与本次作比较，不符合就可以中止了。

......写个中序遍历写的乱七八糟.......

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = []
        # preval存储中序遍历的前一个结点的val
        preval = float('-inf')
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop(-1)
            if preval >= root.val:
                return False
            preval = root.val
            root = root.right
        return True
```

### [230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

**思路：**依旧继承迭代中序遍历的实现代码，k等于0的时候return即可。

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right
        return None
```

### [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

**思路：**继承102.二叉树的层序遍历的代码。改动的地方就是tmp不需要存储所有一层的val了，每次更新就ok，遍历完直接把最后一次的tmp加到结果列表就OK。其余代码基本不需要动。

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        queue = []
        if not root:
            return ans
        queue.append(root)
        while queue:
            queue1 = []
            for i in range(len(queue)):
                if queue[i].left:
                    queue1.append(queue[i].left)
                if queue[i].right:
                    queue1.append(queue[i].right)
                tmp = queue[i].val
            ans.append(tmp)
            queue = queue1
        return ans
```

### [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)

**思路一：**比较easy的方法，直接先得到先序遍历的node列表，再把它展开平铺。

````python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.sequ = []
        # 递归实现的先序遍历
        def preorder(root):
            if not root:
                return
            self.sequ.append(root)
            preorder(root.left)
            preorder(root.right)
        preorder(root)
        # 对先序遍历的node列表进行建树操作
        for i in range(len(self.sequ)):
            self.sequ[i].left = None
            self.sequ[i].right = self.sequ[i+1] if i+1 < len(self.sequ) else None
````

**思路二：**看的官方题解，可以把空间复杂度降到O(1)，思路get到了，但严格的算法正确性我无法理解证明。

核心是寻找前缀结点，前序遍历是“根左右”，如果没有左子树，那么这里是不需要执行平铺操作的。

如果有左子树，那么我们可以把它的右子树整个接到左子树~~进行先序遍历的最后一个结点的右指针上去~~**这个地方实际上是最右的结点的右指针，而不是最右的叶子节点的右指针，也就是说可能此时左边还有孩子，但不需要在这里处理，在前面的迭代中会先把他们处理掉的！！！**然后再把整个左子树接到root的右指针上去，同时把左指针置为None。处理完当前结点，下一个处理的应该是变换完成后的右孩子结点，不断执行，直到右孩空了。（无法理解的就是这里会不会有漏掉的或者说怎么样的。。。其他的能get）

```python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        cur = root
        while cur:
            if cur.left:
                # 有左孩，nxt用于便于后续将变换后的整个左边移到cur的右指针
                # pre用于寻找左子树的最右孩子
                nxt = pre = cur.left
                while pre.right:
                    pre = pre.right
                # 把cur的右孩子接到pre的右指针
                pre.right = cur.right
                # 将变换后的整个左边移到cur的右指针
                cur.right = nxt
                # 左为None
                cur.left = None
            # 继续处理变换后的cur的第一个右孩
            cur = cur.right
```

### [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

**思路：**建立一个辅助函数，传入四个索引，分别是当前前序遍历的起点、终点和中序遍历的起点、终点的索引，通过前序遍历的起点就是当前的根的性质，得到root，并提供root的值获取其在中序序列中的索引，到此可以在中序遍历中获取到左子树的遍历数组的长度。有了这个长度之后，就可以去得到左右子树的前序和中序遍历的序列了，于是进行递归的建树操作，递归建立左子树和右子树即可。

同时，在其中为了方便通过root的值获取到root在中序序列中的索引，建立字典：index，以中序遍历的结果值为key，每个值的索引为value建立index字典。同时注意递归结束的条件，先序遍历的序列的起点的索引大于终点的索引时结束递归。

**语法补充：**``index = {elment:i for i,elment in enumerate(inorder)}``这种建立字典的方式要熟悉！！！

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 左 中 右
        # 中 左 右
        def helperbuild(pre_L, pre_R, in_L, in_R):# pre_L是先序遍历的结果列表的第一个元素的索引，后面三个同理
            if pre_L > pre_R:
                return
            # 前序遍历中第一个点就是根结点(pre_root是root在先序遍历中的索引)
            pre_root = pre_L
            # 获取root在中序遍历中的索引
            in_root = index[preorder[pre_root]]
            root = TreeNode(preorder[pre_root])
            # 算出左子树中的节点数
            num_L = in_root - in_L
            # 递归建立左子树
            root.left = helperbuild(pre_root+1, pre_root+num_L, in_L, in_root-1)
            # 递归建立右子树
            root.right = helperbuild(pre_root+num_L+1, pre_R, in_root+1, in_R)
            return root
        n = len(preorder)
        # 建立index便于获取root在中序遍历序列中的索引
        index = {elment:i for i,elment in enumerate(inorder)}
        return helperbuild(0, n-1, 0, n-1)
```

### [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/)

**思路：**哇，这鬼题目我觉得好难。类似于两数之和那个鬼题目。一边遍历一边存储前缀和，一边计算符合条件的情况，这里有几个需要注意的点。（递归真是有点难理解。。。其实实际上是从叶子节点那一坨开始s每次传递的值就变回之前没有加上node值得情况）

1.   cnt[0] = 1 还是为了便于记录从根结点出发的路径是符合情况的情况
2.   在递归处理node的左右子树之前，cnt[s] += 1，但是递归语句之后要注意进行回溯的操作，即cnt[s] -= 1

````python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.ans = 0
        # cnt用于存储到目前结点结束时的出现的所有前缀和
        cnt = defaultdict(int)
        # 为了便于记录从根结点出发的路径是符合情况的情况
        cnt[0] = 1 
        def dfs(node, s):
            if not node:
                return 
            # s为从根节点（其实更准确的说，是从递归开始的点，只不过递归入口确实是root）
            # 到目前的node，这一条路径的总和（因为这个s是一路累加下去的，也不需要什么回溯操作）
            s += node.val
            # s - targetSum是我们需要的前缀和
            # 到cnt中去寻找有多少个，加到ans中
            self.ans += cnt[s - targetSum]
            cnt[s] += 1
            # 前缀和为s的情况加到cnt里面
            # 递归处理左子树和右子树
            dfs(node.left, s)
            dfs(node.right, s)
            # 注意回溯一下前缀和为s的情况，因为这个地方在递归处理了node的左子树和右子树之后
            # 那就需要去看看没有node的情况了对吧，那对应的cnt也要删除才是对的
            cnt[s] -= 1
        dfs(root, 0)
        return self.ans
````

### [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

**思路：**都想录个视频讲解了...

递归解决问题，首先递归终点就是root为空，那返回None，为了统一，直接返回root也是一样的；然后root和p或者是q相等，那也是直接返回root。

对左子树进行递归找最近的公共祖先，同样对右子树也执行这个操作，如果左右子树返回的都不是空，那就说明找到了p或者q，那当前的root就是p和q的最近公共祖先；如果只有一个不是空，那就把这个找到的p或者q返回；如果都是空的情况下，返回的还是None，left空，else right返回right这个None。

这个地方理解的几个难点emmm越想越迷糊，不要去递归他了，就从最平凡的思考，哪几种情况递归直至，left和right有哪些情况要分别怎么处理。然后有个就是找到了这个最近的公共祖先之后，还得一直往上传递，不会出错吗？-----不会的，这样的情况下已经找到祖先了，那么另外一棵树一定是每次返回None，就变成了left和right一空一不空，然后一直返回不空的那个，也就是最近公共祖先一直不变地传递到函数地入口去了！！

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # root为空或者root和要找的相等了，直接返回root即可。
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 情况一：左右都不空
        if left and right:
            return root
        # 情况二三：一空或两空
        return left if left else right
```

### [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

**思路：**递归求解每个节点开始的路径中和最大的那一条，并且是允许不到叶子节点的，所以如果求出来小于0直接设为0，也就是不要任何节点就好了。然后这个题目中所定义的最大路径和，是可以两边的，求这个的过程在前面提到的递归中解决掉，一边求单边最大，左右单边最大求出来后，就可以更新此节点的最大路径和了，再用全局变量globalmax来不断存储并更新，直到递归结束，globalmax中存储的就是最大路径和了。（不一定过root哦）

````python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.globalmax = float('-inf')
        # max_single_sum是用于求每个结点开始的路径中和最大的路径（不一定会到叶子节点）
        def max_single_sum(root):
            # 为空递归终止，路径为0
            if not root:
                return 0
            # leftgain是左子树和最大的路径，如果他小于0，那就直接舍弃掉（因为也是可以不到叶子节点的）
            leftgain = max(max_single_sum(root.left), 0) 
            rightgain = max(max_single_sum(root.right), 0)
            # cursum是求当前节点此题定义的最大路径和，也就是可以同时有左右路线
            cursum = root.val + leftgain + rightgain
            # globalmax就是更新存储此题所定义的那个最大路径和
            self.globalmax = max(cursum, self.globalmax)
            # 这个函数求的是常规节点的最大路径和，所以是单边的路径，不要返回出错了！！
            return root.val + max(leftgain, rightgain)
        max_single_sum(root)
        return self.globalmax
````

## 图论

### [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

**思路：**其实就是最基本的图的dfs就行，``dfs``函数对图进行遍历，图里头的方向就是上下左右四个都要考虑，然后避免重复计算，访问了的岛屿，将``‘1’``置为``‘0’``，，遇到上下左右的坐标合法，且未进行访问的递归进行访问即可。 

然后在主函数里面，遍历整个图，遇到等于1 的就将岛屿数加一即可，因为每遇到一次1就代表需要开启一次``dfs``的递归操作，也就是新的岛屿。

尤其要注意的一个点，这里面的1都是字符，不要写``grid[i][j]  ==  1``，而是``grid[i][j] == '1'``

```python
class Solution:
    def dfs(self, grid, r, c):
        # 访问了就置0
        grid[r][c] = 0
        rn = len(grid)
        cn = len(grid[0])
        # 四个方向都要考虑
        for x,y in [(r,c-1),(r,c+1),(r-1,c),(r+1,c)]:
            if 0 <= x < rn and 0 <= y < cn and grid[x][y]=='1':
                # 递归DFS
                self.dfs(grid,x,y) 
    def numIslands(self, grid: List[List[str]]) -> int:
        # ans存储岛屿数量
        ans = 0
        # 全图遍历
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                # 存在未访问的1
                if grid[i][j] == '1':
                    # 开个新的DFS
                    self.dfs(grid,i,j)
                    ans += 1
        return ans
```

### [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

**思路：**多源BFS。既然来到了BFS那就考虑用队列来进行实现。所以此处不该跑到递归去了，应该是while结合队列。

首先遍历grip统计新鲜橘子的数量和，并将当前所有的腐烂的橘子（也就是BFS的多个起点）存储于列表中。开始正式BFS，每次对当前队列中所有的腐烂橘子作为起点开始向四个方向DFS。不断循环，直到腐烂橘子队列空了，或者新鲜橘子到了0，就返回。

`````python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rn, cn = len(grid), len(grid[0])
        # 统计新鲜橘子数和腐烂的橘子的row和col
        rotted = list()
        fresh = 0
        for row in range(rn):
            for col in range(cn):
                if grid[row][col] == 1:
                    fresh += 1
                elif grid[row][col] == 2:
                    rotted.append((row,col))
        # 记录BFS轮数
        minute = 0
        while rotted and fresh != 0:
            minute += 1
            # 此次的BFS的起点数目
            n = len(rotted)
            for i in range(n):
                x,y = rotted.pop(0)
                # 考虑四个方向的BFS
                if x+1 < rn and grid[x+1][y] == 1:
                    # 新鲜橘子腐烂
                    grid[x+1][y] = 2
                    # 新鲜橘子数减一
                    fresh -= 1
                    # 这个新腐烂的橘子加入rotted中，是下一轮的起点之一
                    rotted.append((x+1,y))
                if 0 <= x-1 and grid[x-1][y] == 1:
                    grid[x-1][y] = 2
                    fresh -= 1
                    rotted.append((x-1,y))
                if y+1 < cn and grid[x][y+1] == 1:
                    grid[x][y+1] = 2
                    fresh -= 1
                    rotted.append((x,y+1))
                if 0 <= y-1 and grid[x][y-1] == 1: 
                    grid[x][y-1] = 2
                    fresh -= 1
                    rotted.append((x,y-1))
        # 循环结束了，fresh还没到0，则返回-1
        if fresh != 0:
            return -1
        else:
            return minute
`````

### [207. 课程表](https://leetcode.cn/problems/course-schedule/)

**思路：**维护三个东西：

​		1.indegree存储各个课程的入度，并且通过索引对应课程号

​		2.graph = {key：前置课程，value：所有以key为前置课程的课程}

​			目的是为了便于选择了key之后，对value里面所有课的indegree执行减一操作

​		3.zero_cour存储目前没上的且入度为0的课程号，每次都从这里面选择课程来上

所以就是首先完成上面三个东西的初始化，然后加入一个计数器来记录当前上过了的课程数，不断从zero_cour选课上，更新对应indegree和注意indegree变为0要加入到zero_cour中去，直到zero_cour空，看计数器和要上的课程数是否相等即可。

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # 记录每门课程的入度，
        # 因为课程计为0到numCourses-1，所以可以不建字典，使用列表就ok，对应索引位置存储对应课程号的入度
        indegree = [0] * numCourses
        # key为前置课程，value为以key为前置课程的课
        graph = {}
        #存储入度为0的课程的优先队列
        zero_cour = []
        # 建立好indegree和graph
        for i in range(len(prerequisites)):
            # pre_cour是前置课程
            pre_cour = prerequisites[i][1]
            # course是以pre_cour为前置课程的课程
            course = prerequisites[i][0]
            # 更新course的入度
            indegree[course] += 1
            if pre_cour in graph:
                graph[pre_cour].append(course)
            else:
                graph[pre_cour] = [course]
        # 建立好0入度课程队列
        for i in range(len(indegree)):
            if indegree[i] == 0:
                zero_cour.append(i)
        # 当前学完的课程
        cur_num = 0
        while zero_cour:
            # 选择一门入度为0的课程学习
            cur_num += 1
            cur_cour = zero_cour.pop()
            # 更新gragh中以当前课程为前置课程的indegree
            # 同时要注意一下，先判断这个if，因为可能这门课不是任何课的前置，所以导致本身就不在graph中
            if cur_cour in graph:
                for course in graph[cur_cour]:
                    indegree[course] -= 1
                    if indegree[course] == 0:
                        zero_cour.append(course)
        if cur_num == numCourses:
            return True
        else:
            return False
```

四刷这种写法用 defaultdict(list)进行了一点点优化

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        # 各课的入度
        zero_in = [0] * numCourses
        # 入度为0的课
        queue = []
        # defaultdict更方便，可以减少判断
        pre_dict = defaultdict(list)
        for i in range(len(prerequisites)):
            pre_dict[prerequisites[i][1]].append(prerequisites[i][0])
            zero_in[prerequisites[i][0]] += 1
        for i in range(len(zero_in)):
            if zero_in[i] == 0:
                queue.append(i)
        while queue:
            cur = queue.pop(0)
            numCourses -= 1
            for cours in pre_dict[cur]:
                # 不会出现重复入队，因为会变为-1，而不是一直保持0
                zero_in[cours] -= 1
                if zero_in[cours] == 0:
                    queue.append(cours)

        if numCourses == 0:
            return True
        return False
```



### [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

**思路：**我觉得难点就是在Trie树结构的理解上，每个节点就是两个东西一个is_end来判断是否结束，然后children是一个列表，有26个元素，分别代表26个字母，初始的时候都是None，当某个对应索引这个字母要插入，那None就改为TrieNode了。然后就是我脑海里面总是去和普通的树节点去对应，就感觉children是TrieNode指向的另一个节点，但实际上更准确的理解，就是把children当成这个TrieNode自身内部的一个结构就行了。



<img src="C:\Users\Leoti\AppData\Roaming\Typora\typora-user-images\image-20251104142625807.png" alt="image-20251104142625807" style="zoom:67%;" />

```python
# 自己设置TrieNode类，初始化is_en和children
class TrieNode:
    def __init__(self):
        self.is_end = False
        self.children = [None] * 26
class Trie:
    # 初始化时则实例化一个TrieNode类就ok
    def __init__(self):
        self.root = TrieNode()
    # 插入
    def insert(self, word: str) -> None:
        node = self.root
        # 对word中每个字符去找到对应索引上是否不断指向新的TrieNode
        # 为空就需要开辟新的
        # 之前有就沿着往下走就行
        for char in word:
            # 计算字符索引
            index = ord(char) - ord('a')
            # 通过把对应的None修改为TrieNode来实现插入字符
            if node.children[index] == None:
                node.children[index] = TrieNode()
            # 沿着一路往下
            node = node.children[index]
        # 到达末尾，将结束标志设为0
        # 并且实际上这个最后的TrieNode的children是26个None
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            index = ord(char) - ord('a')
            if node.children[index] == None:
                return False
            node = node.children[index]
        # 到此结束了就是存在，没结束就不存在
        return node.is_end
    # 注意一下完全相等也算作前缀
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            index = ord(char) - ord('a')
            if node.children[index] == None:
                return False
            node = node.children[index]
        # 不管结没结束都算前缀存在
        return True
```

## 回溯

### [46. 全排列](https://leetcode.cn/problems/permutations/)

**思路：**回溯的思想，每次选择一个数放到比如第一个位置，再选择一个数放到二号位置，为了这些选择都可以全面选择，每次选好后又进行撤销，去进行不同的选择，大概就是这样理解。

建议：递归本身就很难理清，还要加上回溯，所以还是一样，只去考虑最平凡的情况，不要想着到底怎么回溯怎么递归！！！！！

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        # ans存放可能的排列
        ans = []
        # first参数代表的是当前在选择first这个位置上要放哪个数字
        def backtrack(first):
            # 如果放到n这里来了，说明所有位置都放完了，当前的nums是一个排列可能（）注意不是n-1
            # 这个地方其实也就是递归终止条件
            if first == n:
                ans.append(nums[:])
            # for循环，对于first开始的到末尾的每一个位置都要进行操作
            for i in range(first, n):
                # 这个相当于first这个位置我选择放置nums[i]
                nums[first], nums[i] = nums[i], nums[first]
                # 继续考虑first+1的位置放什么东西
                backtrack(first+1)
                # 以前面的first放nums[i]的基准下不断递归，放好了后头的数字
                # 那么我们就需要注意first这个位置还可以放其他的数字
                # 为了放其他的数字，就需要进行恢复处理，也就是回溯
                nums[i], nums[first] = nums[first], nums[i]
            return
        backtrack(0)
        return ans
```

### [78. 子集](https://leetcode.cn/problems/subsets/)

**思路：**递归+回溯，递归终止条件为所有元素都处理完了，即索引到了n（nums数组的长度）。每次分为选择当前元素和不选择两种情况，并不断向后递归。

**语法补充：**在python里面所有东西都是传递引用，只不过有些东西是不可变的（如int，tuple，str），所以看起来变成了值传递，所以，在ans中append的时候，一定要是暂时的tmp的切片，也就是tmp的浅复制，要不然引用传递的话，后续tmp给回溯到空列表了，结果就完全不对了。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        self.ans = [] # 最终的答案
        self.tmp = [] # 临时存储
        def backtrack(index):
            if index == len(nums):
                # self.ans.append(self.tmp)这样子写的话返回的全是空列表！！因为tmp直接append是把他的引用append了
                # 后续tmp回溯的时候ans里面一样也会回溯！！
                self.ans.append(self.tmp[:])
                return
            # 选择当前的元素
            self.tmp.append(nums[index])
            # 继续处理下一个index上的元素的取舍
            backtrack(index+1)
            
            # 把选择的当前元素pop掉，也就是不选择
            self.tmp.pop(-1)
            # 这里需要同样地去处理index+1的元素，而不是回溯后就不管了
            backtrack(index+1)
            return
        backtrack(0)
        return self.ans
```

### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

**思路：**还是递归+回溯，先建立phoneMap这个数字和字母对应字典，依旧对digits中每个位置上的元素进行选择，因为这个单个数字对应多个字母，所以在for循环里面进行回溯，然后这个地方和上面那个题目的区别还有一个，就是只有一处递归。

全排列其实也是一次递归，我个人认为区别就是，全排列和字母组合都是要求不能说哪个数字我就不要了，但是子集这题是可以的。所以子集在回溯后，相当于不要这个位置的字母、数字了，然后马上进一步递归处理。而全排列和电话组合在回溯发生后，不能直接去处理下一个位置，因为这个位置一定要选择一个数字、字母！！

**语法补充：**``''.join(tmp)``：将tmp列表中的元素，通过``''``中的符号进行连接，这里是空格。

```python
class Solution(object):
    def letterCombinations(self, digits):
        ans = []
        tmp = []
        n = len(digits)
        # 数字字母对应字典
        phoneMap = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        def backtrace(index):
            # index到n就是处理完了，加到ans中去
            if index == n:
                ans.append(''.join(tmp))
                return 
            # 每个数字对应的字母有好几个，所以用for循环来对每个字母都要尝试选择
            for char in phoneMap[digits[index]]:
                # tmp中加入这个字母
                tmp.append(char)
                # 递归处理下一个数字对应的字母
                backtrace(index+1)
                # 回溯到选择这个字母之前，然后去换一个此数字对应的字母
                tmp.pop(-1)
        backtrace(0)
        return ans
```

### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

**思路：**递归终止条件注意是``cur_target``为0，那就是刚好，``ans``加上``tmp``的切片。如果小于0了，那就说明当前方案不可行，直接return。

然后主函数for循环里面的``i``的范围是``index``到``n``，``tmp``加上当前的数值后，递归处理，此处注意``index``从``i``开始，因为可以重复！，对应的cur_target减去当前值，然后就是进行回溯了。

````python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        n = len(candidates)
        ans = []
        tmp = []
        def backtrack(index, cur_target):
            # tmp是满足条件的
            if cur_target == 0:
                ans.append(tmp[:])
            # tmp这条路走不通了
            if cur_target < 0:
                return
            for i in range(index, n):
                # 选择当前值
                tmp.append(candidates[i])
                # 递归处理下一个选择，但是选择可以重复！
                # 不是backtrack(index, cur_target-candidates[i])！要不然会出现重复的组合，要保证不能选择i之前的啊
                backtrack(i, cur_target-candidates[i])
                # 回溯到没有选择此数值的情况去
                # 而且我本来担心回溯了之后，下次又选了这个值，但实际上不会，因为for循环里面的话会强制到下一个了
                # 简单来说就是这次选择了他，下次你才有机会选择它，否则下次不会再选他
                tmp.pop(-1)
        backtrack(0, target)
        return ans
````

### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

**思路：**为什么这个地方没有for循环可是也没有在回溯后调用backtrack函数呢？事实上这里类似一个隐式的for循环，因为只有两个选择，要么左括号，要么右括号，左括号选择后进行回溯，相当于还是在此位置进行选择。所以对于任何一个位置，首先尝试左括号是否可以在此处放置，在考虑右括号是否可以在此处放置，都有考虑到。

````python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        tmp = []
        #  left是左括号的数目，right是右括号的数目
        def backtrack(left, right):
            # tmp长度到达2n就ok
            if len(tmp) == 2*n:
                ans.append(''.join(tmp))
            # 当左括号的数目少于n时，可以选取左括号
            if left < n:
                tmp.append('(')
                backtrack(left+1, right)
                tmp.pop(-1)
            # 回溯后，此位置上，如果此时right是少于left的那么此位置可以选取右括号
            if right < left:
                tmp.append(')')
                backtrack(left, right+1)
                tmp.pop(-1)
        backtrack(0,0)
        return ans
````

### [79. 单词搜索](https://leetcode.cn/problems/word-search/)

**思路：**``dfs(i, j, index)``用来进行深度递归和剪枝，详细实现见代码注释。

对board每个位置遍历均调用一次dfs。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        nr = len(board)
        nc = len(board[0])
        nw = len(word)
        # i,j是当前判断的字符的索引，即board[i][j]
        # index是当前匹配到word的字符的索引
        def dfs(i, j, index):
            # 如果i或者j越界了，或者board此位置的值和word中现在要去匹配的不一样，返回False
            if not 0 <= i < nr or not 0 <= j < nc or board[i][j] != word[index]:
                return False
            # word的最后一个字母也匹配成功了，返回True！
            if index == nw-1:
                return True
            # 把这个位置置为空，避免重复使用了
            board[i][j] = ''
            # or连接，有任何一个成功的就ok
            res = dfs(i+1, j, index+1) or dfs(i, j+1, index+1) or dfs(i-1, j, index+1) or dfs(i, j-1, index+1) 
            # 回溯（只有匹配成功才会有置空，所以回溯的时候就赋值为word的index对应字母就ok）
            board[i][j] = word[index]
            return res
        
        # board每个位置都作为dfs起点遍历
        for i in range(nr):
            for j in range(nc):
                if dfs(i,j,0):
                    return True
        return False
```

### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

**思路：**递归终止条件为对整个字符串处理完，也就是待处理的为n这个索引开始的就好了。然后利用for循环，对传入的``index``开始到末尾的每个地方都尝试进行分割，如果是回文串，那就加到tmp中，并且递归处理从该切割片段的后一个索引开始的字串，在加上一个回溯的操作即可。

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        # 最终的结果，ans中的元素是tmp列表
        ans = []
        tmp = []
        # dfs(index)代表寻找[index:]的分割方法
        def dfs(index):
            if index == n:
                ans.append(tmp[:])
                return  
            # 对于从index开始到结束位置的分割点都要考虑                         
            for i in range(index, n):
                # t暂时表示此时的分割
                t = s[index:i+1]
                # 如果找到一个合法的分割
                if t == t[::-1]:
                    # 则把t加到tmp这个存储分割片段的列表中
                    tmp.append(t)
                    # 处理分割片段后面的字符串
                    dfs(i+1)
                    # 回溯此次分割
                    tmp.pop(-1)
        dfs(0)
        return ans
```

### [51. N 皇后](https://leetcode.cn/problems/n-queens/)

**思路：**坐标规律：对于列的话比较好办，就用col集合来记录所有已经放置的列即可；同时又因为斜线上也不能再放置queen，所以还要存储斜线的信息，观察到，当在某个位置放置了queen后，那么对于queen所在的主对角线的那条斜线上，所有点的横坐标减去纵坐标的值是相等的，所以可以存储坐标值差在dia_main里面；而queen所在的次对角线方向上的点而言，可以发现他们的横纵坐标值相等，所以存储坐标之和在dia_sub里面，每次对于放置的位置只需考察是否不在上述三个集合里面就ok。

**语法补充**：集合``set()``中：增加是``add()``；移除是``remove()``

```python
class Solution:
    # 初始为全为'.'的列表
    solution = ['.'] * n
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 用于生成一种可能的布局对应的输出格式list
        def generateBoard(queen):
            board = list()
            for i in range(n):
                # queen[i]代表的是第i行的queen放置的col值,所以此行代码放置好了一层的queen
                solution[queen[i]] = 'Q'
                # 将防止好的一行加入board中
                board.append(''.join(solution))
                # 复原solution来处理下一行的queen放置的位置
                solution[queen[i]] = '.'
            return board
        
        def backtrack(row):
            # 每一行都放好了，生成符合输出格式要求的一个排列，加到ans中
            if row == n:
                board = generateBoard(queen)
                ans.append(board)
                return
            # i代表queen在该行的每一列进行放置
            for i in range(n):
                # i不在已经放置的列里；row-i代表不在不能放置的主对角线里；row+i代表不在不能放置的次对角线里
                if (i not in col) and (row-i not in dia_main) and (row+i not in dia_sub):
                    # queen此行的放置位置则选择i；此列不能放了，col中加入；主对角线和次对角线对应加入row-i和row+i的情况
                    queen[row] = i;col.add(i);dia_main.add(row-i);dia_sub.add(row+i)
                    # 递归处理下一行
                    backtrack(row+1)
                    # 回溯（和前面的完全反着来就OK，queen是因为原始值为-1）
                    queen[row] = -1;col.remove(i);dia_main.remove(row-i);dia_sub.remove(row+i)
        
        # queen[i]代表第i行queen放置在第几列，初始为-1就ok
        queen = [-1] * n
        # 所有的可能排列答案
        ans = []
        # 存储已经不能再放的主对角线规律值；用set()方便判断是否在其中
        dia_main = set()
        # 存储已经不能再放的次对角线规律值
        dia_sub = set()
        # 存储已经不能再放的列值
        col = set()
        backtrack(0)
        return ans
```

## 二分查找

### [35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)

**思路：**二分查找最基础版本。几个记忆点，**while循环条件带等号，返回的是left**。

具体的理解这个表格很有用，其实主要就是left和right的变化原理决定的。

| **移动情况**             | **目的**                     | **left 和 right 的变化** | **循环结束时 left 最终指向...**                              |
| ------------------------ | ---------------------------- | ------------------------ | ------------------------------------------------------------ |
| **`nums[mid] < target`** | 中间值太小了，目标值在右侧。 | `left = mid + 1`         | `left` 向右移动，它始终指向**第一个大于或等于 `target` 的元素的位置**。 |
| **`nums[mid] > target`** | 中间值太大了，目标值在左侧。 | `right = mid - 1`        | `right` 向左移动，它始终指向**最后一个小于 `target` 的元素的位置**。 |

````python
class Solution(object):
    def searchInsert(self, nums, target):
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right)//2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                return mid
        return left
````

### [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)

**思路：**两次二分查找，先找到要插入的行，因为基本的二分查找是返回left，left对应的是第一个大于等于target的元素位置，所以第一次二分查找要对矩阵的最后一列而不是第一列进行查找。找到对应的行号之后，对该行再次进行一次二分查找找到插入的列，将该元素与target对比，相等则返回Ture，否则返回False即可。

需要注意：如果给的元素比矩阵的任何元素都要大，那么第一次二分查找可能导致left指向了矩阵外的第一行，所以要加一个对left的越界判断条件，越界了可以直接返回False了。如果没有越界，对行进行二分时，不会有越界风险，无需额外处理。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 基础二分法实现
        def binsearch(nums, target):
            left = 0
            right = len(nums)-1
            while left <= right:
                mid = (left + right)//2
                if nums[mid] > target:
                    right = mid - 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    return mid
            return left
        
        nr, nc = len(matrix), len(matrix[0])
        # 将矩阵的最后一列放到nums中
        nums = [matrix[i][nc-1] for i in range(nr)]
        row = binsearch(nums, target)
        # 越界判断
        if row > nr-1:
            return False
        # 对行进行二分查找
        col = binsearch(matrix[row], target)
        if matrix[row][col] == target:
            return True
        else:
            return False
```

### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

**思路：**两次偏向性二分查找，第一次找左边界，所以进行一个左偏性的二分查找，每次找到target时，不直接结束查找，而是把right指针的值先赋给first后，将right向左移动一个位置，继续进行查找，直至left大于right才结束。第二次查找右边界时，则对应进行又偏性二分查找，最终返回最右的边界end。

````python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        first, end = -1, -1
        # 第一次确定左边界first，进行左偏性二分查找
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                # 先将mid暂时给first
                first = mid
                # 不结束二分查找，而是把right向左移动一位，尝试寻找更左边的first
                right = mid - 1
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
        # 第二次二分查找来确定end，进行右偏性二分查找
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                # mid暂时先给end
                end = mid
                # 不结束二分查找，将left向右移一位，尝试查找更右边的边界
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
        return [first,end]
````

### [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

**思路：**每次对于mid而言，必定有一边是完全有序的，我们需要找到完全有序的那一边，因为只有完全有序我们才能通过比较target和区间的两个端点的值的大小来判断是否真的在这个区间内，如果真的在这个区间内，就会很好办。所以问题的核心在于找到真的有序的那一半，我们只需要通过``if nums[mid] >= nums[left]:``就可以完成判断！！如果成立，那么mid的左边就是完全有序的，反之则是mid的右边有序。

``if nums[mid] >= nums[left]``此处需要注意一定是大于等于，如果没有等于，要求mid严格大于的话，对于左边这个区间只有1个元素的情况下，就会导致错误判断右边才是有序的！！

````python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (left+right) // 2
            if nums[mid] == target:
                return mid
            # mid左边的数组有序，在此处进行left和right的更新
            # 注意一定是>=不能只考虑大于的情况，因为一个元素的情况下，左边是有序的
            if nums[mid] >= nums[left]:
                # 左边有序，则可以通过端点值和target的比较，来判断target在mid的左边还是右边左
                if nums[left] <= target < nums[mid]:
                    right = mid-1
                # 不在左边的有序区间，那么就在右边
                else:
                    left = mid+1
            # mid右边的数组有序，则在此处进行left和right的更新
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid+1
                else:
                    right = mid-1
        return -1
````

### [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

**思路：**最小值位于旋转点处，而旋转点在二分查找过程中，除了刚好就是mid的位置上，其他情况，一定是位于无序的那边，所以需要首先判断mid是否为旋转点，再去寻找有序的一边，不断去对无序的进行二分查找操作。同时，在开始二分查找之前，while循环外部可以先判断数组是否真的进行了旋转操作。

以及最后可以证明，left所指向的位置会逐渐收敛于最小值也就是旋转点的位置，right则会指向其前一个位置。最简单的（1，0）可以很快的得出此结论。

````python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (left+right) // 2
            # nums[mid] < nums[mid-1]那就说明mid位置正好是旋转点，返回
            # mid要大于0，一是保证mid-1合法，二是mid为0不可能是旋转点，否则就是没有进行旋转
            # 用于判断mid是不是旋转点
            if mid > 0 and nums[mid] < nums[mid-1]:
                return nums[mid]
            # mid比0处数字大，则左侧有序，旋转点位于mid右侧
            # nums[0]这个地方不能是nums[left]!!
            # left一直在变动，如果变到一个较小的值，比nums[left]大根本就不能确定左侧的有序性！
            # 固定和nums[0]进行比较才是正确的做法！
            if nums[mid] >= nums[0]:
                left = mid+1
            # 否则，旋转点位于mid左侧
            else:
                right = mid-1
        # 如果发生了旋转，在while循环里面会最后找到旋转点的，如果在while里没找到，就是没有发生旋转！
        # return第一个元素就ok
        return nums[0]
````

五刷代码太漂亮了：

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        if nums[n-1] >= nums[0]:
            return nums[0]
        left, right = 0, n-1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] >= nums[0]:
                left = mid + 1
            else:
                right = mid - 1
        return nums[left]
```

### [4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

**思路：**假设我们的中位数是所有的数中第k小的数，那么我们考虑每次去排除k//2个数，分别找到两个数组的第k//2个数，比较大小，较小者所在的数组前面那些数肯定不是第k小的数，全部排除，假设这部分数有count个，那么接下来递归寻找第k-count小的数即可。更详细地看代码注释。

对于长度为n和m的两个数组的中位数，对于奇数则是所有数中第(n+m+1) // 2小的数，对于偶数则是第(n+m) // 2小的数和第(n+m) // 2+1小的数的算术平均。

为了统一处理，合并为处理第（n+m+1）//2小和（n+m+2）//2小的数的平均数。

````python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n = len(nums1)
        m = len(nums2)
        # first和second求的是要找第几小的数
        # 如nums1长度为4，nums2长度为5，那么first = 5，second = 5，即找到所有数里面第5小的数求和再求平均就是所要求的中位数
        first = (n+m+1) // 2
        second = (n+m+2) // 2
        # start1,end1代表在数组nums1中的始末坐标范围；k为寻找第k小的数
        def get_k_min(n1,start1,end1,n2,start2,end2,k):
            # 目前处理的两个数组的长度
            len1 = end1 - start1 + 1
            len2 = end2 - start2 + 1 
            # 短的数组放到前面，便于处理数组变空的情况
            if len1 > len2:
                return get_k_min(n2,start2,end2,n1,start1,end1,k)
            # 一个数组变空了，那么直接在第二个数组中找到目前第k小的数返回就行
            if len1 == 0:
                return n2[start2+k-1]
            # k到了1，只需找到第一小的数，且此时两个数组都不为空，只需返回两个数组的第一个数的较小者就行
            if k == 1:
                return min(n1[start1],n2[start2])
            # 分别找到两个数组中第k//2个数的索引
            # 索引和最后一个元素的索引取min避免越界
            index1 = min(start1+k//2-1,end1)
            index2 = min(start2+k//2-1,end2)
            # 谁小，那么该数字所在数组的前面这些数都不可能是第k小的数
            # 更新参数就行递归
            if n1[index1] <= n2[index2]:
                # return get_k_min(n1,index1+1,end1,n2,start2,end2,k-k//2)
                # k的更新由于前面index的地方使用了min来处理，所以真正去掉的数可能不是k//2个
                return get_k_min(n1,index1+1,end1,n2,start2,end2,k-(index1-start1+1))
            else:
                return get_k_min(n1,start1,end1,n2,index2+1,end2,k-(index2-start2+1))
        # 对第first和second小的数求平均即可    
        return (get_k_min(nums1,0,n-1,nums2,0,m-1,first) + get_k_min(nums1,0,n-1,nums2,0,m-1,second)) * 0.5
````

## 栈

### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)

**思路：**题目简单，但是修改了很多次小错误才正确。思路就是左括号入栈，右括号匹配，匹配不上返回False，一路匹配完之后，栈不是空的返回False否则返回True。

注意的点：

1、不要上来就匹配，要先判断栈是否为空，要是栈空了直接返回False了

 2、不要匹配完了就``returnTrue``了，可能是匹配结束，但栈里面还有东西呢，说明是有问题的啊，所以是``return not stack``

````python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        thsis = {'(':')','[':']','{':'}'}
        for char in s:
            if char in thsis:
                stack.append(char)
            else:
                if not stack:
                    return False
                tmp = stack.pop()
                if thsis[tmp] != char:
                    return False
        return not stack
````

### [155. 最小栈](https://leetcode.cn/problems/min-stack/)

**思路：**很天才的设计思想。一个stack作为主栈来进行pop，push，top这些，关键在于辅助栈minstack的操作，minstack初始存一个inf在里面，然后每次主栈push的时候，将当前push进来的值和minstack的栈顶值进行比较，将两者中间较小的值进行比较，那么此时其实每次minstack的栈顶存储的都是目前主栈里面最小的那个数，每次主栈pop的时候，minstack也一起pop就行，反正minstack栈顶永远保持是目前主栈中所有元素的最小值。

初始化为inf就是为了能将第一个元素和后续的元素的处理一起合并了，都是和栈顶元素比较大小。

```python
class MinStack:
    # 题目中已经说了，pop，top，getmin操作都是对非空栈操作

    # 借助一个辅助栈来存储min值，和主栈一起pop，push
    def __init__(self):
        self.stack = []
        # 初始化inf
        self.minstack = [float('inf')]

    def push(self, val: int) -> None:
        self.stack.append(val)
        # minstack存储最小值（栈顶元素）和这次插入的val的大小，存储新的min值
        self.minstack.append(min(self.minstack[-1], val))

    def pop(self) -> None:
        # 主栈和辅助栈一起pop
        self.stack.pop(-1)
        self.minstack.pop(-1)
        

    def top(self) -> int:
        # 别手快写成stack(-1)
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minstack[-1]
```

### [394. 字符串解码](https://leetcode.cn/problems/decode-string/)

**思路：**使用一个栈就能完成此题，其实最麻烦的就是字符串这个地方出栈又入栈的，什么时候要取逆序。

分几个遇到的字符情况解释：

1.``'['/'字母'``：遇到左括号或者字母，都直接入栈

2.``'数字'``：其实也是直接入栈，只不过有可能是两位数或者甚至三位数，所以要获取到完整的数字再进行入栈操作

3.``']'``： 把栈中所有字符串依次出栈存于tmp_str中，对其进行整体取逆序，也就是说对于其中的字符串是不会每个都翻转的，只是整体顺序取反，然后将其进行join成一个完整字符串后就可以进行倍数计算了。所得的字符串再入栈即可。

![Q394](E:\笔记\Q394.jpg)

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        i = 0
        while i < len(s):
            # 左括号直接入栈
            # 字母也直接入栈
            if s[i] == '[' or s[i].isalpha():
                stack.append(s[i])
                i += 1
            
            # 数字的话获取完整的数字再入栈
            elif s[i].isdigit():
                tmp_num = 0
                # 注意一下记得在while中加上i < len(s)的条件就ok
                while s[i].isdigit() and i < len(s):
                    tmp_num = tmp_num * 10 + int(s[i])
                    i += 1
                stack.append(tmp_num)
            elif s[i] == ']':
                tmp_str = []
                # 最麻烦的就是要看看那个地方字符的顺序
                # 把栈中到[之前的所有字符或者字符串全部pop出来
                while stack[-1] != '[':
                    tmp_str.append(stack.pop())
                stack.pop()
                # 此时栈顶一定是数字times
                times = stack.pop()
                # 此时对于pop出来的所有字符整体逆序，再连接join
                tmp_str = tmp_str[::-1]
                # 乘以次数
                tmp_str = times * (''.join(tmp_str))
                # 结果入栈即可
                stack.append(tmp_str)
                i += 1
        # return的时候记得join一下，因为确实在处理过程中是分散的
        return ''.join(stack)
```

### [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

**思路：**单调栈。stack栈中存储天数的索引，这些天数索引所对应的当天的温度是一个单调减的规律，所以叫做单调栈。ans用于存储结果数组，初始化为全0。

对于每一天的温度，我们拿这个温度和栈顶，去比较，如果比栈顶高，就可以返回栈顶索引index（即是第几天）过了i-index（i是当前这天的索引）天后是第一个比自己温度高的日子，然后不断地去pop然后和新的栈顶比较，直到栈空，或者比栈顶温度低了，那就入栈。

如果一开始就比栈顶温度低，那就直接入栈就ok。整个的思路十分巧妙，stack中存储索引而不是值方便获取索引对应地值，也方便按照索引更新ans对应位置的数。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        ans = [0] * len(temperatures)
        stack = []
        # 循环每一天的温度
        for i in range(len(temperatures)):
            # 注意是比较温度，而stack中是索引
            while stack and temperatures[i] > temperatures[stack[-1]]:
                # 当前温度比stack栈顶高，那就能找到栈顶对应的答案了，pop出栈顶索引
                index = stack.pop()
                # 栈顶索引和ans中索引一一对应，ans更新为当前天减去栈顶天数
                ans[index] = i - index
            # 当前天的索引入栈
            stack.append(i)
        return ans
```

### [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

**思路：**最重要的点，这个题目等价于求解以各个height作为高度的最大柱形中的max。（但实际上我并不是很能明白为什么只要考虑这些情况就可以了，为什么不考虑别的高度下的最大柱形呢？emm背下来这个结论算了）

去求各个height作为高度的最大柱形，就需要分别找到在该柱形左右的高度小于他的第一个柱形，比如是left和right，那么就可以求得此高度下的最大柱形为$$height×（left-right-1）$$.问题就转化为如何去求这些东西了。

求解的核心数据结构依旧是单调栈结构。怎么去想到单调栈呢？求上一题的时候，寻找第一个温度比当前高的，而此题则是寻找第一个高度比当前低的，似乎有很大的相似，所以会考虑去使用单调栈来实现。

现在回到单调栈，这个题目和温度题相反，是利用单调增高的单调栈，那么每次遇到比栈顶矮的height就到了一个pop出栈顶的height，求解栈顶的height所对应的最大柱形的时候了。可是，对于柱形面积最大的情况，除了右边要比当前的height矮，左边也需要找到比height矮的地方，这怎么办？难道还要其他的数据结构？实际上并不需要，单调栈里已经隐式地满足了这个条件，因为都是单调增高的，所以height在栈中的前一个高度，一定是要比height矮的！（当然，这个地方其实是有可能和height高度一致的，那这个时候的返回岂不是会导致少计算了左边这块高度相同的柱形的面积？其实并不会，因为计算完当前的之后，只有当前的这个height会出栈，左边高度相同的那个height依旧在栈中，他会计算出一个以相同的height作为高的最大柱形的值，并且会是一个更大的值，覆盖掉了刚刚求的小值，所以不会出现错误）

此外，同样为了方便处理，stack里面存储的依旧是索引，而非真实的height。以及为了方便处理首末两个柱形，加入一左一右，两个height为0的哨兵来处理。

**语法补充：**列表拼接          $$\text{heights} = [0] + \text{heights} + [0]$$

​		`[0]`：创建一个只包含一个元素 `0` 的新列表。**`+` 运算符**：在 Python 中，`+` 运算符用于**列表的拼接（Concatenation）**，它会将右侧的列表内容连接到左侧列表的末尾，生成一个**新的列表**。

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 存储最大柱形面积值
        res = 0
        # 栈中存储单调的高度增加的height所对应的索引,并存一个0哨兵进去，减少对第一个柱子计算的判断
        stack = [0]
        # 设置左右两个哨兵节点
        heights = [0] + heights + [0]
        size = len(heights)
        # 求除去两个哨兵节点外的height所对应的最大柱形
        # 注意是range(1, size),一开始从1开始就行，因为左哨兵已经入栈了
        # 到size-1是必须的，要不然右哨兵完全没用上
        for i in range(1, size):
            while stack and heights[i] < heights[stack[-1]]:
                # 当前高度应该是栈顶的索引对应的高度
                # 而且求了之后要pop的，直接在这里pop也行
                cur_height = heights[stack.pop()]
                # 当前宽的话是右边小于height的索引-左边小于height的索引-1
                cur_width = i - stack[-1] - 1
                # 更新res
                res = max(res, cur_height * cur_width)
            # 当前索引入栈
            stack.append(i)
        return res
```

## 堆

### [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

**思路一：**基于快排的实现。通过随机选择pivot，并设置big和small列表，存储遍历数组中大于所选择的pivot的和小于pivot的数。对比要寻找的第k个最大的数，big中的数量如果大于等于k，那就在big中进行下一次快排。进一步判断，如果big加上等于pivot的数字的总数小于k，那就在small中进行快排，大于等于k，那说明这个数就是此次的pivot，返回即可。（方法很好理解，就是有点费时间和空间）

<看了下面那道题的快速排序写法，发现这个解法确实也是可以优化的，或者说这个版本就是把本身的快速排序简化了，本身就是可以不用big和small，而是进行交换，最后把pivot放到他该在的位置上去，并且这个位置的index是可以跟踪的，所有小于pivot和大于pivot的数的个数都是可以知道的。>更新：思路二就是解决了这个问题

 **语法补充：**`random.choice()`用于**从一个非空序列（如列表、元组或字符串）中随机选择一个元素**。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick_sort(nums,k):
            # 随机选择作为pivot
            pivot = random.choice(nums)
            big, small = [], []
            for num in nums:
                # 大的放入big
                if num > pivot:
                    big.append(num)
                # 小的放入small
                elif num < pivot:
                    small.append(num)
            # 要找的是从大到小排列的第k个，如果大数的个数大于等于k，那么就去找big中的k大
            if k <= len(big):
                return quick_sort(big,k)
            # len(nums)-len(small)这是大数和相等数的个数，要是k比这还大那就是在small这个数组里了
            elif k > len(nums)-len(small):
                # return quick_sort(small,k)
                # 这里的不再是k了，前面已经排除掉了len(nums)-len(small)个大的，所以是用k减去这些的个数
                return quick_sort(small, k-(len(nums)-len(small)))
            # 不在big和small里，那就是相等
            return pivot
        return quick_sort(nums, k)
```

**思路二：**还是快排来实现，就是最原始的那种快排，没有那么好理解，但是时间复杂度降了，而且确实很很重要。

原来快速排序分为两种啊，最开始接触的那种是可以叫做挖坑法，每次要填坑，学名叫做Lomuto分区；而现在这种直接小于pivot和大于pivot的直接交换的也是快速排序的一种，效率更高，学名叫做Hoare分区。Lomuto分区每次结束都会至少把一个元素放到有序序列的最终位置上，但Hoare分区只能保证某一侧的所有元素小于pivot，而另一侧的元素都大于pivot。

**语法补充：**两种快速排序/快速选择算法；

**Lomuto分区：**

- 保证pivot在最终正确位置
- 返回pivot的索引
- 分区：`[l, p-1]` < pivot, `[p]` = pivot, `[p+1, r]` > pivot

**Hoare分区：**

- 不保证pivot在分界点
- 返回分界点j（就当是规定。）
- 分区：`[l, j]` ≤ pivot, `[j+1, r]` ≥ pivot

````python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # nums是查找的数组，left和right是查找的左右边界索引，
        # k是表示找的是第几小的数，需要注意是从第0小的数开始计算的
        def quickselect(nums, left, right, k):
            pivot = nums[left]
            i, j = left - 1, right + 1
            # 递归终止条件
            if left == right:
                return nums[left]
            # 和二分查找不一样，这里是小于没有等于
            # 如果等于还进行交换有可能会陷入死循环的哈
            while i < j:
                i += 1
                while nums[i] < pivot:
                    i += 1
                j -= 1
                while nums[j] > pivot:
                    j -= 1
                if i < j:
                    nums[i], nums[j] = nums[j], nums[i]
            if j >= k:
                return quickselect(nums, left, j, k) 
            else:
                return quickselect(nums, j+1, right, k) 
        # 比如有1，2，3，4，5五个数，要找第2个最大的数，那就是4，5-2=3，转为找第三小的数，再结合索引从0开始就ok了
        return quickselect(nums, 0, len(nums)-1, len(nums)-k)   
````

**思路三：**利用堆来实现，更不好理解了，但是堆排序还真的挺重要的。那就是实现一个堆排序，然后不断把墩顶移除，一直到第k个大的到了堆顶就行。

其实堆排序真挺简单的！一个``HeapAdjust(arr, n, i)``，一个``BuildHeap(arr)``

````python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        ''' 堆调整算法；从当前节点开始向下调整，直到叶子节点结束,此处建立大根堆 '''
        # arr是建堆的数组;n是arr的长度;i是当前调整的节点的索引
        def HeapAdjust(arr, n, i):
            # 最大值的索引初始化为i
            max = i
            # 左右孩子索引（一定不要搞错了，要加1的）
            left = 2 * i + 1
            right = 2 * i + 2
            # 左孩大，则把最大值索引更新
            if left < n and arr[left] > arr[i]:
                max = left
            # 右孩大则把最大值索引再次更新
            if right < n and arr[right] > arr[max]:
                max = right
            # 如果i不是max，那么说明出现了更新，先把目前位置更新
            # 再继续向下传递到叶子节点
            if max != i:
                arr[max], arr[i] = arr[i], arr[max]
                HeapAdjust(arr, n, max)
        ''' 建堆算法，从最后一个非叶子节点开始向上，每个都不断调用向下调整的堆调整方法 '''
        def BuildHeap(arr):
            n = len(arr)
            for i in range(n//2-1,-1,-1):
                HeapAdjust(arr, n, i)
        
        # 建堆
        BuildHeap(nums)
        n = len(nums)
        # 不断将堆顶的最大的元素和最后元素交换位置，并向下调整
        # 执行k-1次，此时堆顶就是第k个最大的元素
        # n-1-（k-1）+1=n-k+1
        for i in range(n-1,n-k,-1):
            nums[i], nums[0] = nums[0], nums[i]
            HeapAdjust(nums, i, 0)
        return nums[0]
````

### [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

**思路：**很巧妙的思路。首先用个Counter把各个元素出现的次数统计一下，然后获取最大的出现次数，建立一个桶，桶里头全是空列表，个数等于最大的出现次数加1，然后把桶中的索引当作元素出现的次数，把次数等于该值的全部放到该索引处的“子桶”里。最后按照桶的逆序来不断获取出现频率高的元素。

**语法补充：**

1、``[[] for _ in range(max_cnt+1)]``和``[[] * (max_cnt+1)]``

- `[0] * n` ✅：创建n个独立的整数0
- `[[]] * n` ❌：创建n个引用指向同一个列表
- `[[] * n]` ❌：语法错误，实际上只创建一个桶
- `[[] for _ in range(n)]` ✅：创建n个独立的空列表

2、Counter（）用于list上时，会返回一个字典，key是list的元素，value是元素出现的次数

3、values():`` 是 Python 中 `Counter` 对象的一个方法，用于获取计数器中所有的计数值（频率值）。

4、对于列表使用+运算符来连接多个列表，而append()则是向列表中加入这个完整的元素。

- **`+` 操作符**：连接两个列表（列表与列表），将右操作数的**元素**添加到左操作数
- **`append()`**：向列表添加单个元素（可以是任何对象），将整个右操作数作为一个**单个元素**添加

5、reversed() 和.reverse()

|     特性     |  `reversed()`  |       `.reverse()`        |
| :----------: | :------------: | :-----------------------: |
|   **类型**   |    内置函数    |         列表方法          |
|  **返回值**  | 返回反转迭代器 | 返回 `None`，原地修改列表 |
|  **原列表**  |  不改变原列表  |        改变原列表         |
| **适用对象** | 任何可迭代对象 |          仅列表           |

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # cnt返回一个字典，key是nums中的元素，value是元素出现的次数
        cnt = Counter(nums)
        # 找到最大的出现次数
        max_cnt = max(cnt.values())
        # 建立一个桶，元素全为0，元素个数为最大的出现次数加1
        # 对应索引表示出现的次数，对应索引的列表存储的则是该出现次数对应的元素
        buckets = [[] for _ in range(max_cnt+1)]
        # 将所有元素按照出现次数times的索引加入该索引的列表中
        for num,times in cnt.items():
            buckets[times].append(num)
        ans = []
        # reversed按照出现频率从高到低来访问
        for bucket in reversed(buckets):
            ans += bucket
            # 这里限制了一定会有符合的答案，所以肯定会到达这个条件的
            if len(ans) == k:
                return ans 
```

### [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)

**思路：**咱就是说，为什么什么限制都没有的情况下，会想到用堆来完成.....虽然说确实用堆很高效，但是没有限制要多高的时间空间效率啊。

但是思路还是很牛逼的，借助两个堆来实现，一个大顶堆用来存储较小的那一半数字，一个小顶堆用于存储较大的那一半数字。

插入新元素按照两个堆的大小来分类，一直保持小顶堆是元素较多的那一个。谁相对少就插入到谁那边，同时因为插入的这个数不一定就是应该插到数量较小的这边的，所以先在此次不执行插入的那边执行一次插入，取操作后的堆顶来插入，原因如下：

- 如果这个元素本身就是要插到较小的这边的，那么他也会是出现在堆顶的，所以不变；

- 而如果不是要插入较小的这边的，经过这样的操作，就能在保证两个堆中元素的相对大小下，平衡俩个堆的元素数量。

**语法补充：**Python中的堆操作：（首先要注意一个点，就是python内置的堆都是小顶堆，所以实现大顶堆要对元素进行取反操作）

`heapq` 模块的设计思想是，它操作的是**普通的 Python 列表**，然后**在其内部维持**这个列表的堆特性（最小堆）。所以真正操作的对象并不是堆！只是普通列表，所以不需要在开始就进行heapify操作！

| 操作类型     | 函数/方法                    | 时间复杂度 | 描述                   | 示例                                 |
| :----------- | :--------------------------- | :--------- | :--------------------- | :----------------------------------- |
| **堆化操作** | `heapq.heapify(heap)`        | O(n)       | 将列表原地转换为堆     | `heapq.heapify([3,1,4])` → `[1,3,4]` |
| **添加元素** | `heapq.heappush(heap, item)` | O(log n)   | 向堆中添加元素         | `heapq.heappush(heap, 0)`            |
| **弹出元素** | `heapq.heappop(heap)`        | O(log n)   | 弹出并返回最小元素     | `min_val = heapq.heappop(heap)`      |
| **查看堆顶** | `heap[0]`                    | O(1)       | 查看最小元素（不弹出） | `min_val = heap[0]`                  |

```python
from heapq import *
class MedianFinder:
    def __init__(self):
        ''' 
        一定是小顶堆存大数，大顶堆存小数
        这样的话小顶堆的堆顶就是大数中最小的，大顶堆的堆顶就是小数中最大的，这样的话两个堆的顶可以相连！
        '''
        # 一个小顶堆，存储较大的那一半数字
        self.minheap = []
        # 一个大顶堆，存储较小的那一半数字
        self.maxheap = []
    
    '''插入时一定保证小顶堆的元素个数不会小于大顶堆 '''
    def addNum(self, num: int) -> None:
        # 两个堆元素个数相等，则向minheap小顶堆插入;但是这个元素不一定是真的属于较大的那一部分的
        # 所以先向maxheap大顶堆插入该元素的相反数，再把maxheap大顶堆的堆顶的相反数插入minheap小顶堆
        if len(self.minheap) == len(self.maxheap):
            heappush(self.maxheap, -num)
            heappush(self.minheap,-heappop(self.maxheap))
        # 同样，当minheap个数大于maxheap时，该向maxheap执行插入，
        # 插入的数依旧是在minheap中插入后，取minheap的堆顶的相反数插入maxheap
        else:
            heappush(self.minheap, num)
            heappush(self.maxheap, -heappop(self.minheap))
            
    def findMedian(self) -> float:
        # 两个堆的元素个数不相等，返回minheap小顶堆的堆顶（因为插入时永远保证minheap小顶堆的个数大于等于大顶堆）
        if len(self.maxheap) != len(self.minheap):
            return self.minheap[0]
        # 两个堆的元素个数相等，返回两个顶的平均数（记得取大顶堆的相反数）
        else:
            return (-self.maxheap[0]+self.minheap[0])*0.5
```

## 贪心算法

### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

**思路：**卧槽了，easy题啊，真的没思路，一看到解析又如醍醐灌顶.....

 就是不断地维护一个前面日子的最低价格就行，我在某个位置卖出的时候最大利润一定是在前面地最低价买入的。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 最小价初始化为第一天价格
        min_price = prices[0]
        # 最大利润初始化为0
        max_profit = 0
        for i in range(1, len(prices)):
            # 最小买入价格要在当前利润计算后才能更新
            # 先计算当前利润
            cur_profit = prices[i] - min_price
            # 更新当前最大利润
            max_profit = max(max_profit, cur_profit)
            # 最后更新最小买入价格
            min_price = min(min_price, prices[i])
        return max_profit
```

### [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

**思路：**不断地去更新能到达的最右边的边界索引，断开了那就只能最后返回False了，如果什么时候边界索引能够到达最后一个数字的地方，那就直接返回True。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        # 初始化能到达的最右边为0
        right = 0
        for i in range(n):
            # 当前位置在能到达的位置上，才进行更新，否则实际上是中间就断开了
            if i <= right:
                # 不断更新能到达的最右边界
                right = max(right, i + nums[i])
            # 能到数组最后的位置，返回True
            if right >= n-1:
                return True
        return False
```

三刷的时候的代码，我觉得比较好。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        further = 0
        for i in range(len(nums)):
            if further < i:
                return False
            further = max(further,i+nums[i])
        return True
```

### [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)

**思路：**对于当前节点，会有一个可以到达的最远右边界，在未到达该边界时，我们一直往右一步一步走，不进行跳跃，但会不断记录如果在该点进行跳跃，可以到达的最远边界；当我们终于走到了边界时，此时不得不进行一次跳跃了，实际上这次跳跃是从前面记录的能到的右边界最远的点进行的一次跳跃，只是看起来像是在边界进行的一次跳跃；此时便产生了新的右边界，重复之前的操作，不跳但记录能跳的最远地方，直到再次抵达边界，不得不进行跳跃。

同时，如果在中间执行的时候，tmp_right已经能到达数组的末尾了，那么就可以直接返回了！！这样子尤其方便处理掉``nums = [0]``这个情况，初始的时候tmp_right是0，完全不需要跳跃就可以直接返回！

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        # 跳跃次数
        ans = 0
        # 此次行走的右边界（或者说就是走到哪才开始下一次跳跃）
        tmp_right = 0
        # 走到边界时进行跳跃之后。新的右边界（也就是tmp_right左边的点能到的最右边界）
        next_right = 0
        for i in range(len(nums)):
            # 不断更新下次右边界
            next_right = max(next_right, i+nums[i])
            # 如果当前的最右边界，在更新之前就已经能到达nums的末尾了，在此时就可以直接返回了
            if tmp_right >= len(nums)-1:
                return ans
            # 到达边界，不得不执行跳跃；并同步更新新的边界
            if i == tmp_right:
                ans += 1
                tmp_right = next_right
            
        return ans
```

### [763. 划分字母区间](https://leetcode.cn/problems/partition-labels/)

**思路：**很像跳跃游戏Ⅱ的思路。首先遍历一遍字符串s，获取每个字母在字符串中出现的最后一个位置的索引，对应存于长度为26的列表中。

再次开始遍历，不断遍历并更新所遍历到的字符的最后一次出现的索引，直至，当前所遍历到的位置索引和最后出现的位置索引相等，那么说明到此为止可以进行一次划分！将划分点后一个位置作为新的起点继续进行划分即可。

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = [0] * 26
        # 存储每个字母出现的最后位置的索引
        for index,char in enumerate(s):
            last[ord(char)-ord('a')] = index
        ans = []
        # 划分区间的两端初始化为全0
        start, end = 0, 0
        for i in range(len(s)):
            # end不断更新为目前该区间内出现的字母对应的最后出现的索引
            end = max(last[ord(s[i])-ord('a')], end)
            # i和最后出现的位置相等，那么说明目前这个start到end里的所有字母不再会在后面出现了
            # 执行一次划分
            if end == i:
                ans.append(end-start+1)
                start = end + 1
        return ans
```

## 动态规划

### [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

**思路：** `` if n == 0 or n==1:``
                          ``return 1 ``
             `` return self.climbStairs(n-1) + self.climbStairs(n-2)``

这样写逻辑倒是没什么毛病，但是重复计算的太多了，必定超时。这里用到常用的滚动数组，每次计算时，只需要前两个值，所以三个变量，不停滚动记录即可！！

然后稍微注意一下初始化的值就行了。动态规划的规律还是比较明显：$$f(x)=f(x−1)+f(x−2) （f(x)代表的是到第x个台阶的方式）$$

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        # p代表到r的前两个阶梯的方法；q代表到r的前一个阶梯的方法；r代表到当前阶梯的方法
        # 他们的初始化要保证0个阶梯有1种走法，1个阶梯时有1种走法
        # 所以pq都暂存为0，而r则代表0个阶梯时的走法数
        p = 0
        q = 0
        r = 1
        # 动态规划中常用的东西：滚动数组
        for i in range(n):
            p = q
            q = r
            r = p + q
        return r
```

### [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

**思路：**其实比较easy就是细节问题。

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        # 初始化第一行为1
        ans = [[1]]
        # 后面只需要构造numRows-1行了，所以应该是range(1, numRows)
        for i in range(1, numRows):
            # 上一行的列表
            pre_row = ans[i-1]
            # 第i行列表，元素初始化为全1，这样可以不去更新首末两个元素
            # 第i行的i从0开始计数，所以第i行的元素个数是i+1
            cur_row = [1] * (i+1)
            # 第i行的首末两个元素不进行更新，所以从1开始，i-1结束，range(1,i)
            for j in range(1,i):
                # 此行的第二个元素，是上一行的第一个和第二个元素的和
                cur_row[j] = pre_row[j-1] + pre_row[j]
            # 加入结果列表之中
            ans.append(cur_row)
        return ans
```

三刷写的，我觉得更好哈哈

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        numrow = [[1]]
        for i in range(1,numRows):
            pre_row = [0] + numrow[i-1] + [0]
            cur_row = []
            for j in range(len(pre_row)-1):
                cur_row.append(pre_row[j]+pre_row[j+1])
            numrow.append(cur_row)
        return numrow
```

### [198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

**思路：**状态转移方程：$$dp[i]=max(dp[i−2]+nums[i],dp[i−1])$$
           只有一间房屋，则偷窃该房屋，如果只有两间房屋，选择其中金额较高的房屋进行偷窃，然后还是可以用那个滚动数组来实现。

（aaaaaaaaa，看解析才会写啊，为啥这个动态规划就是写不会呢？？）

**三刷**：这个地方我想用dp[i]来表示抢第i个房子的时候的收获，结果发现不行，最直观的理解就是最后一个房子就不一定要被抢啊。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 初始化类似爬楼梯
        k_1 = 0
        k_2 = 0
        k_3 = nums[0]
        for i in range(1,len(nums)):
            # 轮转数组
            k_1 = k_2
            k_2 = k_3
            # k_3按照转移方程更新，取抢前一个房间和不抢前一个房间但抢此房间的最大值
            k_3 = max(k_2, k_1+nums[i])
        return k_3
```

### [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)

**思路：**使用``dp``数组存储和索引相等的数字所需要的最小完全平方数的个数，初始全为0。``dp[i]``初始化为``i``，即全由``1``构成的情况，然后就开始不断更新，每次选取一个完全平方数，从1的平方开始递增，取原来``dp[i]``，``i``减去一个完全平方数后的``dp[i-j*j]``+1两者的最小值不断更新，直至所有小于等于i的完全平方数都考虑完了，``dp[i]``也就确定下来了。

```python
class Solution:
    def numSquares(self, n: int) -> int:
        # dp[i]的位置存储数字i所需的最小的完全平方数的个数（n+1个就是为了方便一一对应）
        dp = [0] * (n+1)
        # 一定是到range(n+1)要不然dp[n]都没求呢
        for i in range(n+1):
            j = 1
            # 最糟糕的情况就是全为1的和
            dp[i] = i
            # 从j=1开始不断进行遍历
            while i >= j*j:
                # 取dp[i]目前值和i这个数中选择一个j*j这个完全平方数后的值的较小者
                dp[i] = min(dp[i], dp[i-j*j]+1)
                j += 1
        return dp[n]
```

### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

**思路：**使用``dp``数组存储和索引相等的数字所需要的最少硬币数，初始全为0。``dp[i]``初始化为``inf``，并将dp[0]单独初始化为0，提供一个相当于是遍历起点的东西。利用嵌套循环，对于每个总值，选取所有可能的硬币值来更新，同时一定要用if先判断是否合法！！取原来``dp[i]``，和``i``减去某个硬币值后的``dp[i-coin]``+1两者的最小值不断更新。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount+1)
        # 一定要将dp[0]置为0，提供一个正确的起点（那些inf都是不正确的起点，只是为了在更新的时候取min比较方便）
        dp[0] = 0
        for i in range(amount+1):
            for coin in coins:
                # 要先判断是否合法，避免越界
                if i-coin >= 0:
                    dp[i] = min(dp[i], dp[i-coin]+1)
        # 如果有更新就返回dp数组对应值，否则返回-1
        return dp[amount] if dp[amount] != float('inf') else -1
```

### [139. 单词拆分](https://leetcode.cn/problems/word-break/)

**思路：** ``dp``数组：``dp[i]``代表长度为i的字符串能否划分成功，数组初始化为``False``，并提供一个正确的基准``dp[0]``为``True``。接下来，对于长度从1开始递增到s的长度的每个字符串，对其每个位置进行划分，判断前j个字符``dp[j]``和后面剩下的字符串是否均在set中，只要能找到一个合法划分，即可结束当前长度字符串的划分。

| **变量/索引** | **含义**                                                     | **范围**       |
| ------------- | ------------------------------------------------------------ | -------------- |
| **`n`**       | 字符串 `s` 的长度。                                          | -              |
| **`dp[i]`**   | 字符串 `s[0:i]` 是否可拆分。                                 | $i \in [0, n]$ |
| **外层 `i`**  | **当前要判断的子串的结束索引**（即子串的长度）。             | $1$ 到 $n$     |
| **内层 `j`**  | **左侧子串的结束索引**（即分割点，将 `s[0:i]` 分为 `s[0:j]` 和 `s[j:i]`）。 | $0$ 到 $i-1$   |
| **`dp[j]`**   | 对应左侧子串 `s[0:j]` 的结果。                               | -              |
| **`s[j:i]`**  | 对应右侧子串，从 $j$ 开始到 $i-1$ 结束。                     | -              |

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # 转为集合便于判断是否是wordDict中的字符串
        word_set = set(wordDict)
        # dp[i]代表长度为i的字符串是否可以进行划分；所以还是dp长度要比len(s)长1
        dp = [False] * (len(s)+1)
        # dp数组基准
        dp[0] = True
        # i代表长度，所以是1到len(s)
        for i in range(1, len(s)+1):
            # j是对长度为i的字符串进行划分的位置；从0到i-1
            for j in range(i):
                # 长度为j的字符串和从j的位置开始到字符串的末尾这段是否在wordSet中;
                # 对于目前的整个字符串是否在set的判断不是在j=i-1的时候判断的，而是j=0的时候，通过s[0:i]来判断的!!!!!
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    # 只要能找到划分就可以结束目前的划分了
                    break
        return dp[len(s)]
```

### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

**思路：** 感觉解决动态规划问题最重要的就是去构思这个最优子问题啊。

设置数组``dp``，``dp[i] ``以``nums[i]``结尾的最长递增子序列长度而不是前``i个``元素的LIS长度

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp[i]为以nums[i]作为结尾的最长递增子序列
        dp = [1] * len(nums)
        # 从dp[1]开始更新，一直到最后一个字符
        for i in range(1,len(nums)):
            # 对于nums[i]结尾的序列前面的每一个数字进行遍历
            for j in range(i):
                # 如果nums[i]比前面的值大，那么nums[i]就可以接到前面的递增序列上
                if nums[i] > nums[j]:
                    # 更新接上去之后以nums[i]结尾的最长递增子序列长度
                    dp[i] = max(dp[j]+1,dp[i])
        # return dp[len(nums)-1]
        # 一定要注意max值不一定是在最后位置取得！！！可能是在前面取得！！
        return max(dp)
```

### [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)

**思路：**子数组是必须是连续的，同时因为这个地方存在负数，所以之前的乘积是最小的负数，但是再乘一个负数，可能发生逆袭的故事，所以每次都要存储前面的乘积的最大值和最小值。

每次更新时从三个里面取max来更新max_pre.  一是从当前数字开始，二是前面的最大的乘上现在这个数， 三是可能最小的负数乘上现在这个负数就变成最大的了。

ans则不断更新最大值，最后进行返回。

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        # 全部初始化为第一个数的值
        ans = max_pre = min_pre = nums[0]
        # 从第二个数开始遍历
        for i in range(1,len(nums)):
            # 一定要用临时变量存储，否则后面max_pre变了，导致min_pre的求解会出现问题
            tmp_max, tmp_min = max_pre, min_pre
            # 最大值可能是从当前数重新开始计算；可能是前面的最大值乘上这个数；还有可能是前面最小的一个负数乘上这个负数逆袭
            max_pre = max(nums[i],tmp_max*nums[i],tmp_min*nums[i])
            min_pre = min(nums[i],tmp_max*nums[i],tmp_min*nums[i])
            # ans不断更新取max_pre
            ans = max(max_pre, ans)
        return ans
```

### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

**思路：**dp数组在此题中选用二维数组，``dp[i][j]``代表下标索引从0到i的数是否存在和为j的组合。首先进行初始化，第一行和第一列单独初始化一下，以作为dp的起始基准，其他的全部设为False。

dp数组的更新，需要分为两种情况，比如``dp[i][j]``的更新，分为是否选取nums[i]，不选的话很简单，直接取``dp[i-1][j]``就ok，如果要取的话，则是``dp[i-1][j - nums[i]]``和``dp[i-1][j]``做or操作，而且要注意一下判断j和nums[i]的大小，避免越界错误的发生.

**补充：**发现一个判断循环边界的方法，以i为例，起始为1因为后面有``i-1``，要避免越界错误；终点为n，因为最后要返回的是``dp[n-1][target]``所以要一直求到n-1才行

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        # 1.长度小于2，直接返回
        if n < 2:
            return False
        total = sum(nums)
        # 2.和为奇数，直接返回
        if total % 2 == 1:
            return False
        target = total // 2
        # 3.最大值比target大，直接返回
        if max(nums) > target:
            return False
        # dp[i][j]代表下标索引从0到i的数是否存在和为j的组合
        dp = [[False] * (target+1) for _ in range(n)]
        ''' 
        dp数组基准确定：
            j=0时，dp[i][0]都是True，因为直接什么都不取就行
            i=0时，是相当于取nums[0]，所以dp[0][nums[0]]都是True
        '''
        for i in range(n):
            dp[i][0] = True
        dp[0][nums[0]] = True
        # i为了i-1合法，从1开始，直到n-1即可
        for i in range(1,n):
            # j要取到target才行
            for j in range(target+1):
                # 两种情况：取当前nums[i]，则寻找前i-1下标有没有和为j-nums[i]的；
                #如果不取当前数，那就是直接看dp[i-1][j]是啥就行（两者只要有一个为True就ok）
                if j >= nums[i]:# 避免越界错误！！
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - nums[i]]
                else:
                    dp[i][j] = dp[i-1][j]
        # 最终返回所有下标是否存在和为target的组合
        return dp[n-1][target]
```

### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

**思路一：动态规划**设置dp数组，其中``dp[i]``代表以``s[i]``作为结束时的最长有效括号。因为有效括号一定是``)``结束，所以所有的左括号都赋值为0，而对于遇到右括号时，需要对其前面一个位置的括号进行分类，无非是左还是右的情况。详见下图：

<img src="C:\Users\Leoti\AppData\Roaming\Typora\typora-user-images\image-20251111113110852.png" alt="image-20251111113110852" style="zoom:67%;" />

**补充：**对``dp[i-dp[i-1]-2]``的理解，比如``(()) (())``前面四个括号是组成了一个有效括号组合的，但从第五个开始，第六个第七个都没办法把他们加到里面。直到第八个的右括号出现，使得从第五个开始到第八个形成了一个有效的括号组合，正好和前面四个是邻接的，所以可以进一步扩大，达到8个有效括号的组合。（纯直觉解释，严格证明真的不会）

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # dp[i]代表s[i]结尾的能构成的最长有效括号
        # 处理一下s为空串的情况
        if not s:
            return 0
         # (作为结尾，肯定是0
        dp = [0] * len(s)
        # 遇到）,需要考察其前一个符号
        for i in range(1,len(s)):
            if s[i] == ')':
                # 前一个是(,则判断i-2是否合法就可执行加2的操作
                if s[i-1] == '(' :
                    dp[i] = dp[i-2] + 2 if i >= 2 else 2
                # 前一个是）
                elif s[i-1] == ')':
                    # 判断是否越界
                    if i - dp[i-1] >= 1:
                        # 判断对应位置有没有配套的（
                        if s[i-dp[i-1]-1] == '(':
                            dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2] if i-dp[i-1]>=2 else dp[i-1]+2
        return max(dp)
```

**思路二：栈**利用栈来实现。保证栈底存储的是目前最后一个没有被匹配的右括号的下标（所以初始化的时候，在栈中需要先存放一个-1保持一致）。（其实严格的逻辑，我还是晕晕乎乎的啊。。）

如果遇到``‘(’``就直接入栈。

如果遇到``‘)’``时，弹出栈顶元素：

​	如果弹出后栈顶元素不为空，那么当前右括号的下标减去弹出来的栈顶元素即为<以当前右括号为结尾的最长有效括号的长度>。

​	如果栈空了，那么说明当前的右括号是不能被匹配的(因为弹出去的就是之前最后一个不能被匹配的右括号的下标)，把当前右括号的下标放入栈中作为<目前最后一个没有被匹配的右括号的下标>。

**补充：**对于这个方法的逻辑的直觉表示，这个里面不会出现一个以上的右括号的下标，如果栈里面有不能匹配的右括号的下标了，那么再来一个右括号就会把前面这个pop出去，所以里面更准确地说是栈底，只会出现一个右括号的索引。实际上呢，这个右括号在计算后续的长度的时候很有用，尤其是对于length = max(length, index-top+1)会出错的情况，如果是``(())``计算第一个右括号出现的时候，其实不会出错，但是如果遇到``()(())``这种情况下计算最后一个右括号的时候，本来应该是6的，但是 index-top+1就变成了4，漏掉了前面两个括号！！！对于这个最后一个无法匹配的右括号的直观理解，就是因为这个括号的出现，后面的再也和前面的合法括号对接不上了！所以存储这个，后头的减他就行。

````python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # 栈中先存入-1
        stack = [-1]
        length = 0
        if not s:
            return 0
        for index,char in enumerate(s):
            # 左括号无脑入栈
            if char == '(':
                stack.append(index)
            # 右括号分栈是否为空
            elif char == ')':
                # 弹出栈顶元素
                top = stack.pop()
                # 栈不空，那么栈顶这个是可以配对的，length更新为当前右括号index-此时的栈顶
                if stack:
                    # length = max(length, index-top+1)会出错
                    # 这里出错的原因和上面dp的做法中为什么要加上dp[i-dp[i-1]-2]是一个道理
                    length = max(length, index-stack[-1])
                # 栈空，这个右括号不能被匹配，压入栈中，作为最后一个不能被匹配的右括号索引
                else:
                    stack.append(index)
        return length
````

## 多维动态规划

**三刷关于dp数组是否要大一行的总结，如果递推公式需要引用 $i-1$ 或 $j-1$ 的状态，并且 $i=1$ 或 $j=1$ 的计算依赖于 $i=0$ 或 $j=0$ 的基准值（而非边界检查，比如考虑最长有效括号的时候，虽然有空串的情况，但是这个情况直接通过边界检查就ok，那么 DP 数组应该大 1。**

### [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

**思路：**二维动态规划，第一行第一列的路径数可以都初始化为1，然后不断更新。``dp[i][j]``代表走到索引位置（i，j）的路径条数。其更新规则就是，左边格子和上边格子的路径条数之和。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 全部初始化为1
        dp = [[1] * n for _ in range(m)]
        # 遍历更新，每一行依赖于上一行的数
        for i in range(1, m):
            for j in range(1,n):
                # 左边格子走过来和上边格子走过来的路径条数之和
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```

### [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

**思路：**二维动态规划，``dp[i][j]``表示到达位置``grid[i][j]``时所经过的最少的和（包括终点位置）。所以初始化第一行第一列为他们一直横着走或者一直向下走的路径和，然后就可以开始更新了，更新的原则是取上方和左方的格子的较小值加上自身这一步的值，每一个地方都是选择较小的，所以最后达到终点时，也能保证是最小的！

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        dp = [[0] * col for _ in range(row)]
        sum1, sum2 = 0, 0
        # 初始化第一行
        for i in range(col):
            sum1 += grid[0][i]
            dp[0][i] = sum1
        # 初始化第一列
        for i in range(row):
            sum2 += grid[i][0]
            dp[i][0] = sum2
        for i in range(1, row):
            for j in range(1, col):
                # 更新dp数组，取上和左的较小值，加上自身的值
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[row-1][col-1]
```

### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

**思路一：动态规划法**

边界条件：长度小于2的时候一定是回文串，直接返回；

初始化：最长回文串为s[0]，长度为1；

​		``dp[i][j]``表示s[i]到s[j]的字符串是不是回文串（包括s[i]、s[j]）,``dp[i][i]``初始化为True，其余全False

dp更新：外层循环限定字符串长度从2开始到len(s)，内层循环i为字符串起点，结合两数可以得出结尾索引，如果越界，直接break，否则进一步判断首尾是否相等，不等的话无需额外操作。如果相等，再细分为两个情况：长度小于等于三，一定是回文串；长度大于三看去头去尾的是否为回文。接下来，进一步判断，如果``dp[i][j]``确实为True，且长度比目前最长回文串的长度大，那就进行最长回文串和最长长度的更新。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        # 最长回文串长度
        max_string = s[0]
        # 长度小于2一定是回文串
        if n < 2:
            return s
        # dp[i][j]表示s[i]到s[j]的字符串是不是回文串（包括s[i]、s[j]）
        dp = [[False] * n for _ in range(n)]
        # dp[i][i]是单个字符的情况，都是回文串
        for i in range(n):
            dp[i][i] = True
        # Length代表回文串的长度
        for Length in range(2, n+1):
            # i代表起点索引
            for i in range(n):
                j = i + Length -1
                # j越界了，那么当前这个长度下后面的i作为起点也没有考虑的意义了
                if j > n-1:
                    break
                # 始末字符一样！需要细分
                if s[i] == s[j]:
                    # 长度小于等于三，一定是回文串
                    if Length <= 3:
                        dp[i][j] = True
                    else:
                        # 否则需要看中间去头去尾的子串是否为回文串
                        dp[i][j] = dp[i+1][j-1]
                    # 如果确实为回文串,且长度更新了；更新max_string和length
                    if dp[i][j]:
                        max_string = s[i:j+1]
        return max_string
```

**思路二：中心扩展法**

按顺序选择s中的字符作为中心，不断同时向左右拓展，直到左和右的字符不相等。需要注意分为中心字符是一个，以及中心字符是两个的情况。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # left,right是扩张中心的左右索引，如果是一个的话就相等了
        def center_expand(left, right, s):
            # 一定是带等号的，否则会遗漏了left在索引为0处的判断
            while left>=0 and right<=len(s)-1 and s[left]==s[right]:
                left -= 1
                right += 1
            return s[left+1:right]
        ans = ''
        for i in range(len(s)):
            # 单元素作为中心
            one_center = center_expand(i, i, s)
            # 两个元素作为拓展起点
            two_center = center_expand(i, i+1, s)
            # 取one_center和ans中较长者
            if len(one_center) > len(ans):
                ans = one_center
            # 取三者最长者
            if len(two_center) > len(ans):
                ans = two_center
        return ans 

```

### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

**思路：**老规矩，难点还是构造合理的dp数组。

此处构造二维dp数组，其中``dp[i][j]``代表text1的前i个字符``text1[i-1]``和text2的前j个字符``text2[j-1]``的最长公共子序列。如果``text1[i-1]``=``text2[j-1]``,那么``dp[i][j]``就等于``dp[i-1][j-1]+1``；如果不相等，那么``dp[i][j]``就需要取``dp[i-1][j]``和``dp[i][j-1]``中的较大者。

（关于取``dp[i-1][j]``和``dp[i][j-1]``中的较大者，实际上我又无法完全理解。。。）

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1, n2 = len(text1), len(text2)
        # dp[i][j]代表text1的前i个字符text1[i-1]和text2的前j个字符text2[j-1]的最长公共子序列长度
        dp = [[0] * (n2+1) for _ in range(n1+1)]
        for i in range(1, n1+1):
            for j in range(1, n2+1):
                if text1[i-1] == text2[j-1]:
                    # 取左斜上的加一
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # 不等则取类似左和上的较大者
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        # 最长的出现在选择两个text的全部的情况下，所以直接返回dp[n1][n2]就行
        return dp[n1][n2]
```

### [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)

**思路：**所有的操作有这些：A中删除；B中删除；A中插入；B中插入；A替换字符；B替换字符，但实际上有些是等价的！！比如删除某个的字符，相当于在另一个中插入。最终不同的应该为：A中插入；B中插入；A替换字符三种！！

设置二维dp数组，其中``dp[i][j]``代表A的前i个字母和B的前j个字母之间的最小编辑距离，初始化基准则有``dp[0][j]``都等于j，``dp[i][0]``都等于i。

对于``dp[i][j]``的更新，需要考虑三个基本操作对应的三种情况来编辑：

对``dp[i][j-1]``只需对B进行一次插入操作；

对``dp[i-1][j]``只需对A进行一次插入操作；

对``dp[i-1][j-1]``：在A[i]B[j]不相等的情况下只需对A执行一次替换操作；如果相等的话则无需任何操作；就可以得到``dp[i][j]``。

所以``dp[i][j]``应该分A[i]B[j]是否相等，来取三者的最小值来进行更新。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1, n2 = len(word1), len(word2)
        # 其中dp[i][j]代表A的前i个字母和B的前j个字母之间的最小编辑距离
        dp = [[0] * (n2+1) for _ in range(n1+1)]
        # i或者j为0的时候最小的编辑拒绝就是非空字符的长度
        for j in range(n2+1):
            dp[0][j] = j
        for i in range(n1+1):
            dp[i][0] = i
        for i in range(1, n1+1):
            for j in range(1,n2+1):
                # 分是否相等来更新dp[i][j]
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
        return dp[n1][n2]
```

## 技巧

### [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

**思路：**利用异或运算，只要出现多次，在题目中限制了都是出现两次，则会由于异或运算的性质成对匹配变为0，最后剩下的只会是那个只出现了一次的元素。

**语法补充：**

**异或运算**

##### 1. 归零性（Identity with Self）任何数和它本身进行异或运算，结果都为 0。

​                $$A \oplus A = 0$$            **用途：** 用于快速清零，或在查找数组中唯一元素时抵消成对出现的元素。

##### 2. 恒等性（Identity with Zero）任何数和 0 进行异或运算，结果仍是它本身。

​                $$A \oplus 0 = A$$

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # return reduce(lambda x,y : x ^ y,nums)
        x = 0
        for num in nums:
            x ^= num
        return x
```

### [169. 多数元素](https://leetcode.cn/problems/majority-element/)

**思路：**利用“一换一”的策略，首次的时候选择第一个元素num[0]作为多数派，如果再次遇到num[0]将其计数器加一，遇到不同元素，就执行一换一政策，将其计数器减一，直至其减到0，换新的元素作为多数派，直至最后，此时的多数派就是真正的多数派。

因为题目中已经限定了多数派是出现次数大于n//2的元素，所以如果是真的多数派，是可以对剩下的元素全部执行一换一操作的。故如果计数器来到了0，他就一定不是多数派。从全局的角度来看，不断一换一，最后剩下的也一定是多数元素。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 初始化多数元素为第一个元素
        more = nums[0]
        # 计数器为1
        count = 1
        for i in range(1,len(nums)):
            # 如果计数器到0了，当前数成为新的多数元素
            if count == 0:
                more = nums[i]
            # 遇到目前的多数元素，计数器加一
            if nums[i] == more:
                count += 1
            # 遇到非目前的多数元素，计数器减一
            else:
                count -= 1
        return more
```

### [75. 颜色分类](https://leetcode.cn/problems/sort-colors/)

**思路：**使用双指针来实现，用p0指向最后一个0元素的后一个索引；p1指向最后一个1元素的后一个索引。

遍历nums，如果当前位置是1，则和p1索引位置上的元素交换元素，并对p1进行加一操作。

如果当前位置是0，则需要注意。**此时如果p1和p0相等，**则说明目前还没有遇到1元素，那么将p0对应索引元素和当前元素交换位置，并对p1和p0都执行加一操作即可。**如果p1≠p0**，则说明此时p0所指向的位置实际上也是第一个元素1的位置，在执行交换后，元素1会被换到遍历到的0元素的那个位置上，所以此时需要同时对p1索引元素和被换了位置的元素1进行一次交换，最后再对p1，p0同时进行加一操作。

如下所示，加粗的是执行交换操作的元素，可以看到执行0的操作破坏了1元素，需要再对1元素执行一次交换

00**1**112**0**22 →  00**0**112**1**22 → 00011**12**22

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        # p0指向最后一个0元素的后一个索引；p1指向最后一个1元素的后一个索引
        p0, p1 = 0, 0
        n = len(nums)
        for i in range(n):
            # 元素1直接交换p1索引元素和当前元素1即可，并将p1后移一位
            if nums[i] == 1:
                nums[p1], nums[i] = nums[i], nums[p1]
                p1 += 1
            # 遍历到0
            elif nums[i] == 0:
                # 先将p0索引对应元素和当前元素交换
                nums[p0], nums[i] = nums[i], nums[p0]
                # 如果p0，p1不等，则需要进一步将p1索引对应元素和被p0移走的元素再次交换
                if p1 > p0:
                    nums[p1], nums[i] = nums[i], nums[p1]
                # p0，p1均右移一位
                p0 += 1
                p1 += 1
```

### [31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

**思路：**首先第一次扫描：从后往前进行扫描，直到找到nums[i-1] < nums[i]的地方，这个nums[i-1]就是需要交换的数字，比如下方的数字”5“

第二次扫描：依旧是从后往前扫描，找到第一个比nums[i-1]大的数字，比如下方的数字”6“，接下来的操作就是交换数字”5“和”6“，数则变为了12386754

最后一步操作：将nums[i-1]后的数字，按照从小到大的顺序进行排列，从前面的操作来看，nums[i-1]后面的数字一定是逆序排列的，所以只需将其逆转即可，最终得到12386457。

​										1 2 3 8 5 7 6 4

依旧直觉感觉正确，但无法严格证明正确。

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        n = len(nums)
        i = n-1
        # 第一次扫描找到nums[i-1]，注意等号，这个里面可能会出现相等的数字的情况
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
        # i如果等于0，那么整个数字排列就是数字从大到小排列的，跳过第二次扫描，直接逆转就ok
        if i != 0:
            j = n-1
            # 第二次扫描，寻找从后往前第一个比nums[i-1]大的数，进行交换
            while j > i-1 and nums[j] <= nums[i-1]:
                j -= 1
            nums[i-1], nums[j] = nums[j], nums[i-1]
        # nums[i-1](不包括)后面所有的数进行逆转
        left, right = i, n-1
        while left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

### [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

**思路：**这个思路就不是能想出来的，也不去理解了。

建立索引和索引处存放的数字的映射（n和nums[n]的映射），将其再类比于指针，比如nums[3] = 6那么就相当于index是一个指针，指向index=6的地方，以此类比于链表，在有重复数字的时候，这个类似的链表一定是有环的，并且这个重复的数字一定是环的入口。故此题可以转化为寻找有环链表的环入口，只需把指针表示转换为求值即可。

比如slow = slow.next 转化为slow =nums[i] ；fast = fast.next.next转化为fast = nums[nums[i]]

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        # 初始的时候先移动一步，避免while不执行的错误
        slow = nums[slow]
        fast = nums[nums[fast]]
        # 第一次判等操作，相等时直接将fast移到链表头部（原因见142环形链表题目）
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
        fast = 0
        # 第二次循环，fast和slow相遇的点就是环的入口，也就是重复的那个数字
        while fast != slow:
            slow = nums[slow]
            fast = nums[fast]
        return fast
```

