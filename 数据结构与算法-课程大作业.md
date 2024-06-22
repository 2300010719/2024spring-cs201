# Assignment #P: 课程大作业

Updated 1009 GMT+8 Feb 28, 2024

2024 spring, Complied by ==董义希，数学科学学院==

# 基本知识点

## 一、逻辑结构

数据的逻辑结构是指数据元素之间的逻辑关系和组织方式，它独立于数据的具体存储方式。数据的逻辑结构可以分为以下几种主要类型：

1. **集合结构 **
   - **定义**：集合结构是最简单的逻辑结构，数据元素之间没有特定的顺序和关系，只是简单地归为一组。
   - **特点**：数据元素无序且互不关联。
2. **线性结构**
   - **定义**：有且仅有一个开始节点和一个终端节点，并且除了头尾，每个数据元素都有一个直接前驱和直接后继，是1对1的关系
   - **特点**：数据元素之间存在一对一的顺序关系
   - **常见的线性结构**有：线性表、栈、队列、串（注意：这四种都是逻辑结构）
3. **非线性结构**
   - **定义**：一个节点可能有多个直接前驱和后继
   - **特点**：数据元素之间存在一对多的层次关系。
   - **常见的非线性结构有**：树、图



## 二、存储结构

存储结构是指数据在计算机存储设备（如内存、磁盘等）中的实际组织和排列方式。

值得一提的是：存储密度$= ($结点数据本身所占的存储量$)/($结点结构所占的存储总量$)$，显而易见，链表储存密度＜1

根据数据元素在存储设备中的排列方式，存储结构主要可以分为以下几种类型：

1. **顺序存储结构 **

   - 在顺序存储结构中，数据元素按线性顺序存储在连续的内存空间中。这种结构的特点是可以通过元素的序号（索引）直接访问元素，访问效率高。
   - **优点**：
     - 访问效率高，可以通过索引直接访问元素，时间复杂度为 \(O(1)\)。
     - 内存空间紧凑，不存在指针存储开销。
   - **缺点**：
     - 插入和删除操作效率较低，尤其是在中间位置进行插入和删除时，可能需要移动大量元素，时间复杂度高。
     - 需要预先分配连续的内存空间，可能导致内存浪费或分配失败。
   - **例子**：
     - 顺序表
     - 基于数组实现的队列 (Queue) 和栈 (Stack)

2. **链式存储结构 **

   - 在链式存储结构中，数据元素存储在不连续的内存空间中，每个元素通过指针或引用链接在一起。链式存储结构的插入和删除操作是通过修改指针进行的。
   - **优点**：
     - 插入和删除操作效率高，时间复杂度为 \(O(1)\)，只需修改指针指向。
     - 不需要预先分配大块连续的内存空间。
   - **缺点**：
     - 访问效率低，必须通过指针逐个遍历元素，时间复杂度为 \(O(n)\)。
     - 需要额外的存储空间存储指针，增加了存储开销。
   - **例子**：
     - 单链表 (Singly Linked List)
     - 双链表 (Doubly Linked List)
     - 循环链表 (Circular Linked List)

3. **索引存储结构**

   - 在索引存储结构中，为了加快查找速度，在数据元素之外建立索引表，通过索引表来快速定位数据元素。索引存储结构常用于大规模数据的查找操作。
   - **优点**：
     - 查找效率高，可以快速定位数据元素。
   - **缺点**：
     - 需要额外的存储空间存储索引表。
     - 插入和删除操作较复杂，需要维护索引表。
   - **例子**：
     - 数据库索引 (B树、B+树)
     - 倒排表

4. **散列存储结构 **

   - 在散列存储结构中，通过散列函数将数据元素的关键字映射到存储地址，实现快速查找和存储。

   - **优点**：

     - 查找、插入和删除操作效率高，时间复杂度为 \(O(1)\)（平均情况下）。

   - **缺点**：

     - 需要设计良好的散列函数以减少冲突。
     - 处理冲突的方法（如开放地址法、链地址法）可能影响性能。

   - **例子**：

     - 哈希表 (Hash Table)

   - 歪楼：**处理冲突**

   - 开放地址法：线性探测法、二次探测法、随机探测法

   - **链地址法**：将所有关键字为同义词的记录存储在一个单链表中，在散列表中只存储所有同义词子表的头指针

     - 拉链法处理冲突简单，且无堆积现象，即非同义词决不会发生冲突，因此**平均查找长度较短**

     - 由于拉链法中各链表上的结点空间是动态申请的，故它更**适合于造表前无法确定表长**的情况

     - 开放定址法为减少冲突，要求装填因子α较小，故当结点规模较大时会浪费很多空间。而拉链法中可取α≥1，且结点较大时，拉链法中增加的指针域可忽略不计，因此节省空间

     - 在用拉链法构造的散列表中，**删除结点的操作易于实现**。只要简单地删去链表上相应的结点即可。而对开放地址法构造的散列表，删除结点不能简单地将被删结点的空间置为空，否则将截断在它之后填人散列表的同义词结点的查找路径。这是因为各种开放地址法中，空地址单元（即开放地址）都是查找失败的条件。因此在用开放地址法处理冲突的散列表上执行删除操作，只能在被删结点上做删除标记，而不能真正删除结点

     - 拉链法的**缺点**：指针需要额外的空间，故当结点规模较小时，开放定址法较为节省空间，而若将节省的指针空间用来扩大散列表的规模，可使装填因子变小，这又减少了开放定址法中的冲突，从而提高平均查找速度

       

## 三、链表

### 单链表

在链式结构中，除了要存储数据元素的信息外，还要存储它的后继元素的存储地址。

​	因此，为了表示**每个数据元素$$a_{i}$$与其直接后继元素$$a_{i+1}$$之间的逻辑关系，对数据$$a_{i}$$来说，除了存储其本身的信息之外，还需要存储一个指示其直接后继的信息（即直接后继的存储位置）。我们把存储数据元素信息的域称为数据域，把存储直接后继位置的域称为指针域。指针域中存储的信息称做指针或链。这两部分信息组成数据元素$$a_{i}$$的存储映像，称为结点（$$Node$$​）。**

​	我们把链表中第一个结点的存储位置叫做头指针。有时为了方便对对链表进行操作，会在单链表的第一个结点前附设一个节点，称为头结点，此时头指针指向的结点就是头结点。

![在这里插入图片描述](/Users/dyx/Desktop/20210207165354972.png)

​	空链表，头结点的直接后继为空。![在这里插入图片描述](/Users/dyx/Desktop/20210207165435359.png)



### 双链表

​	**双向链表$$(Double$$ $$Linked$$ $$List)$$是在单链表的每个结点中，再设置一个指向其前驱结点的指针域。**所以在双向链表中的结点都有两个指针域，一个指向直接后继，另一个指向直接前驱。



### 循环链表

​	**将单链表中终端节点的指针端由空指针改为指向头结点，就使整个单链表形成一个环，这种头尾相接的单链表称为单循环链表，简称循环链表。**

​	然而这样会导致访问最后一个结点时需要$$O(n)$$的时间，所以我们可以写出**仅设尾指针的循环链表**。



# 树

## 一、定义

#### 1、节点和边

​	**树**由节点及连接节点的边构成。树有以下属性：

​		❏ 有一个根节点；
​		❏ 除根节点外，其他每个节点都与其唯一的父节点相连；
​		❏ 从根节点到其他每个节点都有且仅有一条路径；
​		❏ 如果每个节点最多有两个子节点，我们就称这样的树为二叉树。

#### 2、递归

​	一棵树要么为空，要么由一个根节点和零棵或多棵子树构成，子树本身也是一棵树。每棵子树的根节点通过一条边连到父树的根节点。



## 二、题目

### 06646: 二叉树的深度

```
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None

def tree_depth(node):
    if node is None:
        return 0
    left_depth = tree_depth(node.left)
    right_depth = tree_depth(node.right)
    return max(left_depth, right_depth) + 1

n = int(input())  # 读取节点数量
nodes = [TreeNode() for _ in range(n)]

for i in range(n):
    left_index, right_index = map(int, input().split())
    if left_index != -1:
        nodes[i].left = nodes[left_index-1]
    if right_index != -1:
        nodes[i].right = nodes[right_index-1]

root = nodes[0]
depth = tree_depth(root)
print(depth)
```

### 24729: 括号嵌套树

包括定义preorder，postorder

```
class TreeNode:
    def __init__(self, value): #类似字典
        self.value = value
        self.children = []

def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点


def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

# 主程序
def main():
    s = input().strip()
    s = ''.join(s.split())  # 去掉所有空白字符
    root = parse_tree(s)  # 解析整棵树
    if root:
        print(preorder(root))  # 输出前序遍历序列
        print(postorder(root))  # 输出后序遍历序列
    else:
        print("input tree string error!")

if __name__ == "__main__":
    main()
```

### 24750: 根据二叉树中后序序列建树

```
"""
后序遍历的最后一个元素是树的根节点。然后，在中序遍历序列中，根节点将左右子树分开。
可以通过这种方法找到左右子树的中序遍历序列。然后，使用递归地处理左右子树来构建整个树。
"""

def build_tree(inorder, postorder):
    if not inorder or not postorder:
        return []

    root_val = postorder[-1]
    root_index = inorder.index(root_val)

    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]

    left_postorder = postorder[:len(left_inorder)]
    right_postorder = postorder[len(left_inorder):-1]

    root = [root_val]
    root.extend(build_tree(left_inorder, left_postorder))
    root.extend(build_tree(right_inorder, right_postorder))

    return root


def main():
    inorder = input().strip()
    postorder = input().strip()
    preorder = build_tree(inorder, postorder)
    print(''.join(preorder))


if __name__ == "__main__":
    main()

```

### 22275: 二叉搜索树的遍历

二叉搜索树依赖于这样一个性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。我们称这个性质为二叉搜索性。

```
def post_order(pre_order):
    if not pre_order:
        return []
    root = pre_order[0]
    left_subtree = [x for x in pre_order if x < root]
    right_subtree = [x for x in pre_order if x > root]
    return post_order(left_subtree) + post_order(right_subtree) + [root]

n = int(input())
pre_order = list(map(int, input().split()))
print(' '.join(map(str, post_order(pre_order))))
```

### 05455: 二叉搜索树的层次遍历

```
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(node, value):
    if node is None:
        return TreeNode(value)
    if value < node.value:
        node.left = insert(node.left, value)
    elif value > node.value:
        node.right = insert(node.right, value)
    return node

def level_order_traversal(root):
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal

numbers = list(map(int, input().strip().split()))
numbers = list(dict.fromkeys(numbers))  # remove duplicates
root = None
for number in numbers:
    root = insert(root, number)
traversal = level_order_traversal(root)
print(' '.join(map(str, traversal)))
```

输入只有若干个未排序数字

### 27928: 遍历树

```
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def traverse_print(root, nodes):
    if root.children == []:
        print(root.value)
        return
    pac = {root.value: root}
    for child in root.children:
        pac[child] = nodes[child]
    for value in sorted(pac.keys()):
        if value in root.children:
            traverse_print(pac[value], nodes)
        else:
            print(root.value)


n = int(input())
nodes = {}
children_list = []
for i in range(n):
    info = list(map(int, input().split()))
    nodes[info[0]] = TreeNode(info[0])
    for child_value in info[1:]:
        nodes[info[0]].children.append(child_value)
        children_list.append(child_value)
root = nodes[[value for value in nodes.keys() if value not in children_list][0]]
traverse_print(root, nodes)

```

请你对输入的树做遍历。遍历的规则是：遍历到每个节点时，按照该节点和所有子节点的值从小到大进行遍历

### 02775: 文件结构“图”

```
class Node:
    def __init__(self,name):
        self.name=name
        self.dirs=[]
        self.files=[]

def print_(root,m):
    pre='|     '*m
    print(pre+root.name)
    for Dir in root.dirs:
        print_(Dir,m+1)
    for file in sorted(root.files):
        print(pre+file)
        
tests,test=[],[]
while True:
    s=input()
    if s=='#':
        break
    elif s=='*':
        tests.append(test)
        test=[]
    else:
        test.append(s)
for n,test in enumerate(tests,1):
    root=Node('ROOT')
    stack=[root]
    print(f'DATA SET {n}:')
    for i in test:
        if i[0]=='d':
            Dir=Node(i)
            stack[-1].dirs.append(Dir)
            stack.append(Dir)
        elif i[0]=='f':
            stack[-1].files.append(i)
        else:
            stack.pop()
    print_(root,0)
    print()
```

### 04080:Huffman编码树

```
import heapq
n = int(input())
l = list(map(int,input().split()))
heapq.heapify(l)
ans = 0
for i in range(n-1):
    x = heapq.heappop(l)
    y = heapq.heappop(l)
    z = x + y
    ans += z
    heapq.heappush(l,z)
print(ans)
```

### 01760:Disk Tree

```
class TreeNode:
    def __init__(self,value):
        self.value = value
        self.neighbor=[]
        self.neighborlabor=[]
def print_tree(roots):
    roots.sort(key=lambda x: x.value)
    out=[]
    for i in roots:
        out.append(i.value)
        if i.neighbor:
            i.neighbor.sort(key=lambda x: x.value)
        for j in i.neighbor:
            for k in print_tree([j]):
                out.append(' '+k)
    return out

def build_tree(tree):
    rootslabor = []
    roots = []
    for line in tree:
        if line[0] not in rootslabor:
            rootslabor.append(line[0])
            roots.append(TreeNode(line[0]))
            node = roots[-1]
        else:
            t0 = rootslabor.index(line[0])
            node = roots[t0]
        for t in range(1,len(line)):
            if line[t] not in node.neighborlabor:
                node.neighborlabor.append(line[t])
                node1 = TreeNode(line[t])
                node.neighbor.append(node1)
                node = node1
            else:
                t0 = node.neighborlabor.index(line[t])
                node1 = node.neighbor[t0]
                node = node1
    for k in print_tree(roots):
        print(k)

n = int(input())
tree = []
for _ in range(n):
    l = input()
    l2 = l.strip()
    line = l2.split("\\")
    tree.append(line)
build_tree(tree)
```

### 晴问9.5: 平衡二叉树的建立（AVL）

    class Node:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None
            self.height = 1
    
    class AVL:
        def __init__(self):
            self.root = None
            
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)
    
    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)
    
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
    
        balance = self._get_balance(node)
    
        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
    
        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
    
        return node
    
    def _get_height(self, node):
        if not node:
            return 0
        return node.height
    
    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y
    
    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    
    def preorder(self):
        return self._preorder(self.root)
    
    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)
        
    n = int(input().strip())
    sequence = list(map(int, input().strip().split()))
    
    avl = AVL()
    for value in sequence:
        avl.insert(value)
    
    print(' '.join(map(str, avl.preorder())))

AVL 平衡树的全称是 Adelson-Velsky and Landis 平衡树。它是由两位前苏联的计算机科学家，即Георгий Максимович Адельсон-Вельский（Georgy Maximovich Adelson-Velsky）和Евгений Михайлович Ландис（Evgenii Mikhailovich Landis）于1962年提出的一种自平衡二叉搜索树。

这种树的名称取自这两位科学家的姓氏的首字母缩写。AVL 平衡树通过在每个节点上维护一个平衡因子（balance factor）来实现平衡。平衡因子是指节点的左子树高度与右子树高度之差的绝对值。通过不断调整树的结构，AVL 树能够保持树的平衡，使得在最坏情况下的查找、插入和删除操作的时间复杂度保持在 O(log n)。

AVL 平衡树的特点是在每次插入或删除节点时，会通过旋转操作来调整树的结构，使得平衡因子在特定的范围内，通常是 -1、0、1。这样的平衡状态能够保证树的高度始终保持在较小的范围内，提供了较快的查找和更新操作。

总结起来，AVL 平衡树是一种自平衡二叉搜索树，通过调整树的结构来保持树的平衡性，以提供高效的查找和更新操作。

# 并查集

## 晴问9.6.1 学校的班级个数（1）

最基本的

```
def find(x):
    if parent[x] != x: # 如果不是根结点，继续循环
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

n, m = map(int, input().split())
parent = list(range(n + 1))	# parent[i] == i，则说明元素i是该集合的根结点

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

classes = set(find(x) for x in range(1, n + 1))
print(len(classes))
```

## 01182:食物链

```
n,k = map(int,input().split())
parent = list(range(3*n + 1))
current = [1]*(3*n+1)
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    x = find(x)
    y = find(y)
    if current[x] < current[y]:
        x, y = y, x
    current[x] += current[y]
    parent[y] = x
fake = 0
for i in range(k):
    d,x,y = map(int,input().split())
    if x>n or y>n:
        fake += 1
    else:
        if d == 1:
            if find(x) == find(y+n) or find(x) == find(y+2*n):
                fake += 1
            else:
                union(x,y)
                union(x+n,y+n)
                union(x+2*n,y+2*n)
        if d == 2:
            if find(x) == find(y) or find(x) == find(y+2*n):
                fake += 1
            else:
                union(x,y+n)
                union(x+n,y+2*n)
                union(x+2*n,y)
print(fake)
```

一个点对应多个指标

## 07734: 虫子的生活

```
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


def solve_bug_life(scenarios):
    for i in range(1, scenarios + 1):
        n, m = map(int, input().split())
        uf = UnionFind(2 * n + 1)  # 为每个虫子创建两个节点表示其可能的两种性别
        suspicious = False
        for _ in range(m):
            u, v = map(int, input().split())
            if suspicious:
                continue

            if uf.is_connected(u, v):
                suspicious = True
            uf.union(u, v + n)  # 将u的一种性别与v的另一种性别关联
            uf.union(u + n, v)  # 同理


        print(f'Scenario #{i}:')
        print('Suspicious bugs found!' if suspicious else 'No suspicious bugs found!')
        print()


# 读取场景数量并解决问题
scenarios = int(input())
solve_bug_life(scenarios)
```

# 图论

## 一、定义

**顶点Vertex** 

顶点又称节点，是图的基础部分。

**边Edge** 

边是图的另一个基础部分。两个顶点通过一条边相连，表示它们之间存在关系。边既可以是单向的，也可以是双向的。如果图中的所有边都是单向的，我们称之为有向图。

**度Degree**

顶点的度是指和该顶点相连的边的条数。特别是对于有向图来说，顶点的出边条数称为该顶点的出度，顶点的入边条数称为该顶点的入度。

**权值Weight**

顶点和边都可以有一定属性，而量化的属性称为权值，顶点的权值和边的权值分别称为点权和边权。



有了上述定义之后，再来正式地定义**图Graph**。图可以用G来表示，并且G = (V, E)。其中，V是一个顶点集合，E是一个边集合。每一条边是一个二元组(v, w)，其中w, v∈V。可以向边的二元组中再添加一个元素，用于表示权重。子图s是一个由边e和顶点v构成的集合，其中e⊂E且v⊂V。

**路径Path**

 路径是由边连接的顶点组成的序列。

**环Cycle** 

环是有向图中的一条起点和终点为同一个顶点的路径。没有环的图被称为无环图，没有环的有向图被称为有向无环图，简称为DAG。

## 二、题目

### sy379: 有向图的邻接表 简单

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），将其按邻接表的方式存储，然后输出整个邻接表。

```
n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)

for i in range(n):
    num = len(adjacency_list[i])
    if num == 0:
        print(f"{i}({num})")
    else:
        print(f"{i}({num})", ' '.join(map(str, adjacency_list[i])))
```

### 04128:单词序列

共两行，第一行为开始单词和结束单词（两个单词不同），以空格分开。第二行为若干的单词（各不相同），以空格分隔开来，表示词典。单词长度不超过5,单词个数不超过30

类似词梯

```
from collections import defaultdict
graph = defaultdict(list)
start, end = input().split()
words = input().split()
words.append(end)
l = len(start)
for x in words:
    for i in range(l):
        temp = x[:i] + "_" + x[i + 1:]
        graph[temp].append(x)
vertice = [start]
visited = set()
path = {}
path[start] = (start, 1)
for x in vertice:
    for i in range(l):
        temp = x[:i] + "_" + x[i + 1:]
        for y in graph[temp]:
            if y not in visited:
                visited.add(y)
                vertice.append(y)
                path[y] = (y, path[x][1] + 1)
                if y == end:
                    print(path[y][1])
                    exit()
print(0)
```

### 02754: 八皇后

dfs and similar

```
def is_safe(board, row, col):
    # 检查当前位置是否安全
    # 检查同一列是否有皇后
    for i in range(row):
        if board[i] == col:
            return False
    # 检查左上方是否有皇后
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        if board[i] == j:
            return False
        i -= 1
        j -= 1
    # 检查右上方是否有皇后
    i = row - 1
    j = col + 1
    while i >= 0 and j < 8:
        if board[i] == j:
            return False
        i -= 1
        j += 1
    return True

def queen_dfs(board, row):
    if row == 8:
        # 找到第b个解，将解存储到result列表中
        ans.append(''.join([str(x+1) for x in board]))
        return
    for col in range(8):
        if is_safe(board, row, col):
            # 当前位置安全，放置皇后
            board[row] = col
            # 继续递归放置下一行的皇后
            queen_dfs(board, row + 1)
            # 回溯，撤销当前位置的皇后
            board[row] = 0

ans = []
queen_dfs([None]*8, 0)
#print(ans)
for _ in range(int(input())):
    print(ans[int(input()) - 1])
```

### sy321: 迷宫最短路径（bfs+print_path）

```
from collections import deque

MAX_DIRECTIONS = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid_move(x, y):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and not in_queue[x][y]

def bfs(start_x, start_y):
    queue = deque()
    queue.append((start_x, start_y))
    in_queue[start_x][start_y] = True
    while queue:
        x, y = queue.popleft()
        if x == n - 1 and y == m - 1:
            return
        for i in range(MAX_DIRECTIONS):
            next_x = x + dx[i]
            next_y = y + dy[i]
            if is_valid_move(next_x, next_y):
                prev[next_x][next_y] = (x, y)
                in_queue[next_x][next_y] = True
                queue.append((next_x, next_y))

def print_path(pos):
    prev_position = prev[pos[0]][pos[1]]
    if prev_position == (-1, -1):
        print(pos[0] + 1, pos[1] + 1)
        return
    print_path(prev_position)
    print(pos[0] + 1, pos[1] + 1)

n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

in_queue = [[False] * m for _ in range(n)]
prev = [[(-1, -1)] * m for _ in range(n)]

bfs(0, 0)
print_path((n - 1, m - 1))
```

### dfs有向图判断环

```
def has_cycle(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    color = [0] * n

    def dfs(node):
        if color[node] == 1:
            return True
        if color[node] == 2:
            return False

        color[node] = 1
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node] = 2
        return False

    for i in range(n):
        if dfs(i):
            return "Yes"
    return "No"

# 接收数据
n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
print(has_cycle(n, edges))
```

### dfs找连通块数目

```
def dfs(node, visited, adjacency_list):
    visited[node] = True
    for neighbor in adjacency_list[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, adjacency_list)

n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)

visited = [False] * n
connected_components = 0
for i in range(n):
    if not visited[i]:
        dfs(i, visited, adjacency_list)
        connected_components += 1

print(connected_components)
```

### 顶点有权重（dfs）

```
def max_weight(n, m, weights, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    max_weight = 0

    def dfs(node):
        visited[node] = True
        total_weight = weights[node]
        for neighbor in graph[node]:
            if not visited[neighbor]:
                total_weight += dfs(neighbor)
        return total_weight

    for i in range(n):
        if not visited[i]:
            max_weight = max(max_weight, dfs(i))

    return max_weight

# 接收数据
n, m = map(int, input().split())
weights = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
print(max_weight(n, m, weights, edges))
```

### sy384: 无向图的顶点层号（bfs)

现有一个共n个顶点、m条边的无向连通图（假设顶点编号为从`0`到`n-1`）。我们称从s号顶点出发到达其他顶点经过的最小边数称为各顶点的层号。求图中所有顶点的层号。

```
from collections import deque

def bfs(n, m, s, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    distance = [-1] * n
    distance[s] = 0

    queue = deque([s])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance

# 接收数据
n, m, s = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
distances = bfs(n, m, s, edges)
print(' '.join(map(str, distances)))
```

### 解决最小生成树问题的贪心算法(Kruskal算法)

```
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            if self.rank[px] == self.rank[py]:
                self.rank[py] += 1

def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    res = 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            res += w
    if len(set(uf.find(i) for i in range(n))) > 1:
        return -1
    return res

n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))
print(kruskal(n, edges))
```

### sy396: 最小生成树-Prim算法 简单

```
import heapq

def prim(graph, n):
    visited = [False] * n
    min_heap = [(0, 0)]  # (weight, vertex)
    min_spanning_tree_cost = 0

    while min_heap:
        weight, vertex = heapq.heappop(min_heap)

        if visited[vertex]:
            continue

        visited[vertex] = True
        min_spanning_tree_cost += weight

        for neighbor, neighbor_weight in graph[vertex]:
            if not visited[neighbor]:
                heapq.heappush(min_heap, (neighbor_weight, neighbor))

    return min_spanning_tree_cost if all(visited) else -1

def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]

    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    min_spanning_tree_cost = prim(graph, n)
    print(min_spanning_tree_cost)

if __name__ == "__main__":
    main()

```

### 道路（dijkstra）

```
import heapq

def dijkstra(n, edges, k):
    graph = [[] for _ in range(n+1)]
    for u, v, w, x in edges:
        graph[u].append((v, w, x))

    pq = [(0, 0, 1, 0)]

    while pq:
        dist, money, node, step= heapq.heappop(pq)
        if node == n:
            return dist
        for neighbor, length, weight in graph[node]:
            new_dist = dist + length
            new_money = money + weight
            new_step = step + 1
            if new_money <= k and new_step < n:
                heapq.heappush(pq, (new_dist, new_money, neighbor, new_step))

    return -1

# Read input
k = int(input())
n = int(input())
m = int(input())
edges = [list(map(int, input().split())) for _ in range(m)]

# Solve the problem and print the result
result = dijkstra(n, edges, k)
print(result)
```

过路费+距离

### sy386: 最短距离 简单（dijkstra）

```
import heapq

def dijkstra(n, edges, s, t):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    pq = [(0, s)]  # (distance, node)
    visited = set()
    distances = [float('inf')] * n
    distances[s] = 0

    while pq:
        dist, node = heapq.heappop(pq)
        if node == t:
            return dist
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    return -1

# Read input
n, m, s, t = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

# Solve the problem and print the result
result = dijkstra(n, edges, s, t)
print(result)
```



### 20741:两座孤岛最短距离（bfs+dfs）

```
from collections import deque

def dfs(x, y, grid, n, queue, directions):
    """ Mark the connected component starting from (x, y) as visited using DFS. """
    grid[x][y] = 2  # Mark as visited
    queue.append((x, y))
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
            dfs(nx, ny, grid, n, queue, directions)

def bfs(grid, n, queue, directions):
    """ Perform BFS to find the shortest path to another component. """
    distance = 0
    while queue:
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if grid[nx][ny] == 1:
                        return distance
                    elif grid[nx][ny] == 0:
                        grid[nx][ny] = 2  # Mark as visited
                        queue.append((nx, ny))
        distance += 1
    return distance

def main():
    n = int(input())
    grid = [list(map(int, input())) for _ in range(n)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = deque()

    # Start DFS from the first '1' found and use BFS from there
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j, grid, n, queue, directions)
                return bfs(grid, n, queue, directions)

if __name__ == "__main__":
    print(main())
```

### 28046:词梯

```
from collections import deque

def construct_graph(words):
    graph = {}
    for word in words:
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i + 1:]
            if pattern not in graph:
                graph[pattern] = []
            graph[pattern].append(word)
    return graph

def bfs(start, end, graph):
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        word, path = queue.popleft()
        if word == end:
            return path
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i + 1:]
            if pattern in graph:
                neighbors = graph[pattern]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
    return None

def word_ladder(words, start, end):
    graph = construct_graph(words)
    return bfs(start, end, graph)

n = int(input())
words = [input().strip() for _ in range(n)]
start, end = input().strip().split()

result = word_ladder(words, start, end)

if result:
    print(' '.join(result))
else:
    print("NO")
```

经典例题

## 01258:Agri-Net

```
class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False
        elif self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True

def kruskal(n, edges):
    dsu = DisjointSetUnion(n)
    mst_weight = 0
    for weight, u, v in sorted(edges):
        if dsu.union(u, v):
            mst_weight += weight
    return mst_weight

while True:
    try:
        n = int(input().strip())
        edges = []
        for i in range(n):
            row = list(map(int, input().split()))
            for j in range(i + 1, n):
                if row[j] != 0:
                    edges.append((row[j], i, j))
        print(kruskal(n, edges))
    except EOFError:
        break
```

kruskal算法的应用

# 其他

## 28190: 奶牛排队(单调栈)

```
N = int(input())
heights = [int(input()) for _ in range(N)]

left_bound = [-1] * N
right_bound = [N] * N

stack = []  # 单调栈，存储索引

# 求左侧第一个≥h[i]的奶牛位置
for i in range(N):
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()

    if stack:
        left_bound[i] = stack[-1]

    stack.append(i)

stack = []  # 清空栈以供寻找右边界使用

# 求右侧第一个≤h[i]的奶牛位
for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()

    if stack:
        right_bound[i] = stack[-1]

    stack.append(i)

ans = 0

# for i in range(N-1, -1, -1):  # 从大到小枚举是个技巧
#     for j in range(left_bound[i] + 1, i):
#         if right_bound[j] > i:
#             ans = max(ans, i - j + 1)
#             break
#
#     if i <= ans:
#         break

for i in range(N):  # 枚举右端点 B寻找 A，更新 ans
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)
```

## Mergesort

```
def merge_sort(l):
    if len(l) <= 1:
        return l, 0
    mid = len(l) // 2
    left, left_count = merge_sort(l[:mid])
    right, right_count = merge_sort(l[mid:])
    l, merge_count = merge(left, right)
    return l, left_count + right_count + merge_count


def merge(left, right):
    merged = []
    left_index, right_index = 0, 0
    count = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] >= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
            count += len(left) - left_index
    merged += left[left_index:]+right[right_index:]
    return merged, count


n = int(input())
l = []
for i in range(n):
    l.append(int(input()))
l, ans = merge_sort(l)
print(ans)
```

## 02746:约瑟夫问题

```
def f(n,m):
    monkey = [1] * n
    j = 0
    for i in range(n-1):
        count = 0
        while count < m:
            if monkey[j]:
                count += 1
            if count == m:
                monkey[j] = 0
            j = (j+1) % n
    return monkey.index(1)+1
while True:
    try:
        a,b = map(int,input().split())
        if a:
            print(f(a,b))
    except EOFError:
        break
```

轮流枪毙

## 02773: 采药（0，1背包问题）

```
T, M = map(int, input().split())
dp = [ [0] + [0]*T for _ in range(M+1)]

t = [0]
v = [0]
for i in range(M):
        ti, vi = map(int, input().split())
        t.append(ti)
        v.append(vi)

for i in range(1, M+1):			# 外层循环（行）药草M
        for j in range(0, T+1):	# 内层循环（列）时间T
                if j >= t[i]:
                        dp[i][j] = max(dp[i-1][j], dp[i-1][j-t[i]] + v[i])
                else:
                        dp[i][j] = dp[i-1][j]

print(dp[M][T])
```

## 04135:月度开销（二分查找）

```
n, m = map(int, input().split())
expenditure = [int(input()) for _ in range(n)]

left,right = max(expenditure), sum(expenditure)

def check(x):
    num, s = 1, 0
    for i in range(n):
        if s + expenditure[i] > x:
            s = expenditure[i]
            num += 1
        else:
            s += expenditure[i]
    
    return [False, True][num > m]

res = 0

def binary_search(lo, hi):
    if lo >= hi:
        global res
        res = lo
        return
    
    mid = (lo + hi) // 2
    #print(mid)
    if check(mid):
        lo = mid + 1
        binary_search(lo, hi)
    else:
        hi = mid
        binary_search(lo, hi)
        
binary_search(left, right)
print(res)
```



# Notice

import sys

sys.setrecursionlimit()

Print(*l,sep='')

x.sort(key=lambda x:(x[0],x[1]))

```
while True:
    try:
        
    except EOFError:
        break
```

```
from functools import lru_cache
@lru_cache(maxsize=None)
```

defaultdict是 collections 包下的一个模块，defaultdict 在初始化时可以提供一个 default_factory 的参数，default_factory 接收一个工厂函数作为参数， 可以是 int、str、list 等内置函数，也可以是自定义函数。

# 课程总结

在数算课程的学习过程中，不知不觉也已经过去了一个学期。数算这门课程的新知识点特别多，特别是对于我这样信息基础不好的同学来说任务量比较大，花费的时间也比较多。老师在课程上十分用心，当在课程群里面有提问多时候，老师会很详细地回答，包括一些群里的大佬也会给出自己的解答。对我而言，虽然考试成绩并不算理想，但我在这个课上学到了很多新的，有用的知识，也对编程有了更加深刻的理解，进步了很多。所以特别感谢闫老师在这一个学期的悉心教导，让我受益匪浅。