"""
Question - 1
Given two strings s and t, determine whether some anagram of t is a substring of s.

"""
def is_anagram(s1, s2):
    s1 = list(s1)
    s2 = list(s2)
    s1.sort()
    s2.sort()
    return s1 == s2

def question1(s, t):
    match_length = len(t)
    pattern_length = len(s)
    for i in range(pattern_length - match_length + 1):
        if is_anagram(s[i: i+match_length], t):
            return True
    return False

print question1('audacity', 'udacity') # True
print question1('udacity', 'od') # False

"""
Question - 2 
Given a string a, find the longest palindromic substring contained in a.
Your function definition should look like question2(a), and return a string.
"""

def isPalindrome(s):
    if not s:
        return False
    
    return s == s[::-1]

def question2(s):
    if not s:
        return ""

    n = len(s)
    longest, left, right = 0, 0, 0
    for i in xrange(0, n):
        for j in xrange(i + 1, n + 1):
            substr = s[i:j]
            if isPalindrome(substr) and len(substr) > longest:
                longest = len(substr)
                left, right = i, j
    
    result = s[left:right]
    return result
    
    
print question2("malayalam") # malayalam
print question2("bananas") # anana
print question2("shishir") # s



'''	
Question - 3
Given an undirected graph G, find the minimum spanning tree within G. A minimum spanning tree connects all vertices in a graph with the smallest possible total weight of edges. 


'''

import collections

parent = dict()
rank = dict()

def make_set(vertex):
    parent[vertex] = vertex
    rank[vertex] = 0

def find(vertex):
    if parent[vertex] != vertex:
        parent[vertex] = find(parent[vertex])
    return parent[vertex]

def union(vertex1, vertex2):
    root1 = find(vertex1)
    root2 = find(vertex2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]:
                rank[root2] += 1

def question3(graph):
    for vertex in graph.keys():
        make_set(vertex)

    minimum_spanning_tree = set()

    edges = get_edges(graph)
    edges.sort()
    for edge in edges:
        weight, vertex1, vertex2 = edge
        if find(vertex1) != find(vertex2):
            union(vertex1, vertex2)
            minimum_spanning_tree.add(edge)

    adj = collections.defaultdict(list)
    for weight, vertex1, vertex2 in minimum_spanning_tree:
        adj[vertex1].append((vertex2, weight))
        adj[vertex2].append((vertex1, weight))
    return adj


def get_edges(adj):
    edge_list = []
    for vertex, edges in adj.iteritems():
        for edge in edges:
            if vertex < edge[0]:
                edge_list.append((edge[1], vertex, edge[0]))
    return edge_list

graph1 = {
    'A': [('B', 1), ('C', 5), ('D', 3)],
    'B': [('A', 1), ('C', 4), ('D', 2)],
    'C': [('B', 4), ('D', 1)],
    'D': [('A', 3), ('B', 2), ('C', 1)],
}
minimum_spanning_tree1 = {
    'A': [('B', 1)],
    'B': [('A', 1), ('D', 2)],
    'C': [('D', 1)],
    'D': [('C', 1), ('B', 2)]
}

graph2 = {
    'A': [('B', 2), ('C', 5)],
    'B': [('A', 2), ('C', 4)],
    'C': [('A', 5), ('B', 4)]
}

minimum_spanning_tree2 = {
    'A': [('B', 2)],
    'B': [('A', 2), ('C', 4)],
    'C': [('B', 4)]
}


graph3 = {
    'A': [('B', 2), ('C', 3)],
    'B': [('A', 2), ('C', 4), ('D', 2)],
    'C': [('A', 3), ('B', 4), ('D', 3), ('E', 2), ('F', 6), ('G', 3)],
    'D': [('B', 2), ('C', 3), ('E', 1)],
    'E': [('C', 2), ('D', 1), ('G', 2)],
    'F': [('C', 6), ('G', 4)],
    'G': [('C', 3), ('E', 2), ('F', 4)]
}


minimum_spanning_tree3 = {
    'A': [('B', 2)],
    'B': [('A', 2), ('D', 2)],
    'C': [('E', 2)],
    'D': [('E', 1), ('B', 2)],
    'E': [('D', 1), ('C', 2), ('G', 2)],
    'F': [('G', 4)],
    'G': [('E', 2), ('F', 4)]
}



print question3(graph1) # minimum_spanning_tree1
print question3(graph2) # minimum_spanning_tree2
print question3(graph3) # minimum_spanning_tree3



'''
Question - 4
Find the least common ancestor between two nodes on a binary search tree. The least common ancestor is the farthest node from the root that is an ancestor of both nodes. For example, the root is a common ancestor of all nodes on the tree, but if both nodes are descendents of the root's left child, then that left child might be the lowest common ancestor. You can assume that both nodes are in the tree, and the tree itself adheres to all BST properties. The function definition should look like question4(T, r, n1, n2), where T is the tree represented as a matrix, where the index of the list is equal to the integer stored in that node and a 1 represents a child node, r is a non-negative integer representing the root, and n1 and n2 are non-negative integers representing the two nodes in no particular order.
'''

class Element(object):
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class BST(object):
    def __init__(self, root):
        self.root = Element(root)

    def insert(self, new_val):
        self.insert_helper(self.root, new_val)

    def insert_helper(self, current, new_val):
        if current.data < new_val:
            if current.right:
                self.insert_helper(current.right, new_val)
            else:
                current.right = Element(new_val)
        else:
            if current.left:
                self.insert_helper(current.left, new_val)
            else:
                current.left = Element(new_val)

    def search(self, find_val):
        return self.search_helper(self.root, find_val)

    def search_helper(self, current, find_val):
        if current:
            if current.data == find_val:
                return True
            elif current.data < find_val:
                return self.search_helper(current.right, find_val)
            else:
                return self.search_helper(current.left, find_val)
        return False

def lca(root, n1, n2):

    
    if root is None:
        return None

   
    if(root.data > n1 and root.data > n2):
        return lca(root.left, n1, n2)

   
    if(root.data < n1 and root.data < n2):
        return lca(root.right, n1, n2)

    return root.data


def question4(matrix, root, n1, n2):
    bst = BST(root)
    for node in matrix[root]:
        bst.insert(node)

    
    for row in reversed(range(len(matrix))):
        for node in matrix[row]:
            bst.insert(node)


    return lca(bst.root, n1, n2)
    
    
print question4([[0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]],
                  3,
                  1,
                  4) # 3
                  
print question4([[0, 0, 0, 0, 0],
                 [1, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 1],
                 [0, 0, 0, 0, 0]],
                 3,
                 1,
                 2) # 1


'''
Question - 5
Find the element in a singly linked list that's m elements from the end.
For example, if a linked list has 5 elements, the 3rd element from the end is
the 3rd element. The function definition should look like question5(ll, m),
where ll is the first node of a linked list and m is the "mth number from the
end.
'''

global ll
ll = None

class Element(object):
    def __init__(self, data):
        self.data = data
        self.next = None

def add(new_data):
    global ll
    node = Element(new_data)
    node.next = ll
    ll = node

def question5(ll, m):
    element1 = ll
    element2 = ll
    c = 0
    
    if(ll is not None):
        while(c < m):
            element2 = element2.next
            c += 1

    while(element2 is not None):
        element1 = element1.next
        element2 = element2.next

    return element1.data

add("0")
add("9 WXYZ")
add("8 TUV")
add("7 PQRS")
add("6 MNO")
add("5 JKL")
add("4 GHI")
add("3 DEF")
add("2 ABC")
add("1")


print question5(ll, 1) # 0
print question5(ll, 4)  # 7 PQRS
print question5(ll, 8) # 3 DEF
