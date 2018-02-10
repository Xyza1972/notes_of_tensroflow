import numpy as np
import pandas as pd
from math import inf
import time
import pickle

# class Node:
#     def __init__(self, row, col):
#         self.row = row
#         self.col = col
#         self.pos = [row, col]
#         self.value = df.loc[row, col]
#         self.left = None
#         self.up = None
#         self.right = None
#         self.down = None
#
# def set_graph(root):
#     if not root:
#         return
#
#     global temp
#     row = root.row
#     col = root.col
#     value = root.value
#
#     temp += 1
#
#     if col> 0 and df.loc[row, col-1] < value:
#         root.left = Node(row, col-1)
#     if row > 0 and df.loc[row-1, col] < value:
#         root.up = Node(row-1, col)
#     if col < max_col and df.loc[row, col+1] < value:
#         root.right = Node(row, col+1)
#     if row < max_row and df.loc[row+1, col] < value:
#         root.down = Node(row+1, col)
#
#     set_graph(root.left)
#     set_graph(root.up)
#     set_graph(root.right)
#     set_graph(root.down)
#
# def output(root):
#     if not root:
#         return
#
#     global g
#     key = ','.join(str(i) for i in root.pos)
#     if root.left:
#         if key in g:
#             g[key].add(','.join(str(i) for i in root.left.pos))
#         else:
#             g[key] = set()
#             g[key].add(','.join(str(i) for i in root.left.pos))
#     if root.up:
#         if key in g:
#             g[key].add(','.join(str(i) for i in root.up.pos))
#         else:
#             g[key] = set()
#             g[key].add(','.join(str(i) for i in root.up.pos))
#     if root.right:
#         if key in g:
#             g[key].add(','.join(str(i) for i in root.right.pos))
#         else:
#             g[key] = set()
#             g[key].add(','.join(str(i) for i in root.right.pos))
#     if root.down:
#         if key in g:
#             g[key].add(','.join(str(i) for i in root.down.pos))
#         else:
#             g[key] = set()
#             g[key].add(','.join(str(i) for i in root.down.pos))
#     output(root.left)
#     output(root.up)
#     output(root.right)
#     output(root.down)

def is_max(df, row, col, max_row, max_col):
    value = df.loc[row, col]
    if col> 0 and df.loc[row, col-1] > value:
        return False
    if row > 0 and df.loc[row-1, col] > value:
        return False
    if col < max_col and df.loc[row, col+1] > value:
        return False
    if row < max_row and df.loc[row+1, col] > value:
        return False
    return True

class Skiing:
    def __init__(self, df, root_pos, max_row, max_col):
        self.df = df
        self.max_row = max_row
        self.max_col = max_col
        self.root_row = root_pos[0]
        self.root_col = root_pos[1]
        self.root_pos = root_pos
        self.root = ','.join([str(self.root_row), str(self.root_col)])
        self.graph = {}
        self.no_edges = 0
        self.node_list = []

    @property
    def no_nodes(self):
        return len(self.node_list)

    def is_min(self, row, col):
        value = self.df.loc[row, col]
        if col> 0 and self.df.loc[row, col-1] < value:
            return False
        if row > 0 and self.df.loc[row-1, col] < value:
            return False
        if col < self.max_col and self.df.loc[row, col+1] < value:
            return False
        if row < self.max_row and self.df.loc[row+1, col] < value:
            return False
        return True

    def make_graph(self, row, col):
        self.no_edges += 1
        key = ','.join([str(row), str(col)])
        if key not in self.node_list:
            self.node_list.append(key)
        if self.is_min(row, col):
            return
        value = self.df.loc[row, col]

        if col> 0 and self.df.loc[row, col-1] < value:
            left = ','.join([str(row), str(col-1)])
            if key in self.graph:
                if left in self.graph[key]:
                    pass
                else:
                    self.graph[key].append(left)
            else:
                self.graph[key] = []
                self.graph[key].append(left)
            if left in self.graph:
                pass
            else:
                self.make_graph(row, col-1)

        if row > 0 and self.df.loc[row-1, col] < value:
            up = ','.join([str(row-1), str(col)])
            if key in self.graph:
                if up in self.graph[key]:
                    pass
                else:
                    self.graph[key].append(up)
            else:
                self.graph[key] = []
                self.graph[key].append(up)

            if up in self.graph:
                pass
            else:
                self.make_graph(row-1, col)

        if col < self.max_col and self.df.loc[row, col+1] < value:
            right = ','.join([str(row), str(col+1)])
            if key in self.graph:
                if right in self.graph[key]:
                    pass
                else:
                    self.graph[key].append(right)
            else:
                self.graph[key] = []
                self.graph[key].append(right)

            if right in self.graph:
                pass
            else:
                self.make_graph(row, col+1)

        if row < self.max_row and self.df.loc[row+1, col] < value:
            down = ','.join([str(row+1), str(col)])
            if key in self.graph:
                if down in self.graph[key]:
                    pass
                else:
                    self.graph[key].append(down)
            else:
                self.graph[key] = []
                self.graph[key].append(down)

            if down in self.graph:
                pass
            else:
                self.make_graph(row+1, col)

    def get_value(self, pos):
        if isinstance(pos, str):
            loc = pos.split(',')
        else:
            loc = pos
        row = int(loc[0])
        col = int(loc[1])
        value = self.df.loc[row, col]
        return value

    def topologicalSort(self, root, visited, stack):
        visited[root] = True
        if root not in self.graph:
            stack.append(root)
            return
        for node in self.graph[root]:
            if visited[node] == False:
                self.topologicalSort(node, visited, stack)
        stack.append(root)

    def find_longest_path(self):
        visited = dict.fromkeys(self.node_list, False)
        stack=[]

        for node in visited:
            if visited[node] == False:
                self.topologicalSort(node, visited, stack)

        dist = dict.fromkeys(self.node_list, -inf)
        dist[self.root] = 1

        while (stack):
            vertex = stack.pop()
            if dist[vertex] != -inf and vertex in self.graph:
                for i in self.graph[vertex]:
                    if dist[i] < dist[vertex] + 1:
                        dist[i] = dist[vertex]+1

        dist_df = pd.DataFrame.from_dict(data=dist, orient='index')
        dist_df.columns = ['dist']
        length = dist_df['dist'].max()
        end_nodes = dist_df.loc[dist_df['dist']==length].index.values
        end_values = []
        for pos in end_nodes:
            end_values.append(self.get_value(pos))

        start = self.get_value(self.root)
        end = min(end_values)

        return length, start-end

def main():

    # Read data from text and convert to pandas
    df = pd.read_table('./map.txt', header=None, delim_whitespace=True, skiprows=1)
    max_row = df.shape[0]-1
    max_col = df.shape[1]-1
    df.reset_index()
    max_length = 0
    max_drop = 0

    root_list = []
    print('----- find all possible roots -----')
    start = time.time()
    for index, row in df.iterrows():
        for idx, value in row.iteritems():
            if is_max(df, index, idx, max_row, max_col):
                root_list.append([index,idx])
    with open('roots.npy', 'w') as f:
        np.save(f, root_list)

    max_loop=len(root_list)
    print('----- find total {} roots in {} sec-----'.format(max_loop, time.time()-start))
    cnt = 0

    print('----- start looping through roots -----')
    start = time.time()
    for root in root_list:
        cnt += 1
        if cnt%500 == 0:
            print('processed {} roots, at {} sec'.format(cnt, time.time()-start) )

        skier = Skiing(df, root, max_row, max_col)
        skier.make_graph(root[0], root[1])
        if skier.no_edges <= max_length:
            continue
        else:
            length, drop = skier.find_longest_path()
            if length >=max_length:
                max_length = length
                if drop>max_drop:
                    max_drop = drop

    print('----- finish finially!!! -----')
    print('time taken: {}'.format(time.time()-start))
    print('max length: {}'.format(max_length))
    print('max drop: {}'.format(max_drop))

if __name__ == "__main__":
    main()
