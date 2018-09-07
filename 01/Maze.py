import numpy as np

class Node(object):
    def __init__(self, row, col, father = None):
        self.row = row
        self.col = col
        self.father = father
    def print(self):
        print("(%d, %d) "%(self.row, self.col), end = "")

is_vis = np.zeros((18, 36), dtype = int)
maze_table = np.zeros((18, 36), dtype = int)
target_row, target_col, begin_row, begin_col = 0, 0, 0, 0
row_steps = [-1, 0, 1, 0]
col_steps = [0, 1, 0, -1]

def isValid(row, col):
    if row >= 18 or row < 0 or col >= 36 or col < 0:
        return False
    if maze_table[row, col] == 1 or is_vis[row, col] == 1:
        return False
    return True

def printNode(now_node):
    if now_node == None:
        return
    printNode(now_node.father)
    now_node.print()

def bfs():
    queue = [Node(begin_row, begin_col)]
    flag = 0
    while len(queue) > 0 and flag == 0:
        now_father = queue[0]
        row, col = now_father.row, now_father.col
        for i in range(4):
            if isValid(row + row_steps[i], col + col_steps[i]):
                now_node = Node(row + row_steps[i], col + col_steps[i], now_father)
                queue.append(now_node)
                is_vis[row + row_steps[i], col + col_steps[i]] = 1
                if now_node.row == target_row and now_node.col == target_col:
                    flag = 1
                    break
        queue.pop(0)
    printNode(queue[len(queue) - 1])
    
if __name__ == '__main__':
    with open('MazeData.txt') as f:
        row, col = 0, 0
        for line in f.readlines():
            now_line = str(line).strip()
            col = 0
            for ch in now_line:
                if ch == 'S':
                    begin_row = row
                    begin_col = col
                    is_vis[row, col] = 1
                    maze_table[row, col] = 2
                elif ch == 'E':
                    target_row = row
                    target_col = col
                    maze_table[row, col] = 3
                else:
                    maze_table[row, col] = int(ch)
                col += 1
            row += 1
    bfs()