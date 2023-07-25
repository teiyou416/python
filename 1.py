board = [[0], [1], [0], [1], [1], [0],[1]]

# 邻居数组为给定的单元格找到8个相邻的单元格
neighbors = [1,-1]

rows = len(board)

# 创建一个原始板的副本
print(board)
print("--------------------------------------------------")
copy_board = [[board[row]] for row in range(rows)]
# 逐个单元地迭代
for row in range(rows):     
         #规则1或规则3
        if copy_board[row]== 1 and (copy_board[row+1]==1 and copy_board[row-1]==1):
            copy_board[row] = 1
        if copy_board[row]== 1 and (copy_board[row+1]==0 and copy_board[row-1]==1):
            copy_board[row] = 0
        if copy_board[row]== 1 and (copy_board[row+1]==1 and copy_board[row-1]==0):
            copy_board[row] = 1
        if copy_board[row]== 1 and (copy_board[row+1]==0 and copy_board[row-1]==0):
            copy_board[row] = 0
        if copy_board[row] == 0 and (copy_board[row+1]==0 and copy_board[row-1]==1):
            copy_board[row] = 1
        if copy_board[row] == 0 and (copy_board[row+1]==1 and copy_board[row-1]==0):
            copy_board[row] = 0      
        if copy_board[row] == 0 and (copy_board[row+1]==1 and copy_board[row-1]==1):
            copy_board[row] = 0
        if copy_board[row] == 0 and (copy_board[row+1]==0 and copy_board[row-1]==0):
            copy_board[row] = 1
        
        print(copy_board)

