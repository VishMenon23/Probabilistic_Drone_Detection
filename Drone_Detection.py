from pickle import TRUE
import random
import numpy as np
from queue import PriorityQueue
from collections import deque
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.patches as mpatches




def Move_Right(matrix,temp_prob_matrix):
    copy_of_beliefs = np.copy(temp_prob_matrix)
    rows = len(matrix)
    cols = len(matrix[0])
    	
    for i in range(rows):
        for j in range(cols):
            if(j==cols-1):                                                              # Edge Case
                if(matrix[i][j]!=1 and matrix[i][j-1]!=1):                              # Probability calculation for rightmost cell
                    temp_prob_matrix[i][j]=copy_of_beliefs[i][j]+copy_of_beliefs[i][j-1]    
                    
            if(j-1>=0 and j!=cols-1):                                                  # Edge Case
                if(matrix[i][j]!=1 and matrix[i][j+1]!=1 and matrix[i][j-1]!=1):      # Probability calculation for cells with no walls to the right or left
                    temp_prob_matrix[i][j]=copy_of_beliefs[i][j-1]

            if(j-1>=0 and j!=cols-1):                                                  # Edge Case
                if(matrix[i][j]!=1 and matrix[i][j+1]==1 and matrix[i][j-1]!=1):      # Wall to the right
                    temp_prob_matrix[i][j]=copy_of_beliefs[i][j]+copy_of_beliefs[i][j-1]

            if(j-1>=0 and j!=cols-1):                                                  # Edge Case
                if(matrix[i][j]!=1 and matrix[i][j-1]==1):                              # Wall to the left
                    temp_prob_matrix[i][j]=0
             
            if(j==0):                                                                    # First column
                if(matrix[i][j]!=1 and matrix[i][j+1]!=1):
                    temp_prob_matrix[i][j]=0

            if(j!=0 and j+1<=cols-1 and j-1>=0):                                    # Edge Case
                if(matrix[i][j]!=1 and matrix[i][j-1]==1 and matrix[i][j+1]==1):     # Wall on the right and left
                    temp_prob_matrix[i][j]=copy_of_beliefs[i][j] 
    return temp_prob_matrix

def Move_Left(matrix,temp_prob_matrix):
    copy_of_beliefs = np.copy(temp_prob_matrix)
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(rows):
        for j in range(cols):
            if(j == 0):                                                                         # Probability calculation for leftmost cell
                if(matrix[i][j] != 1 and matrix[i][j+1] != 1): 
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j] + copy_of_beliefs[i][j+1]

            if(j+1 <= cols-1 and j != 0):                                                       # Probability calculation for cells with no walls to the right or left
                if(matrix[i][j] !=1 and matrix[i][j+1] != 1 and matrix[i][j-1] != 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j+1]

            if(j+1 <= cols-1 and j != 0):                                                       # Wall to the left and no wall to the right
                if(matrix[i][j] !=1 and matrix[i][j+1] != 1 and matrix[i][j-1] == 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j] + copy_of_beliefs[i][j+1]

            if(j+1 <= cols-1 and j != 0):                                                       # Wall to the right and no wall to the left
                if(matrix[i][j] !=1 and matrix[i][j+1] == 1):
                    temp_prob_matrix[i][j] = 0

            if(j == (cols-1)):
                if(matrix[i][j] != 1 and matrix[i][j-1] != 1):
                    temp_prob_matrix[i][j] = 0

            if(j != 0 and j+1 <= cols - 1 and j-1 >=0):                                         # Wall to the left and right
                if(matrix[i][j] != 1 and matrix[i][j-1] == 1 and matrix[i][j+1] == 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j]

    return temp_prob_matrix

def Move_Up(matrix,temp_prob_matrix):
    copy_of_beliefs = np.copy(temp_prob_matrix)
    rows = len(matrix)
    cols = len(matrix[0])

    for j in range(cols):
        for i in range(rows):
            if(i==0):                                                                           # Probability of the top row
                if(matrix[i][j] != 1 and matrix[i+1][j] != 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j] + copy_of_beliefs[i+1][j]

            if(i+1 <= rows - 1 and i != 0):                                                     # Probability calculation for cells with no walls above or below
                if(matrix[i][j] != 1 and matrix[i+1][j] != 1 and matrix[i-1][j] != 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i+1][j]

            if(i+1 <= rows - 1 and i != 0):                                                     # Wall above and no wall below
                if(matrix[i][j] != 1 and matrix[i+1][j] != 1 and matrix[i-1][j] == 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j] + copy_of_beliefs[i+1][j]

            if(i+1 <= rows-1 and i != 0):                                                       # Wall below
                if(matrix[i][j] != 1 and matrix[i+1][j] == 1):
                    temp_prob_matrix[i][j] = 0

            if(i == rows - 1):
                if(matrix[i][j] != 1 and matrix[i-1][j] != 1):
                    temp_prob_matrix[i][j] = 0

            if(i != 0 and i+1 <= rows - 1 and i-1 >=0):                                          # No walls above or below
                if(matrix[i][j] != 1 and matrix[i+1][j] == 1 and matrix[i-1][j] == 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j]
    return temp_prob_matrix

def Move_Down(matrix,temp_prob_matrix):
    copy_of_beliefs = np.copy(temp_prob_matrix)
    rows = len(matrix)
    cols = len(matrix[0])

    for j in range(cols):
        for i in range(rows):
            if(i== rows -1):                                                                        # Probability of the bottom row
                if(matrix[i][j] != 1 and matrix[i-1][j] != 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j] + copy_of_beliefs[i-1][j]

            if(i+1 <= rows - 1 and i != 0):                                                         # Probability calculation for cells with no walls above or below
                if(matrix[i][j] != 1 and matrix[i+1][j] != 1 and matrix[i-1][j] != 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i-1][j]

            if(i+1 <= rows - 1 and i != 0):                                                         # Wall below and no wall above
                if(matrix[i][j] != 1 and matrix[i+1][j] == 1 and matrix[i-1][j] != 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j] + copy_of_beliefs[i-1][j]

            if(i-1 >= 0 and i != rows - 1):                                                         # Wall above
                if(matrix[i][j] != 1 and matrix[i-1][j] == 1):
                    temp_prob_matrix[i][j] = 0

            if(i == 0):
                if(matrix[i][j] != 1 and matrix[i+1][j] != 1):
                    temp_prob_matrix[i][j] = 0

            if(i != 0 and i+1 <= rows - 1 and i-1 >=0):                                              # Wall above and below
                if(matrix[i][j] != 1 and matrix[i+1][j] == 1 and matrix[i-1][j] == 1):
                    temp_prob_matrix[i][j] = copy_of_beliefs[i][j]
    return temp_prob_matrix

#A-Star heuristic 
def heuristic_evaluation(cell1,cell2):
    x1,y1=cell1
    x2,y2=cell2
    return abs(x1-x2) + abs(y1-y2)                                                          # Manhattan Distance between the cells. 

def astar_original(temp_matrix, s,d):
    count_astar=0
    source=(s[0],s[1])
    destination=(d[0],d[1])
    g_score={}                                                                  # Dictionary to store the distance between the source and each cell
    f_score={}                                                                  # Dictionary to store the value of g(n) + h(n)
    for i in range(rows):
        for j in range(cols):
            g_score[(i,j)]=float('inf')                                         # Initializing the g and f value of each cell to a maximum 
            f_score[(i,j)]=float('inf')
    g_score[source]=0                                                           # Initializing the scores of the source node
    f_score[source]=heuristic_evaluation(source,(d[0],d[1]))                                                 
    pq=PriorityQueue()
    pq.put((heuristic_evaluation(source,(d[0],d[1])),heuristic_evaluation(source,(d[0],d[1])),source))            
    Path={}
    flag=0
    while not pq.empty():
        current_cell=pq.get()[2]                                                                       # The cell with the smallest f score is returned by the priority queue
        if current_cell==(d[0],d[1]):                                                      # The destination has been reached
            flag=1
            break
        x = current_cell[0]
        y = current_cell[1]
        l1=[x,y-1]
        l2=[x,y+1]
        l3=[x-1,y]
        l4=[x+1,y]                                                                                      # The neighbors are now examined
        if (y - 1 >= 0 and temp_matrix[x][y - 1] == 0):               # Left child
            child_cell=(x,y-1)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(d[0],d[1]))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(d[0],d[1])),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
                   
        if (y + 1 <= cols-1 and temp_matrix[x][y + 1] == 0):      # Right child
            child_cell=(x,y+1)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(d[0],d[1]))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(d[0],d[1])),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
                    
        if (x - 1 >= 0 and temp_matrix[x - 1][y] == 0):                # Up child
            child_cell=(x-1,y)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(d[0],d[1]))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(d[0],d[1])),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
                      
        if (x + 1 <= rows-1 and temp_matrix[x + 1][y] == 0):     # Down child
            child_cell=(x+1,y)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(d[0],d[1]))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(d[0],d[1])),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
    ans=[]
    if(flag==0):
        return ans        
    fwdPath={}
    cell=(d[0],d[1])
    ans.append((d[0],d[1]))
    while cell!=source:                                                                                 # Traversing the path using the parent to reconstruct the path
        ans.append(Path[cell])
        cell=Path[cell]
    return ans[::-1]

def Search_For_Drone(matrix,prob_matrix):
    Action_Set=['Right','Left','Up','Down']                       # Defining the actions available for the drone
    list_of_actions=[]                                            # List to store every move made  
    number_of_moves=0                                             # Counter variable to maintain number of steps  
    for action in Action_Set:                                     # Running each section until most of the probabilities accumulate near the walls/boundaries  
        print('Action = ',action)
        if(action=='Right'):                                      # Right  
            for i in range(cols):
                wall_check=0
                prob_matrix=Move_Right(matrix,prob_matrix)
                for r in range(rows):
                    for c in range(cols):
                        if(prob_matrix[r][c]!=0):
                            if(c+1<cols and matrix[r][c+1]==0):
                                wall_check=1
                list_of_actions.append('Right')
                # plt.imshow(prob_matrix)
                # plt.show()
                number_of_moves=number_of_moves+1
                if(wall_check==0):                                 # Breaking if all probabilities are near the walls/boundaries  
                    break
        if(action=='Left'):                                        # Left
            for i in range(cols):
                wall_check=0
                prob_matrix=Move_Left(matrix,prob_matrix) 
                for r in range(rows):
                    for c in range(cols):
                        if(prob_matrix[r][c]!=0):
                            if(c-1>=0 and matrix[r][c-1]==0):
                                wall_check=1
                list_of_actions.append('Left')
                # plt.imshow(prob_matrix)
                # plt.show()
                number_of_moves=number_of_moves+1 
                if(wall_check==0):
                    break
        if(action=='Up'):                                           # Up
            for i in range(rows):
                wall_check=0
                prob_matrix=Move_Up(matrix,prob_matrix)
                for r in range(rows):
                    for c in range(cols):
                        if(prob_matrix[r][c]!=0):
                            if(r-1>=0 and matrix[r-1][c]==0):
                                wall_check=1
                list_of_actions.append('Up')
                # plt.imshow(prob_matrix)
                # plt.show()
                number_of_moves=number_of_moves+1
                if(wall_check==0):
                    break
        if(action=='Down'):                                         # Down
            for i in range(rows):
                wall_check=0
                prob_matrix=Move_Down(matrix,prob_matrix)
                for r in range(rows):
                    for c in range(cols):
                        if(prob_matrix[r][c]!=0):
                            if(r+1<rows and matrix[r+1][c]==0):
                                wall_check=1
                list_of_actions.append('Down') 
                # plt.imshow(prob_matrix)
                # plt.show()
                number_of_moves=number_of_moves+1 
                if(wall_check==0):
                    break
        print(prob_matrix)
        print('____________________________________________________')  
    # CHECK   
    values=[]
    for x in range(rows):
        for y in range(cols):
            if(prob_matrix[x][y]!=0):
                values.append(prob_matrix[x][y])
    plt.imshow(prob_matrix,cmap='Reds')
    plt.show()
    # CHECK 
    sum=0
    for x in range(rows):
        for y in range(cols):
            sum=sum+prob_matrix[x][y]
    print("SUM ",sum)
    print("number_of_moves ",number_of_moves)
    
    print(prob_matrix)

    while(True):
        vals=[]
        for r in range(rows):                                               # Storing all non-zero elements of the probability matrix
            for c in range(cols):
                if(prob_matrix[r][c]!=0):
                    vals.append((r,c)) 
        if(len(vals)==1):                                                   # Termination Condition
            break
        print("vals ",vals)
        distances={}                                                        # Dictionary to store the points and the distances between them
        for s in range(len(vals)):
            source=vals[s]
            for d in range(len(vals)):
                destination=vals[d]
                if(source==destination):
                    continue
                path=astar_original(matrix,source,destination)
                distances[source,destination]=len(path)
        sorted_values = sorted(distances.values())
        sorted_dist={}
        for i in sorted_values:                                             # Sorting the dictionary based on values
            for k in distances.keys():
                if distances[k] == i:
                    sorted_dist[k] = distances[k]       
        point=list(sorted_dist.keys())[0]
        path = astar_original(matrix,point[0],point[1])  
        a=point[0]
        b=point[1]
        next_point=path[1]
        if(a[0]==next_point[0] and a[1]+1==next_point[1]):                  # Moving the probabilities towards each other based on the direction from the path
            print('Goes Right')
            prob_matrix=Move_Right(matrix,prob_matrix)
            list_of_actions.append('Right')
            # plt.imshow(prob_matrix)
            # plt.show()
            number_of_moves=number_of_moves+1
        elif(a[0]==next_point[0] and a[1]-1==next_point[1]):
            print('Goes Left')
            prob_matrix=Move_Left(matrix,prob_matrix) 
            list_of_actions.append('Left')
            # plt.imshow(prob_matrix)
            # plt.show()
            number_of_moves=number_of_moves+1 
        elif(a[0]+1==next_point[0] and a[1]==next_point[1]):
            print('Goes Down')
            prob_matrix=Move_Down(matrix,prob_matrix)
            list_of_actions.append('Down')
            # plt.imshow(prob_matrix)
            # plt.show()
            number_of_moves=number_of_moves+1
        elif(a[0]-1==next_point[0] and a[1]==next_point[1]):
            print('Goes Up')
            prob_matrix=Move_Up(matrix,prob_matrix)
            list_of_actions.append('Up') 
            # plt.imshow(prob_matrix)
            # plt.show()
            number_of_moves=number_of_moves+1   
        print(prob_matrix)
        print() 
        print("number_of_moves= " ,number_of_moves)       
        flag=0
        for r in range(rows):
            for c in range(cols):
                if(prob_matrix[r][c]==1):
                    flag=1
                    break  
            if(flag==1):    
                break  
        if(flag==1):
            break
    return [number_of_moves,list_of_actions]    


with open('schema.txt') as f:                                 # Reading the Text file
    lines = f.readlines()  
rows=len(lines)    
cols=len(lines[0])-1
print(rows,cols)
free=0
wall=0
matrix = np.zeros((rows,cols),dtype=int)  
prob_matrix = np.zeros((rows,cols),dtype=float)             
for i in range(rows):                                          # Converting the text file into a 2d matrix where 0 represents a path while a 1 represents a wall
    for j in range(cols):
        if(lines[i][j]=='_'):
            matrix[i][j]=0
            free=free+1 
        else:
            matrix[i][j]=1 
            wall=wall+1 
            
for i in range(rows):
    for j in range(cols):
        if(matrix[i][j]==0):             
            prob_matrix[i][j]=(1/free)                         # Defining the initial Probabilities for the grid

print(matrix)
plt.imshow(prob_matrix,cmap='Reds')
plt.show()
Result_From_Search = Search_For_Drone(matrix,prob_matrix)      # Moving the drone until we finalize it's location 

print("number_of_moves= " ,Result_From_Search[0])  
print("Sequence of Moves= ",Result_From_Search[1])  
for i in range(rows):
    for j in range(cols):
        if(prob_matrix[i][j]!=0):
            drone_position=[i,j]
            matrix[i][j]=5
            break
print("Final position of drone= ",drone_position)   
print(matrix)
plt.imshow(matrix,cmap='Reds')
plt.show()       

    



  


