#%%
import numpy as np

#%%
def index_from_xy(x, y, nX, nY):
    return x+y*nX;

def define_environment_map(n_rows=5, n_cols=5, inner_wall_coords=[[1,2],[2,2],[2,3],[2,4]], 
                 startX=3, startY=4, goalX=1, goalY=3):
    bool_2D_np_array = np.zeros((n_rows, n_cols),dtype=bool)
    for coord in inner_wall_coords:
        bool_2D_np_array[coord[0]][coord[1]] = True
    
    nX, nY = bool_2D_np_array.shape
    inner_wall = []
    up_forbidden = []
    down_forbidden = []
    left_forbidden = []
    right_forbidden = []
    for x in range(nX):
        for y in range(nY):
            if bool_2D_np_array[x,y] == True:
                inner_wall.append(index_from_xy(x, y, nX, nY))
                if x > 0 and bool_2D_np_array[x-1, y] == False:
                    right_forbidden.append(index_from_xy(x-1, y, nX, nY))
                if x < nX-1 and bool_2D_np_array[x+1, y] == False:
                    left_forbidden.append(index_from_xy(x+1, y, nX, nY))
                if y > 0 and bool_2D_np_array[x, y-1] == False:  
                    down_forbidden.append(index_from_xy(x, y-1, nX, nY))
                if y < nY-1 and bool_2D_np_array[x,y+1] == False:   
                    up_forbidden.append(index_from_xy(x, y+1, nX, nY))
    startIndex = index_from_xy(startY, startX, nX, nY)
    endIndex = index_from_xy(goalY, goalX, nX, nY)

    return inner_wall, up_forbidden, down_forbidden, left_forbidden, right_forbidden, startIndex, endIndex

#%%
if __name__ == '__main__':
    inner_wall, up_forbidden, down_forbidden, left_forbidden, right_forbidden, startIndex, endIndex = define_environment_map()