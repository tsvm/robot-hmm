import numpy as np

MOVE_N, MOVE_E, MOVE_S, MOVE_W, KEEP_POS = 0, 1, 2, 3, 4


"""
Contains helper functions for the grid.
"""

def cell_allowed(grid, row, col):
    # not an obstacle 
    # row is in grid
    # col is in grid
    return  row < grid.shape[0] and row >= 0 \
        and col < grid.shape[1] and col >= 0 \
        and grid[row, col] == 0 


def cell_observed_state(grid, row, col):
    """
    Get the string describing the observed state of the cell.
    It consists of 4 positions.
    Each position tells us whether the north, east, south or west cell has an obstacle or grid end.
    0 means you can move in that position.
    1 means the position is forbidden.
    """
    state = np.ones(4)
    allowed_cells = []
    # row, col = cell
    if cell_allowed(grid, row-1, col): 
        state[MOVE_N] = 0
        allowed_cells.append((row-1, col))
    if cell_allowed(grid, row+1, col): 
        state[MOVE_S] = 0
        allowed_cells.append((row+1, col))
    if cell_allowed(grid, row, col+1): 
        state[MOVE_E] = 0
        allowed_cells.append((row, col+1))
    if cell_allowed(grid, row, col-1): 
        state[MOVE_W] = 0
        allowed_cells.append((row, col-1))

    return state, allowed_cells
