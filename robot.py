
"""
"""
import random
import math
import numpy as np
from grid import cell_allowed, cell_observed_state
from grid import MOVE_N, MOVE_E, MOVE_S, MOVE_W, KEEP_POS

LOG = False

class Robot():
    def __init__(self, grid, initial_position):
        self.admissible_positions = np.zeros(5) # allowed positions (0 or 1)
        self.possible_moves = np.zeros(5)       # probability of the allowed moves
        self.observed_obstacles = np.zeros(4)            # obstacles in the neighbouring cells
        self.grid = grid
        self.current_position = initial_position # row, col of the current possition
        self.calc_admissible_positions()


    def step(self):
        """
        Perform one step.
        Calculate the next position based on the current position and possible moves.
        """
        if LOG: print('admissible_positions:', self.admissible_positions, np.nonzero(self.admissible_positions)[0])

        # Select the next move randomly from the admissible positions
        next_move = np.random.choice(np.nonzero(self.admissible_positions)[0])
        if LOG: print('nm:',next_move)

        row, col = self.current_position
        if next_move == MOVE_N:   row -= 1
        elif next_move == MOVE_S: row += 1
        elif next_move == MOVE_E: col += 1
        elif next_move == MOVE_W: col -= 1
        self.current_position = (row, col)
        self.calc_admissible_positions()


    def calc_admissible_positions(self):
        """
        Determine the admissible moves according to the current position,
        grid boundaries and the obstacles in the grid.
        Calculates the probabilities of the next moves based on the admissible positions.
        """
        row, col = self.current_position
        obstacles, _ = cell_observed_state(self.grid, row, col)
        admissible_positions = np.ones(5)
        for i in range(4):
            admissible_positions[i] = abs(1-obstacles[i])

        self.admissible_positions = admissible_positions
        self.observed_obstacles = obstacles
        self.possible_moves = self.admissible_positions * (1/np.count_nonzero(self.admissible_positions))

        if LOG: print(self.admissible_positions)
        if LOG: print(self.possible_moves)


    def allowed(self, row, col):
        # not an obstacle 
        # row is in grid
        # col is in grid
        return cell_allowed(self.grid, row, col)









