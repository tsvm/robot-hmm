"""
Plays the robot movement game.
"""

import random
import numpy as np
from robot import Robot
import tkinter as tk
import time
from grid import cell_observed_state
from hmm import HMM

CANVAS_SIZE = 500
GRID_SIZE = 20
CELL_SIZE = int(CANVAS_SIZE / GRID_SIZE)

COL_BACKGROUND = "black"
COL_ROBOT = "orange"
COL_ROBOT_PROB_STATE = "yellow"
COL_OBSTACLE = "red"
COL_LINES = "grey"
COL_PATH = "yellow"
COL_ALLOWED_STATE="white"

RUN_STEPS=100


# Initialize the visual grid
root = tk.Tk()
root.title("Robot")
canvas = tk.Canvas(root, height=CANVAS_SIZE, width=CANVAS_SIZE, bg=COL_BACKGROUND)
canvas.pack(fill=tk.BOTH, expand=True)



###### BEGIN Setup of the visual grid ######
# Function for drawing the lines of the grid in which the robot will be visualized.
def draw_canvas_lines(event=None):
    w = canvas.winfo_width() # Get current width of canvas
    h = canvas.winfo_height() # Get current height of canvas
    canvas.delete('grid_line') # Will only remove the grid_line

    # Creates all vertical lines at intevals of CELL_SIZE
    for i in range(0, w, CELL_SIZE):
        canvas.create_line([(i, 0), (i, h)], tag='grid_line', fill=COL_LINES)

    # Creates all horizontal lines at intevals of CELL_SIZE
    for i in range(0, h, CELL_SIZE):
        canvas.create_line([(0, i), (w, i)], tag='grid_line', fill=COL_LINES)

canvas.bind('<Configure>', draw_canvas_lines)
###### END Setup of the visual grid ######



def main():
    # Get the initial configuration of the robot grid, including obstacles and the initial random position for the robot.
    grid, initial_position = init()
    # Init the robot
    robot = Robot(grid, initial_position)
    # Simulate random movement for a given number of steps
    simulate_movement(robot, RUN_STEPS)


def init():
    """
    Initialize the environment.
    Create a grid with predefined obstacles.
    Choose a random initial position for the robot.
    """
    # Init an empty grid.
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    print('Empty grid:\n', grid)

    # Define obstacles = set 1 in the corresponding cells.
    # obstacles = static_obstacles_2()
    obstacles = random_obstacles(50)
    # Set 1 for an obstacle and show it in the visible grid.
    for cell in obstacles:
        grid[cell] = 1
        fill_cell(canvas, cell[0], cell[1], fill=COL_OBSTACLE)
    print('Grid with obstacles:\n', grid)

    # Get a random position to put the robot on.
    initial_position = get_initial_position(obstacles)

    return grid, initial_position


def static_obstacles_1():
    return [(0, 0), (0, 19), (1,1), (1,2), (1,3), (1,4), (1,5), \
            (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), \
            (6, 10), (7, 10), (8, 10), (9, 10), (10,10), \
            (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), \
            (12, 15), (13, 15), (14, 15), (15, 15), (15, 16), (15, 17), (15, 18)]


def static_obstacles_2():
    # Obstacles on the rows, the first value is the row, the next two are the columns from-to which to draw a line
    row_obstacles = [(1, 1, 5), (14, 3, 12), (15, 15, 18), (3, 8, 12)]
    # Obstacles on the cols, the first value is the col, the next two are the rows from-to which to draw a line
    col_obstacles = [(2, 4, 10), (6, 6, 10), (10, 7, 11), (2, 16, 18), (6, 16, 18), (13, 6, 10), (17, 1, 12)]
    obstacles = []
    for l in row_obstacles:
        for i in range(l[1], l[2]+1):
            obstacles.append((l[0], i))
    for l in col_obstacles:
        for i in range(l[1], l[2]):
            obstacles.append((i, l[0]))
    return obstacles


def static_obstacles_3():
    # Obstacles on the rows, the first value is the row, the next two are the columns from-to which to draw a line
    row_obstacles = [(1, 1, 5), (14, 3, 12), (15, 15, 18), (3, 7, 12), (11, 5, 7), (2, 15, 17), (3, 15, 17)]
    # Obstacles on the cols, the first value is the col, the next two are the rows from-to which to draw a line
    col_obstacles = [(2, 4, 10), (6, 3, 10), (10, 7, 11), (15, 11, 15), (17, 6, 12),\
                     (2, 16, 18), (6, 16, 18), (10, 16, 18), (13, 6, 10),\
                     (4, 17, 19), (8, 17, 19), (12, 17, 19), (16, 17, 19),\
                     (4, 6, 12), (8, 6, 12)]
    obstacles = []
    for l in row_obstacles:
        for i in range(l[1], l[2]+1):
            obstacles.append((l[0], i))
    for l in col_obstacles:
        for i in range(l[1], l[2]):
            obstacles.append((i, l[0]))
    return obstacles


def random_obstacles(number):
    obstacles = []
    for i in range(number):
        obstacles.append((random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)))
    return obstacles


def get_initial_position(obstacles):
    while True:
        initial_position = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if not initial_position in obstacles:
            break
    print('Initial position:', initial_position)
    return initial_position


def simulate_movement(robot, timestep):
    hmm = HMM(robot.grid)
    history_hidden_states = []
    history_observed_states = []
    """
    Simulate the robot movement for a given number of timesteps.
    """
    history_hidden_states.append(robot.current_position)
    observed_state, _ = cell_observed_state(robot.grid, robot.current_position[0], robot.current_position[1])
    history_observed_states.append(str(observed_state))

    # Show the robot in the new position
    fill_cell(canvas, robot.current_position[0], robot.current_position[1])
     # Show the possible robot moves in the grid
    possible_moves = hmm.calculate_forward_probabilities(history_observed_states, history_hidden_states[0])
    old_moves = show_probable_moves(robot, possible_moves)

    for i in range(timestep):      
        time.sleep(1)

        pos_old = robot.current_position
        robot.step()
        pos_new = robot.current_position
        # Save the history of movement for caluclation on the HMM
        history_hidden_states.append(pos_new)
        observed_state, _ = cell_observed_state(robot.grid, pos_new[0], pos_new[1])
        history_observed_states.append(str(observed_state))

        print(printable_time(), 'Moved to a new position', robot.current_position)
        print(history_observed_states)
        print(history_hidden_states)
       
        # Show the robot on its new position
        fill_cell(canvas, pos_old[0], pos_old[1], fill=COL_BACKGROUND)
        fill_cell(canvas, robot.current_position[0], robot.current_position[1], fill=COL_ROBOT)

        # time.sleep(1)
        root.update()

        # Reset the old moves
        reset_old_moves(old_moves)  
        print(printable_time(), 'Reset old moves.')

        # Show the possible robot moves in the grid
        possible_moves = hmm.calculate_forward_probabilities(history_observed_states, history_hidden_states[0])
        old_moves = show_probable_moves(robot, possible_moves)
        # old_moves = show_possible_moves(robot)
        print(printable_time(), 'Show new moves.')
        time.sleep(1)
        root.update()

        # show the robot cell again
        if robot.current_position in possible_moves.keys():
            robot_fill = COL_ROBOT_PROB_STATE
        else:
            robot_fill = COL_ROBOT
        fill_cell(canvas, robot.current_position[0], robot.current_position[1], fill=robot_fill)
        
        root.update()


##### Methods for displaying positions in the grid. #####
def fill_cell(canvas, col, row, fill=COL_ROBOT):
    cell = canvas.create_rectangle(row*CELL_SIZE, col*CELL_SIZE, (row+1)*CELL_SIZE, (col+1)*CELL_SIZE, fill=fill, outline=COL_LINES)
    return cell


def reset_old_moves(moves):
    if moves:
        for move in moves:
            fill_cell(canvas, move[0], move[1], fill=COL_BACKGROUND)


def show_probable_moves(robot, state_probabilities):
    probable_moves = []
    for k, v in state_probabilities.items():
        fill_col = 100+int(v)
        if fill_col > 100: fill_col = 99
        if fill_col < 1: fill_col = 1
        color = "gray{}".format(fill_col)
        probable_moves.append(k)
        fill_cell(canvas, k[0], k[1], fill=color)
    return probable_moves


def show_possible_moves(robot):
    # fill the possible positions
    observed_state, allowed_moves = cell_observed_state(robot.grid, robot.current_position[0], robot.current_position[1])
    print('observed_state:', observed_state)
    for move in allowed_moves:
        # Show the admissible states
        fill_cell(canvas, move[0], move[1], fill=COL_ALLOWED_STATE)
    return allowed_moves

##### Methods for displaying positions in the grid. #####



def printable_time():
    return time.strftime("%row %X", time.gmtime())


if __name__ == '__main__':
    main()


root.mainloop()




