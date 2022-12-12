import numpy as np
import sys
from collections import deque
np.set_printoptions(threshold=sys.maxsize, linewidth=3000)


# go 1 step at a time. from S look for all possible moves (only 0s and 1s)
def hike_this_b(grid, starting_position, ending_position):
    steps = -1 # so that on our first run in the while loop we start at 0
    top_found = False
    pathway_grid = build_pathway_grid(grid, starting_position)
    current_steps = deque([starting_position])

    while not top_found:
        steps += 1
        for i in range(len(current_steps)):
            position = current_steps.popleft()
            if position == ending_position:
                print(f'FOUND IT in {steps} steps')
                top_found = True
                break
            else:
                current_steps.extend(look_around(position, grid, pathway_grid, move_up))


# this time instead of looking for the starting position by location, we want to look for any location where the
# ending height is 0, and we want the starting position to be the mountaintop
def hike_back_down(grid, starting_position, ending_height):
    steps = -1 # so that on our first run in the while loop we start at 0
    scenin_route_found = False
    pathway_grid = build_pathway_grid(grid, starting_position)
    current_steps = deque([starting_position])

    while not scenin_route_found:
        steps += 1
        for i in range(len(current_steps)):
            position = current_steps.popleft()
            if grid[position] == ending_height:
                print(f'FOUND IT in {steps} steps')
                scenin_route_found = True
                break
            else:
                current_steps.extend(look_around(position, grid, pathway_grid, move_down))


# the pathway grid keeps track of where we've already been before, so we don't end up retracing our steps
def build_pathway_grid(grid, starting_position):
    pathway_grid = np.zeros_like(grid, int)
    pathway_grid[starting_position] = 1
    return pathway_grid

def look_around(current_position, grid, pathway_grid, move_function):
    current_y, current_x = current_position
    next_steps = []

    # start moving
    if current_x != 0:
        move_function([0, -1], current_position, pathway_grid, grid, next_steps)
    if current_x != grid.shape[1] - 1:
        move_function([0, 1], current_position, pathway_grid, grid, next_steps)
    if current_y != 0:
        move_function([-1, 0], current_position, pathway_grid, grid, next_steps)
    if current_y != grid.shape[0] - 1:
        move_function([1, 0], current_position, pathway_grid, grid, next_steps)
    return next_steps


def move_up(direction, current_position, pathway_grid, grid, next_stops):
    move(direction, current_position, pathway_grid, grid, next_stops, 'up')


def move_down(direction, current_position, pathway_grid, grid, next_stops):
    move(direction, current_position, pathway_grid, grid, next_stops, 'down')

def move(direction, current_position, pathway_grid, grid, next_stops, hike_direction):
    current_y, current_x = current_position
    next_step = (current_y + direction[0], current_x + direction[1])
    if hike_direction == 'down':
        height_difference = grid[current_position] - grid[next_step]
    else:# if hike_direction == 'up':
        height_difference = grid[next_step] - grid[current_position]
    if pathway_grid[next_step] != 1 and height_difference <= 1:
        next_stops.append(next_step)
        pathway_grid[next_step] = 1


# this part is used in parsing the data to find the locations of the start and end values

def find_grid_position(npgrid, value):
    for gridy, gridx in np.ndindex(npgrid.shape):
        if npgrid[gridy, gridx] == value:
            return (gridy, gridx)


def find_starting_grid_position(npgrid):
    return find_grid_position(npgrid, -14)


def find_ending_grid_position(npgrid):
    return find_grid_position(npgrid, -28)


# parsing the data to set up 2D array
mountain_grid = np.array([list(line.strip()) for line in open('input.txt')], str)
height_grid = np.zeros_like(mountain_grid, int)
for y, x in np.ndindex(mountain_grid.shape):
    height_grid[y, x] = ord(mountain_grid[y, x]) - 97
starting_location = find_starting_grid_position(height_grid)
height_grid[starting_location] = 0 # need to set to 0 bc ascii -97 if S is -14
mountaintop = find_ending_grid_position(height_grid)
mountaintop_value = 25 # need to set to 25 bc ascii -97 if S is -28
height_grid[mountaintop] = mountaintop_value
hike_this_b(height_grid, starting_location, mountaintop)
hike_back_down(height_grid, mountaintop, 0)
