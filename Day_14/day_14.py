import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=3000)


class SandDrop:
    def __init__(self, starting_x, cave_height):
        self.starting_x = starting_x
        self.max_height = cave_height
        self.x = self.starting_x
        self.y = 0
        self.marker = 'o'

    # drops the sand and returns True if we're stuck (could probs break apart into 2 methods)
    def are_we_stuck(self, grid):
        cave_space = '#'
        options = self.explore_options()
        stuck = False
        for option_y, option_x in options:
            if grid[option_y, option_x] == cave_space or grid[option_y, option_x] == self.marker:
                continue
            else:
                self.y, self.x = option_y, option_x
                return stuck # stuck will be false here
        stuck = True # we only get here if we passed all 3 options and there was no viable option to move to
        self.land_the_sand(grid)
        return stuck

    def explore_options(self):
        return [[self.y + 1, self.x], [self.y + 1, self.x - 1], [self.y + 1, self.x + 1]]

    def is_sand_flowing(self, grid):
        options = self.explore_options()
        overflowed_list = []
        for option in options:
            option_y, option_x = option
            if option_y == self.max_height:
                overflowed_list.append(True)
            else:
                overflowed_list.append(False)
        overflowed = all(overflowed_list)
        if overflowed:
            grid[options[0][0], options[0][1]] = 'X'
        return all(overflowed_list)

    def is_top_reached(self):
        top_reached = False
        if self.x == self.starting_x and self.y == 0:
            top_reached = True
        return top_reached

    def land_the_sand(self, grid):
        grid[self.y, self.x] = self.marker
        self.reset()

    def reset(self):
        self.x = self.starting_x
        self.y = 0

# parse it and set everything to ints right from the get go
def part_1_parsing(input_lines):
    parsed_lines = []
    x_set = set()
    y_set = set()
    for line in lines:
        new_line = []
        for item in line:
            new_item = []
            for i, coord in enumerate(item.split(',')):
                coord = int(coord)
                new_item.append(coord)
                if i == 0:
                    x_set.add(coord)
                else:
                    y_set.add(coord)
            new_line.append(new_item)
        parsed_lines.append(new_line)
    x_dim = max(x_set) - min(x_set) + 2
    y_dim = max(y_set) + 1
    return parsed_lines, x_dim, min(x_set), y_dim


# set up the cave layout
def build_cave(x_dim, y_dim):
    grid = np.empty((y_dim, x_dim), dtype=str)
    for x, y in np.ndindex(grid.shape):
        grid[x, y] = '.'
    return grid


# this actually builds out our cave based on input
def mark_cave_setup(input_line):
    starting_x, starting_y = input_line[0]
    list_of_points = input_line
    for item_x, item_y in input_line[1:]:
        if item_x == starting_x:
            y_diff = starting_y - item_y
            if y_diff > 0:
                for y_step in range(item_y + 1, starting_y):
                    list_of_points.append([starting_x, y_step])
            else:
                for y_step in range(starting_y + 1, item_y):
                    list_of_points.append([starting_x, y_step])
        if item_y == starting_y:
            x_diff = starting_x - item_x
            if x_diff > 0:
                for x_step in range(item_x + 1, starting_x):
                    list_of_points.append([x_step, starting_y])
            else:
                for x_step in range(starting_x + 1, item_x):
                    list_of_points.append([x_step, starting_y])
        starting_x = item_x
        starting_y = item_y
    return list_of_points


def mark_cave_layout(input_line, grid, x_min):
    list_of_points = mark_cave_setup(input_line)
    for x, y in list_of_points:
        grid[y, x - x_min + 1] = '#' # add 1 here so that the grid will have a blank x on either side on the base


# this is where we do the work for part 1
def when_do_we_overflow(input_lines):
    parsed_lines, x_axis, lowest_x, y_axis = part_1_parsing(input_lines)
    cave = build_cave(x_axis + 1, y_axis + 1)
    sand_drop_x = 500
    sand_drop_x_index = sand_drop_x - lowest_x + 1
    for line in parsed_lines:
        mark_cave_layout(line, cave, lowest_x)
    return cycle_through(sand_drop_x_index, y_axis, cave)

def cycle_through(sand_drop_x_index, y_axis, cave):
    sand_drop = SandDrop(sand_drop_x_index, y_axis)
    cycle = -1
    overflowed = False
    while not overflowed:
        nowhere_to_go = False
        while not nowhere_to_go:
            overflowed = sand_drop.is_sand_flowing(cave)
            if overflowed:
                break
            nowhere_to_go = sand_drop.are_we_stuck(cave)
        cycle += 1
    print('cycle', cycle)
    return cycle


def cycle_through_without_overflow(sand_drop_x_index, y_axis, cave):
    sand_drop = SandDrop(sand_drop_x_index, y_axis)
    cycle = 0
    reached_the_top = False
    while not reached_the_top:
        nowhere_to_go = False
        while not nowhere_to_go:
            at_the_top = sand_drop.is_top_reached()
            nowhere_to_go = sand_drop.are_we_stuck(cave)
            if nowhere_to_go and at_the_top:
                reached_the_top = True
        cycle += 1
    print('cycle', cycle)
    return cycle


def expand_cave(input_lines):
    parsed_lines, x_axis, lowest_x, y_axis = part_1_parsing(input_lines)
    x_axis *= 8
    y_axis += 1
    lowest_x -= x_axis // 2
    cave = build_cave(x_axis + 1, y_axis + 1)
    sand_drop_x = 500
    sand_drop_x_index = sand_drop_x - lowest_x + 1
    for line in parsed_lines:
        mark_cave_layout(line, cave, lowest_x)
    for x in range(x_axis + 1):
        cave[y_axis, x] = '#'
    return sand_drop_x_index, y_axis, cave


#this one does the work for part 2
def make_a_sand_river(input_lines):
    sand_drop_x_index, y_axis, cave = expand_cave(input_lines)
    cycle_through_without_overflow(sand_drop_x_index, y_axis, cave)


with open('test.txt') as file:
    lines = [line.strip().split(' -> ') for line in file.readlines()]

part_1 = when_do_we_overflow(lines)
part_2 = make_a_sand_river(lines)
