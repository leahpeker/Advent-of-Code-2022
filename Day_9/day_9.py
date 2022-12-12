import numpy as np
import re


class Node:

    def __init__(self, starting_position):
        self.x, self.y = starting_position


class Grid:

    def __init__(self, lines):
        self.lines = lines
        self.grid, self.starting_x, self.starting_y, self.move_matrix = self.build_grid()
        self.grid[self.starting_y, self.starting_x] = 1

    def get_grid_dims(self):
        walking_x, walking_y = [0, 0]
        left_max, right_max, up_max, down_max = [0] * 4
        direction_dict = {
            'L': [-1, 0],
            'R': [1, 0],
            'U': [0, 1],
            'D': [0, -1]
        }

        move_matrix = []

        for line in self.lines:
            step_size = int(re.findall('\d+', line)[0])
            walking_x += direction_dict[line[0]][0] * step_size
            walking_y += direction_dict[line[0]][1] * step_size
            if walking_x < left_max:
                left_max = walking_x
            if walking_x > right_max:
                right_max = walking_x
            if walking_y > up_max:
                up_max = walking_y
            if walking_y < down_max:
                down_max = walking_y
            move_matrix.append([direction_dict[line[0]][0] * step_size, direction_dict[line[0]][1] * step_size])
        x_dim = right_max - left_max + 1
        y_dim = up_max - down_max + 1
        starting_x, starting_y = -left_max, -down_max

        return starting_x, starting_y, x_dim, y_dim, move_matrix

    def build_grid(self):
        starting_x, starting_y, x_dim, y_dim, move_matrix = self.get_grid_dims()
        grid = np.zeros(shape=(y_dim, x_dim), dtype=int)
        return grid, starting_x, starting_y, move_matrix

    def walk_along_grid_1(self):
        head = Node([self.starting_x, self.starting_y])
        tail = Node([self.starting_x, self.starting_y])

        for step in self.move_matrix:
            horizontal = step[0]
            vertical = step[1]
            direction = self.get_direction(horizontal, vertical)
            for i in range(max(abs(horizontal), abs(vertical))):
                if horizontal == 0:
                    head.y += direction
                if vertical == 0:
                    head.x += direction
                tail.y, tail.x = self.follow_the_leader(head, tail, direction)
                self.grid[tail.y, tail.x] = 1

        return self.count_spots()

    def walk_along_grid_2(self):
        node_dict = {}
        knots = 10
        for i in range(knots):
            node_dict[i] = Node(([self.starting_x, self.starting_y]))

        for step in self.move_matrix:
            print(step)
            horizontal = step[0]
            vertical = step[1]
            direction = self.get_direction(horizontal, vertical)
            for i in range(max(abs(horizontal), abs(vertical))):
                if horizontal == 0:
                    node_dict[0].y += direction
                if vertical == 0:
                    node_dict[0].x += direction
                for key, node in node_dict.items():
                    if key != 0:

                        node.y, node.x = self.follow_the_leader(node_dict[key - 1], node, direction)
                self.grid[node_dict[knots - 1].y, node_dict[knots - 1].x] = 1

        return self.count_spots()

    def count_spots(self):
        unique, counts = np.unique(self.grid, return_counts=True)
        count_dict = dict(zip(unique, counts))
        return count_dict[1]

    def follow_the_leader(self, previous, current, direction):
        if abs(previous.x - current.x) > 1 and previous.y != current.y or abs(previous.y - current.y) > 1 and previous.x != current.x:
            if previous.x < current.x:
                current.x -= 1
            else:
                current.x += 1
            if previous.y < current.y:
                current.y -= 1
            else:
                current.y += 1
        elif abs(previous.x - current.x) > 1:
            if previous.x < current.x:
                current.x -= 1
            else:
                current.x += 1
        elif abs(previous.y - current.y) > 1:
            if previous.y < current.y:
                current.y -= 1
            else:
                current.y += 1

        return current.y, current.x

    def get_direction(self, horizontal, vertical):
        if horizontal > 0 or vertical > 0:
            direction = 1
        else:
            direction = -1
        return direction


with open('input.txt') as file:
    lines = [lines.strip() for lines in file.readlines()]

grid = Grid(lines)
# part_1 = grid.walk_along_grid_1()
# print(part_1)
part_2 = grid.walk_along_grid_2()
print('part2:', part_2)

