from collections import defaultdict


class ElfTree:

    def __init__(self, height, grid: list):
        self.height = height
        self.x, self.y = grid
        self.visibility = False
        self.line_of_sight_up = 0
        self.line_of_sight_down = 0
        self.line_of_sight_left = 0
        self.line_of_sight_right = 0
        self.scenic_score = 0

    def cal_scenic_score(self):
        self.scenic_score = self.line_of_sight_up * self.line_of_sight_left * self.line_of_sight_right * self. line_of_sight_down
        return self.scenic_score


def calculate_visible_trees(lines, dictionary):
    sum_of_visible_trees = 0
    dictionary = dictionary
    length = len(lines[0])
    for i, line in enumerate(lines):
        dictionary = step_tree_matrix(dictionary, length, i, lines)
        dictionary[1] = []
        for j, height in enumerate(line):
            tree = ElfTree(int(height), [j, i])
            dictionary = create_tree_matrix(tree, dictionary, 1)
        dictionary = count_visible_trees(dictionary, length)
        for i, tree in enumerate(dictionary[1]):
            if tree.visibility:
                sum_of_visible_trees += 1

    return sum_of_visible_trees


def create_tree_matrix(tree, dictionary, index):
    dictionary[index].append(tree)
    return dictionary


def step_tree_matrix(dictionary, length, index, lines):
    for i in range(length):
        if not dictionary[0][i].height > dictionary[1][i].height:
            dictionary[0][i] = dictionary[1][i]
    bottom_max_height_list = [-1] * length
    for line in lines[:index:-1]:
        for i, tree_height in enumerate(line):
            if int(tree_height) > bottom_max_height_list[i]:
                bottom_max_height_list[i] = int(tree_height)
    dictionary[2] = []
    for j, height in enumerate(bottom_max_height_list):
        tree = ElfTree(int(height), [j, index])
        create_tree_matrix(tree, dictionary, 2)
    return dictionary


def count_visible_trees(dictionary, length):
    left_max = -1
    right_max = -1
    for i in range(length):
        if dictionary[1][i].height > left_max: # left
            left_max = dictionary[1][i].height
            dictionary[1][i].visibility = True
            continue
        if dictionary[1][i].height > dictionary[0][i].height: # top
            dictionary[1][i].visibility = True
            continue
        if dictionary[1][i].height > dictionary[2][i].height: # bottom
            dictionary[1][i].visibility = True
            continue
    for i in range(length - 1, -1, -1):
        if dictionary[1][i].height > right_max:  # right
            right_max = dictionary[1][i].height
            dictionary[1][i].visibility = True
            continue

    return dictionary


def build_giant_matrix(lines):
    dictionary = defaultdict(list)
    for i, line in enumerate(lines):
        for j, height in enumerate(line):
            tree = ElfTree(int(height), [j, i])
            dictionary = create_tree_matrix(tree, dictionary, i)
    return dictionary


def do_stuff(lines):
    length = len(lines)
    line_length = len(lines[0])
    dictionary = build_giant_matrix(lines)
    max_scenic_score = 0
    for index in range(length):
        dictionary, scenic_score = look_around(dictionary, index, length, line_length)
        if scenic_score > max_scenic_score:
            max_scenic_score = scenic_score
    return max_scenic_score


def look_around(dictionary, index, length, line_length):
    max_scenic_score = 0
    if index == 0 or index == length - 1:
        return dictionary, max_scenic_score
    for i, tree in enumerate(dictionary[index]):
        # first look up
        starting_index = index
        while starting_index > 0:
            if tree.height > dictionary[starting_index - 1][i].height:
                starting_index -= 1
            else:
                starting_index -= 1
                break
        tree.line_of_sight_up = index - starting_index

        #then look down
        starting_index = index
        while starting_index < length - 1:
            if tree.height > dictionary[starting_index + 1][i].height:
                starting_index += 1
            else:
                starting_index += 1
                break
        tree.line_of_sight_down = starting_index - index

        # then look right:
        starting_index = i
        while starting_index < line_length - 1:
            if tree.height > dictionary[index][starting_index + 1].height:
                starting_index += 1
            else:
                starting_index += 1
                break
        tree.line_of_sight_right = starting_index - i

        #looking left
        starting_index = i
        while starting_index > 0:
            if tree.height > dictionary[index][starting_index - 1].height:
                starting_index -= 1
            else:
                starting_index -= 1
                break
        tree.line_of_sight_left = i - starting_index

        scenic_score = tree.cal_scenic_score()
        if scenic_score > max_scenic_score:
            max_scenic_score = scenic_score

    return dictionary, max_scenic_score


filename = 'test.txt'
filename = 'input.txt'
with open(filename) as file:
    lines = [x.strip('\n') for x in file.readlines()]


tree_list = []
for i in range(len(lines[0])):
    tree_list.append(ElfTree(-1, [int(i), 0]))
tree_matrix = defaultdict(lambda: tree_list)
calculate_visible_trees(lines, tree_matrix)
do_stuff(lines)
