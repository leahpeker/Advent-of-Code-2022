import ast
from collections import deque


# Node class is used in part 2 to build the tree. It's important to store the whole list so we can cycle through it
class Node:
    def __init__(self, input_list, id):
        self.input_list = input_list
        self.value = self.initialize_value()
        self.current_index = 0
        self.left = None
        self.right = None
        self.id = id


    def initialize_value(self):
        if len(self.input_list) == 0:
            starting_value = None
        else:
            starting_value = self.input_list[0]
        return starting_value
    def next(self):
        if self.current_index + 1 >= len(self.input_list):
            self.value = None
        else:
            self.current_index += 1
            self.value = self.input_list[self.current_index]
        return self

    # we change the values to search through a list, but when we compare it to the next node, we want to start back at
    # index 0
    def reset(self):
        self.value = self.initialize_value()
        self.current_index = 0

# for part 1, compare the sets to determine if they're in the right order
def compare(object_l, object_r):
    for i in range(max(len(object_l), len(object_r))):
        if i >= len(object_l):
            return True
        if i >= len(object_r):
            return False
        if type(object_l[i]) == int and type(object_r[i]) == int:
            if object_l[i] > object_r[i]:
                return False
            elif object_l[i] < object_r[i]:
                return True
        elif type(object_l[i]) == int or type(object_r[i]) == int:
            if type(object_l[i]) == int:
                object_l[i] = [object_l[i]]
            else:
                object_r[i] = [object_r[i]]
            comparison = compare(object_l[i], object_r[i])
            if comparison != None:
                return comparison
        else:
            comparison = compare(object_l[i], object_r[i])
            if comparison != None:
                return comparison


# for part 1, when the sets are in the right order, we add their inices to our total
def find_the_sum_of_passed_indices(input_lines):
    index = 1
    sum_of_indices = 0
    for i in range(0, len(input_lines), 3):
        left = ast.literal_eval(input_lines[i])
        right = ast.literal_eval(input_lines[i + 1])
        if compare(left, right):
            sum_of_indices += index
        index += 1
    print(sum_of_indices)
    return sum_of_indices


# for part 2 - parses and flattens the lists, and adds in the [[2]] and [[6]] decoder keys
def parse_lines_for_decoder(input_lines):
    key_insert_1 = [2]
    key_insert_2 = [6]
    parsed_lines = [key_insert_1, key_insert_2]
    for i in range(0, len(input_lines), 3):
        new_line = []
        flatten(ast.literal_eval(input_lines[i].replace('[]', '[0]')), new_line)
        parsed_lines.append(new_line)
        if new_line == [2] or new_line == [6]:
            print('2 found', i, new_line, input_lines[i])
        new_line = []
        flatten(ast.literal_eval(input_lines[i + 1].replace('[]', '[0]')), new_line)
        parsed_lines.append(new_line)
        if new_line == [2] or new_line == [6]:
            print('2 found', i, new_line, input_lines[i + 1])
    return parsed_lines


def flatten(line, new_line):
    for i in range(len(line)):
        if type(line[i]) == int:
            new_line.append(line[i])
        else:
            flatten(line[i], new_line)


# for part 2, we want to turn each list into a node class and add it to a queue for later use
def build_deque(input_lines):
    node_dequer = deque()
    parsed_lines = parse_lines_for_decoder(input_lines)
    for index, line in enumerate(parsed_lines):
        node = Node(line, index)
        node_dequer.append(node)
    return node_dequer


def get_starting_node(input_deque):
    return input_deque.pop()


# this one does msot of the work. takes items from the queue one at a time and positions them on the
# binary search tree
def build_tree(starting_node, input_deque, top_node_id, current_node=None):
    cycle = 0
    while cycle < top_node_id:
        if not current_node:
            if input_deque:
                current_node = input_deque.pop()
            else:
                return
        while current_node.value and starting_node.value and current_node.value == starting_node.value:
            current_node.next()
            starting_node.next()
        if current_node.value == starting_node.value:
            current_node.reset()
            starting_node.reset()
            if starting_node.left:
                build_tree(starting_node.left, input_deque, top_node_id, current_node)
                if starting_node.id != top_node_id:
                    return
            else:
                current_node.reset()
                starting_node.left = current_node
                if starting_node.id != top_node_id:
                    return
        elif current_node.value == None:
            current_node.reset()
            starting_node.reset()
            if starting_node.left:
                build_tree(starting_node.left, input_deque, top_node_id, current_node)
                if starting_node.id != top_node_id:
                    return
            else:
                current_node.reset()
                starting_node.left = current_node
                if starting_node.id != top_node_id:
                    return
        elif starting_node.value == None:
            current_node.reset()
            starting_node.reset()
            if starting_node.right:
                build_tree(starting_node.right, input_deque, top_node_id, current_node)
                if starting_node.id != top_node_id:
                    return
            else:
                current_node.reset()
                starting_node.right = current_node
                if starting_node.id != top_node_id:
                    return
        elif current_node.value > starting_node.value:
            current_node.reset()
            starting_node.reset()
            if starting_node.right:
                build_tree(starting_node.right, input_deque, top_node_id, current_node)
                if starting_node.id != top_node_id:
                    return
            else:
                current_node.reset()
                starting_node.right = current_node
                if starting_node.id != top_node_id:
                    return
        elif current_node.value < starting_node.value:
            current_node.reset()
            starting_node.reset()
            if starting_node.left:
                build_tree(starting_node.left, input_deque, top_node_id, current_node)
                if starting_node.id != top_node_id:
                    return
            else:
                current_node.reset()
                starting_node.left = current_node
                if starting_node.id != top_node_id:
                    return
        current_node = None

        cycle += 1


# for part 2, here we actually search through our tree to return the items in order
def depth_search_the_tree(treetop, input_list):
    if treetop:
        if treetop.left:
            depth_search_the_tree(treetop.left, input_list)
            treetop.left = None
        input_list.append(treetop.input_list)
        if treetop.right:
            depth_search_the_tree(treetop.right, input_list)
            treetop.right = None
    return input_list


# for part 2, this one does the work, calls our functions, and prints the decoder key
def find_decoder_key(input_lines):
    node_deque = build_deque(input_lines)
    top_node_index = len(node_deque) - 1
    top_node = get_starting_node(node_deque)
    build_tree(top_node, node_deque, top_node_index)

    ordered_list = []
    depth_search_the_tree(top_node, ordered_list)
    decoder_key = 1
    for index, item in enumerate(ordered_list):
        if item == [2] or item == [6]:
            decoder_key *= index + 1
    print(decoder_key)
    return decoder_key


with open('input.txt') as file:
    lines = [line.strip() for line in file.readlines()]

part_1 = find_the_sum_of_passed_indices(lines)
part_2 = find_decoder_key(lines)
