from collections import defaultdict
import re


def part_1(lines):

    stacks = lines[:8]
    stack_names = lines[8].strip()
    string_length = 4
    keys = [stack_names[i:i + 1] for i in range(0, len(stack_names), string_length)]
    int_keys = []
    for key in keys:
        key = int(key)
        int_keys.append(key)
    line_length = len(int_keys)
    stack_dict = defaultdict(list)

    for line in stacks[::-1]:
        value_list = [line[i:i + string_length - 1].strip() for i in range(0, len(line), string_length)]
        for i in range(0, line_length):
            if len(value_list[i]) > 0:
                stack_dict[int_keys[i]].append(value_list[i].strip('[').strip(']'))

    # print(stack_dict)

    moves = lines[10:]
    for move in moves:
        quantity, where_from, where_to = re.findall('\d+', move)
        for i in range(int(quantity)):
            stack_dict[int(where_to)].append(stack_dict[int(where_from)].pop())

    top_string = ''
    for value in stack_dict.values():
        top_string += (value[-1])

    return top_string



def part_2(lines):

    stacks = lines[:8]
    stack_names = lines[8].strip()
    string_length = 4
    keys = [stack_names[i:i + 1] for i in range(0, len(stack_names), string_length)]
    int_keys = []
    for key in keys:
        key = int(key)
        int_keys.append(key)
    line_length = len(int_keys)

    stack_dict = defaultdict(list)
    for line in stacks[::-1]:
        value_list = [line[i:i + string_length - 1].strip() for i in range(0, len(line), string_length)]
        for i in range(0, line_length):
            if len(value_list[i]) > 0:
                stack_dict[int_keys[i]].append(value_list[i].strip('[').strip(']'))

    moves = lines[10:]
    for move in moves:
        quantity, where_from, where_to = re.findall('\d+', move)
        top_bin = []
        for i in range(int(quantity)):
            top_bin.insert(0, stack_dict[int(where_from)].pop())
        stack_dict[int(where_to)].extend(top_bin)

    top_string = ''
    for value in stack_dict.values():
        top_string += (value[-1])

    return top_string


def main(file_name):
    with open(file_name) as file:
        lines = [x.strip('\n') for x in file.readlines()]
    return part_1(lines), part_2(lines)

print(main('input.txt'))