from collections import defaultdict
import re


def organize_directories(lines):

    storage_dictionary = defaultdict(int)
    parent_history = ['']
    current_parent = '/'
    for line in lines:
        if line == '$ cd ..':
            if parent_history[-1] == '':
                current_parent = '/'
            else:
                current_parent = parent_history.pop()
            continue

        if line == '$ ls':
            continue

        if line.startswith('$ cd'):
            if not current_parent == '' and not current_parent == '/':
                parent_history.append(current_parent)
            current_parent = line.replace('$ cd ', '')
            continue

        if line.startswith('dir'):
            continue

        storage_rqd = re.findall('\d+', line)
        if not current_parent == '/':
            for i in range(len(parent_history), 0, -1):
                filepath = '/'.join(parent_history[0:i])
                storage_dictionary[filepath] += int(storage_rqd[0])
            filepath = '/'.join(parent_history) + '/' + current_parent
        else:
            filepath = ''
        storage_dictionary[filepath] += int(storage_rqd[0])

    currently_unused = 70000000 - storage_dictionary['']
    space_rqd = 30000000 - currently_unused
    space_rqd_dict = {0: 30000000, 1: 'string'}
    sum = 0
    for key, value in storage_dictionary.items():
        if value <= 100000:
            sum += value
        if value >= space_rqd and int(value) <= int(space_rqd_dict[0]):
            space_rqd_dict = {0: value, 1: key}
    return 'part 1:', sum, 'part 2:', space_rqd_dict[0]


with open ('input.txt') as file:
    lines = [x.strip() for x in file.readlines()]

print(organize_directories(lines))