import string


priority_dict = {}
for letter, num in zip(list(string.ascii_lowercase), range(1, 27)):
    priority_dict[letter] = num
    priority_dict[letter.upper()] = num + 26

with open('rucksack.txt') as file:
    rucksack_lines = file.readlines()

def part_1(rucksack_lines):
    sum = 0
    for line in rucksack_lines:
        length = int(len(line.strip())/2)
        left = set(line[:length])
        right = set(line[length:])
        intersection = left.intersection(right)
        key = intersection.pop()
        sum += priority_dict[key]
    return sum

def part_2(rucksack_lines):

    counter = 1
    sum = 0
    set_1 = set()
    set_2 = set()
    for line in rucksack_lines:
        line = line.strip()
        if counter == 1:
            set_1 = set(line)
        elif counter == 2:
            set_2 = set(line)
        elif counter == 3:
            set_3 = set(line)
            intersection_1 = set_1.intersection(set_2)
            intersection_2 = intersection_1.intersection(set_3)
            key = intersection_2.pop()
            sum += priority_dict[key]
            counter = 0
        counter += 1
    return sum

print(part_1(rucksack_lines))
print(part_2(rucksack_lines))