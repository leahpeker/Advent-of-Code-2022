


def part_1(lines):
    sum = 0
    for line in lines:
        left, right = [x.split('-') for x in line.split(',')]

        left_set = set(range(int(left[0]), int(left[1]) + 1))
        right_set = set(range(int(right[0]), int(right[1]) + 1))

        intersection = left_set.intersection(right_set)

        if intersection:
            if intersection == left_set or intersection == right_set:
                sum += 1
    return sum


def part_2(lines):

    sum = 0
    for line in lines:
        left, right = [x.split('-') for x in line.split(',')]

        left_set = set(range(int(left[0]), int(left[1]) + 1))
        right_set = set(range(int(right[0]), int(right[1]) + 1))

        intersection = left_set.intersection(right_set)

        if intersection:
            sum += 1

    return sum


def main(file_name):
    with open(file_name) as file:
        lines = [x.strip() for x in file.readlines()]
    return part_1(lines), part_2(lines)

print(main('dataset.txt'))
