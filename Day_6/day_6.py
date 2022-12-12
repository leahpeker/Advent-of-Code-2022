def part_1(line):

    starting_index = 3
    while True:
        string = line[starting_index - 3: starting_index + 1]
        for j in range(len(string)):
            if string.count(string[j]) > 1:
                break
            if j == 3:
                result = starting_index + 1
                return result
        starting_index += 1



def part_2(line):

    starting_index = 13
    while True:
        string = line[starting_index - 13: starting_index + 1]
        for j in range(len(string)):
            if string.count(string[j]) > 1:
                break
            if j == 13:
                result = starting_index + 1
                return result
        starting_index += 1



def main(file_name):
    with open(file_name) as file:
        line = file.read()
    return part_1(line), part_2(line)

print(main('input.txt'))
