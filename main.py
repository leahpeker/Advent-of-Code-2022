def main(file_name):
    with open(file_name) as file:
        lines = [x.strip() for x in file.readlines()]
    return part_1(lines), part_2(lines)