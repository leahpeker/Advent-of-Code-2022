with open('rock_paper_scissors.txt') as file:
    rock_paper_scissors = file.readlines()


def part_1(rock_paper_scissors):

    rps_score_dict = {
        'X': 1,
        'Y': 2,
        'Z': 3,
    }

    winn_possibilities_dict = {
        'X': 'C',
        'Y': 'A',
        'Z': 'B'
    }

    match_dict = {
        'X': 'A',
        'Y': 'B',
        'Z': 'C'
    }

    score = 0
    for line in rock_paper_scissors:
        elf, me = line.split()
        score += rps_score_dict[me]
        # print(elf, 'next', me)
        if elf == match_dict[me]:
            # print('==', elf, me)
            score += 3
        elif elf == winn_possibilities_dict[me]:
            # print('win', elf, me)
            score += 6
    return score


def part_2(rock_paper_scissors):

    points_wld = {
        'X': 0,
        'Y': 3,
        'Z': 6,
    }

    points_for_beating = {
        'A': 2,
        'B': 3,
        'C': 1
    }

    points_for_draw = {
        'A': 1,
        'B': 2,
        'C': 3
    }

    points_for_lose = {
        'A': 3,
        'B': 1,
        'C': 2
    }

    score = 0
    for line in rock_paper_scissors:
        elf, strategy = line.split()
        score += points_wld[strategy]
        if strategy == 'Y':
            score += points_for_draw[elf]
        elif strategy == 'Z':
            score += points_for_beating[elf]
        else:
            score += points_for_lose[elf]

    return score

print(part_1(rock_paper_scissors))
print(part_2(rock_paper_scissors))