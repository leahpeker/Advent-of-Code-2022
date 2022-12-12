def part_1(rock_paper_scissors):

    # contains the points for getting each score, plus the conditions for winning or tying
    score_win_match_dict = {
        'X': [1, 'C', 'A'],
        'Y': [2, 'A', 'B'],
        'Z': [3, 'B', 'C'],
    }

    score = 0
    for line in rock_paper_scissors:
        elf, me = line
        score += score_win_match_dict[me][0]
        if elf == score_win_match_dict[me][2]:
            score += 3
        elif elf == score_win_match_dict[me][1]:
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

    # lays out the amount of points an elf receives first for a straight win, loss, or draw (6, 0, 3)
    # #then tells us how many points
    # wldpoints_wintype_drawtype_losetype_dict = {
    #     'A': [0, 2, 'A'],
    #     'B': [3, 3, 'B'],
    #     'C': [6, 1, 'C'],
    # }

    score = 0
    for line in rock_paper_scissors:
        elf, strategy = line
        score += points_wld[strategy]
        if strategy == 'Y':
            score += points_for_draw[elf]
        elif strategy == 'Z':
            score += points_for_beating[elf]
        else:
            score += points_for_lose[elf]

    return score


with open('rock_paper_scissors.txt') as file:
    # rock_paper_scissors = file.readlines()
    elf_strategy = [line.split() for line in file.readlines()]
print(elf_strategy)
print(part_1(elf_strategy))
print(part_2(elf_strategy))
