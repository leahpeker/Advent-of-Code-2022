import re
import numpy as np
np.set_printoptions(linewidth=400)


class Grid:
    def __init__(self):
        self.grid = np.empty((6, 40), dtype=str)
        self.detect_sprite()

    def detect_sprite(self):
        signal_strength_dict = calc_signal_strength(lines)[0]
        # y corresponds to cycle on current line
        for x, y in np.ndindex(self.grid.shape):
            current_cycle = (y + 1) + (x * 40)
            if current_cycle in signal_strength_dict.keys():
                xreg = signal_strength_dict[current_cycle]
            else:
                for key in range(current_cycle, 0, -1):
                    if key in signal_strength_dict:
                        xreg = signal_strength_dict[key]
                        break
            if y in range(xreg - 1, xreg + 2):
                # self.grid[x, y] = 1
                self.grid[x, y] = '#'
            else:
                self.grid[x, y] = ' '
        print(self.grid)
        return self.grid


def calc_signal_strength(lines):
    cycle = 1
    signal_strength = 1
    signal_strength_dict = {1: 1}

    for line in lines:
        if line.startswith('addx'):
            cycle += 2
            signal_strength += int(re.findall('-?\d+', line)[0])
            signal_strength_dict[cycle] = signal_strength
        else: #if line == 'noop':
            cycle += 1

    signal_strength_total = 0
    interesting_cycles = [20, 60, 100, 140, 180, 220]
    for cycle in interesting_cycles:
        if cycle in signal_strength_dict:
            signal_strength_total += cycle * signal_strength_dict[cycle]
        else:
            for key in range(cycle, 0, -1):
                if key in signal_strength_dict:
                    signal_strength_total += cycle * signal_strength_dict[key]
                    break
    part_1 = signal_strength_total
    return signal_strength_dict, part_1


with open('input.txt') as file:
    lines = [lines.strip() for lines in file.readlines()]
calc_signal_strength(lines)
grid = Grid()
#RFKZCPEF
