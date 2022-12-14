import re
from collections import defaultdict


class Monkey:
    def __init__(self, monkey_id, starting_items, operation, divisibility, true_monkey_to, false_monkey_to, total_divisibility):
        self.monkey_id = monkey_id
        self.test = divisibility
        self.total_divisibility = total_divisibility
        self.true_monkey_to = true_monkey_to # who it goes to if test passes
        self.false_monkey_to = false_monkey_to # who it goes to if test fails
        self.operation = operation # given by operation line in input
        self.starting_items = starting_items
        self.inspections = len(starting_items)

    def pass_items(self, reduce_worry):
        move_items_dict = defaultdict(list)
        for item_importance in self.starting_items:
            where_to, current_importance = self.do_test(item_importance, reduce_worry)
            move_items_dict[where_to].append(current_importance)
        # print(move_items_dict)
        self.starting_items = []
        return move_items_dict

    def do_test(self, item_importance, reduce_worry):
        current_importance = self.do_operation(item_importance, reduce_worry)
        if current_importance % self.test == 0:
            where_to = self.true_monkey_to
        else:
            where_to = self.false_monkey_to
        return where_to, current_importance

    def do_operation(self, item_importance, reduce_worry):
        operation_list = self.operation.replace('old', str(item_importance)).split()
        current_importance = self.eval_binary_expr(*operation_list)
        if reduce_worry:
            current_importance = self.reduce_worry(current_importance)      # don't need to reduce worry in part 2
        else:
            current_importance = self.modular_div(current_importance)
        return current_importance

    def eval_binary_expr(self, op1, oper, op2):
        ops = {
            '+': lambda x, y: x + y,
            '*': lambda x, y: x * y,
        }
        op1, op2 = int(op1), int(op2)
        return ops[oper](op1, op2)

    def reduce_worry(self, item_importance):
        return item_importance // 3 # commented out for part 2, no longer benefit from worry relief

    # worry needs to be managed in part 2... we can continuously evaluate this
    def modular_div(self, item_importance):
        return item_importance % self.total_divisibility

    def add_items(self, item_dict):
        new_items = item_dict[self.monkey_id]
        self.starting_items.extend(new_items)
        self.count_items(new_items)
        item_dict[self.monkey_id] = []
        return item_dict

    def count_items(self, items):
        self.inspections += len(items)


def throw_it_all_around(num_of_cycles, reduce_worry, input_lines):
    monkey_dictionary = make_monkey_dictionary(input_lines)
    current_cycle = 0
    moving_items_dict = defaultdict(list)
    while current_cycle < num_of_cycles:
        for m_id, monkeys in monkey_dictionary.items():
            moving_items_dict = monkeys.add_items(moving_items_dict)
            move_items_dict = monkeys.pass_items(reduce_worry)
            for key, value in move_items_dict.items():
                moving_items_dict[key].extend(value)
        current_cycle += 1

    inspections_list = []
    for monkeys in monkey_dictionary.values():
        inspections_list.append(monkeys.inspections)
    inspections_list = sorted(inspections_list)
    monkey_business = inspections_list[-1] * inspections_list[-2]
    print(monkey_business)
    return monkey_business


def make_monkey_dictionary(input_lines):
    monkey_dict = defaultdict()
    # parsing the data
    length = len(lines)
    total_divisibility = 1
    for i in range(0, length, 7):
        monkey_id = int(re.findall('\d+', lines[i])[0])
        starting_items = [int(item) for item in re.findall('\d+', lines[i + 1])]
        operation = lines[i + 2][lines[i + 2].find('= ') + 2:]
        divisibility = int(re.findall('\d+', lines[i + 3])[0])
        total_divisibility *= divisibility
        true_monkey_to = int(re.findall('\d+', lines[i + 4])[0])
        false_monkey_to = int(re.findall('\d+', lines[i + 5])[0])
        monkey_dict[monkey_id] = Monkey(monkey_id, starting_items, operation, divisibility, true_monkey_to,
                                        false_monkey_to, 1)
    for monkey in monkey_dict.values():
        monkey.total_divisibility = total_divisibility

    return monkey_dict


with open('input.txt') as file:
    lines = [line.strip() for line in file.readlines()]

part_1 = throw_it_all_around(20, True, lines)
part_2 = throw_it_all_around(10000, False, lines)
