from collections import defaultdict, Counter

with open('elves_calorie_list.txt') as file:
   calories_df = file.readlines()

key = 0
calorie_dict = defaultdict(int)
for line in calories_df:
   if not line.strip():
      key += 1
   else:
      calorie_dict[key] += int(line)

calorie_counter = Counter(calorie_dict)
top_3 = calorie_counter.most_common(3)

sum = 0
for key, value in top_3:
   sum += value

print(sum)

print(calorie_counter)