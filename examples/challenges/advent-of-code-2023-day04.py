# Libraries
import re
from pathlib import Path

path = Path('data/2023/day04')

with open(path / 'sample02.txt', 'r') as f:
    lines  = f.readlines()


# ---------------------------
# Part 1
# ---------------------------
#
cumu = 0

# Loop
for i,l in enumerate(lines):
    card, numbers = l.split(":")
    numbers = numbers.split("|")
    card_id = i

    # Find matching numbers
    numbers_win = set(re.findall(r"\d+", numbers[0]))
    numbers_own = set(re.findall(r"\d+", numbers[1]))
    N = len(numbers_own & numbers_win)

    if N:
        cumu += 2**(N-1)

# Show answer
print(cumu)


# ---------------------------
# Part 2
# ---------------------------
# .. note: Extracted from the reddit thread and included
#          a minor modification. This is a nice and
#          efficient solution.

cards = []
for row in lines:
    game = row.split(":")
    numbers = game[1].split("|")
    gid = re.findall(r'\d+', game[0])
    left = set(re.findall(r'\d+', numbers[0]))
    right = set(re.findall(r'\d+', numbers[1]))

    win = len(left & right)
    """
    win = 0
    for n in left:
        for k in right:
            if n == k:
                win += 1
    """

    cards.append([int(gid[0]), win, 1])


for gid, win, copies in cards:
    for i in range(win):
        cards[gid+i][2] += copies

# Count scratchcards
sums = 0
for a,b,c in cards:
    sums += c

# Answer
print(sums)




"""
# ---------------------------
# Part 2
# ---------------------------
# .. note: This solution is really slow because it goes
#          recomputing cards for which we already now
#          the matching. See above for a cleaner solution

# Libraries
from collections import deque

def get_card_id(txt):
    return int(re.search(r'\d+', txt.split(":")[0]).group(0))

def get_matches(txt):
    aux = txt.split(":")[1]
    numbers_win = re_number_list(aux.split("|")[0])
    numbers_own = re_number_list(aux.split("|")[1])
    return numbers_own.intersection(numbers_win)


# Create empty queue
q = deque(lines)
c = 0

# Loop
while len(q):
    c += 1

    item = q.popleft()
    card_id = get_card_id(item)
    matches = get_matches(item)

    for i in range(len(matches)):
        q.append(lines[card_id+i])

# Answer
print(c)
"""




