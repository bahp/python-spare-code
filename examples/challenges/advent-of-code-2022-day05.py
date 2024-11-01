# Libraries
import re
import numpy as np
from pathlib import Path
from collections import deque

# Path
path = Path('data/2022/day05')

with open(path / 'sample01.txt', 'r') as f:
    lines = f.read()

print(lines)

n = 4

# Divide into crates and moves
crates, moves = lines.split("\n\n")
crates = crates.split("\n")
moves = moves.split("\n")

board = np.array([list(l) for l in crates])

print(board)


"""
# Build stacks
stacks = []
for l in crates[:-1]:
    # Get chunks of 4 characters
    aux = [l[i:i+n][1] for i in range(0, len(l), n)]

    # Create stack
    stack = deque([])
    for e in aux:
        if e.isalpha():
            stack.append(e)
    stacks.append(stack)

# Perform moves
for m in moves:
    q, f, t = re.findall('\d+', m)
    print(stacks)
    print(q, f, t)
    for i in range(int(q)):
        e = stacks[int(q)-1].pop()
        stacks[int(t)-1].append(e)

print(stacks)
"""