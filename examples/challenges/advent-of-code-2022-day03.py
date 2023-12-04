# Libraries
import re
import numpy as np
from pathlib import Path

# Path
path = Path('data/2022/day03')

with open(path / 'sample02.txt', 'r') as f:
    lines = f.read().split("\n")

print(lines)

# ----------------------------------
# Part 1
# ----------------------------------
# .. note: The value of each letter is the index in which the
#          letter is in the letters string plus 1 (because the
#          index start at 0).

# Letters
letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

cumu = 0
for l in lines:
    m = int(len(l)/2)
    c1, c2 = set(l[:m]), set(l[m:])
    intersection = c1 & c2
    cumu += letters.index(intersection.pop()) + 1


print(cumu)

# ----------------------------------
# Part 2
# ----------------------------------
# Create idxs
idxs = np.arange(len(lines)).reshape(-1, 3)



cumu = 0
for i,j,k in idxs:
    intersection = \
        set(lines[i]) & \
        set(lines[j]) & \
        set(lines[k])

    cumu += letters.index(intersection.pop()) + 1

print(cumu)