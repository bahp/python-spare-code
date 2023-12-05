# Libraries
import re
import numpy as np
from pathlib import Path

# Path
path = Path('data/2022/day04')

with open(path / 'sample02.txt', 'r') as f:
    lines = f.read().split("\n")

print(lines)

# --------------------------------------------------
# Part 1
# --------------------------------------------------
count = 0
for l in lines:
    range1, range2 = l.split(",")
    s1, e1 = range1.split("-")
    s2, e2 = range2.split("-")

    if (int(s1) <= int(s2) <= int(e2) <= int(e1)) or \
       (int(s2) <= int(s1) <= int(e1) <= int(e2)):
        count += 1

print(count)

# --------------------------------------------------
# Part 1
# --------------------------------------------------
# .. note: Two ranges overlap if the larger of their start
#          values is larger than the smaller of their stop
#          values.

count = 0
for l in lines:
    range1, range2 = l.split(",")
    s1, e1 = range1.split("-")
    s2, e2 = range2.split("-")

    if max(int(s1), int(s2)) <= min(int(e1), int(e2)):
        count += 1

print(count)