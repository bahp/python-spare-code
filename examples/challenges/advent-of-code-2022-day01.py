# Libraries
import re
import numpy as np
from pathlib import Path

# Path
path = Path('data/2022/day01')

with open(path / 'sample02.txt', 'r') as f:
    lines = f.read().split("\n")

print(lines)

# ----------------------------------
# Part 1
# ----------------------------------
m = 0 # max calories
p = 0 # partial calories
for c in lines:

    if c == '':
        if p>m:
            m = p
        p = 0
        continue
    p += int(c)

# Solution
print(m)


# ----------------------------------
# Part 2
# ----------------------------------
m = [] # max calories
p = 0 # partial calories
for c in lines:
    if c == '':
        m.append(p)
        p = 0
        continue
    p += int(c)

# Solution
print(np.sum(sorted(m)[-3:]))