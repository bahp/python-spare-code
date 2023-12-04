# Libraries
import re
import numpy as np
from pathlib import Path

# Path
path = Path('data/2022/day02')

with open(path / 'sample02.txt', 'r') as f:
    lines = f.read().split("\n")


# ----------------------------------
# Part 1
# ----------------------------------
rules = {
    'AX': 1 + 3,
    'AY': 2 + 6,
    'AZ': 3 + 0,
    'BX': 1 + 0,
    'BY': 2 + 3,
    'BZ': 3 + 6,
    'CX': 1 + 6,
    'CY': 2 + 0,
    'CZ': 3 + 3,
}

score = 0
for l in lines:
    l1, l2 = l.split(" ")
    score += rules['%s%s' % (l1, l2)]

print(score)

# ----------------------------------
# Part 2
# ----------------------------------
rules = {
    'AX': 3 + 0,
    'AY': 1 + 3,
    'AZ': 2 + 6,
    'BX': 1 + 0,
    'BY': 2 + 3,
    'BZ': 3 + 6,
    'CX': 2 + 0,
    'CY': 3 + 3,
    'CZ': 1 + 6,
}

score = 0
for l in lines:
    l1, l2 = l.split(" ")
    score += rules['%s%s' % (l1, l2)]

print(score)