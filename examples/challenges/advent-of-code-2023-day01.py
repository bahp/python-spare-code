# Libraries
import re
import numpy as np
from pathlib import Path

# Path
path = Path('./data/2023/day01/')

# -----------------------------
# Part 1
# -----------------------------
# Read document
with open(path / 'sample01.txt', 'r') as f:
    lines = f.readlines()

numbers = []
for l in lines:
    n1 = re.search(r'(\d)', l).group()
    n2 = re.search(r'(\d)', l[::-1]).group()
    numbers.append(int('%s%s' % (n1, n2)))

# Answer
print(np.sum(numbers))





# -----------------------------
# Part 2
# -----------------------------
# Replacements
r = {
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9'
}

def find_first_and_last_digits(l, map):
    """Find first and last digit (including text)

    It uses look ahead to find the character without consuming
    those characters and a simple map to convert the text to
    number if it was not a digit.
    """
    regexp = '(?=(one|two|three|four|five|six|seven|eight|nine|1|2|3|4|5|6|7|8|9))'
    n = re.findall(regexp, l)
    tmp = [n[0], n[-1]]
    tmp = [e if e.isdigit() else map[e] for e in tmp]
    return int(''.join(tmp))

# Read document
with open(path / 'sample02.txt', 'r') as f:
    lines = f.readlines()

numbers = []
for l in lines:
    n = find_first_and_last_digits(l, r)
    numbers.append(n)

# Answer
print(np.sum(numbers))
