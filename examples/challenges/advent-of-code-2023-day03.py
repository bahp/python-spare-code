# Libraries
import re
import numpy as np
from pathlib import Path

# Path
path = Path('./data/2023/day03/')

# -----------------------------
# Part 1
# -----------------------------

# .. note: If loading the whole data and using a regexp in
#          multiline mode, it is important to highlight that
#          each line ends with \n and hence start() and end()
#          will be shifted by the number of lines before.

# .. note: Although we might be able to assume that all lines
#          will be the same length. We will not do that just
#          in case. That is the reason we will be doing the
#          regexp search line by line.

def re_match_to_dict(match, i):
    return {
        'v': match.group(0),
        's': match.start(),
        'e': match.end(),
        'i': i
    }

# Read document
with open(path / 'sample02.txt', 'r') as f:
    lines = f.readlines()

# Retrieve numbers and symbols
numbers, symbols = [], []
for i,line in enumerate(lines):
    numbers += [re_match_to_dict(m, i)
        for m in re.finditer('\d+', line)]
    symbols += [re_match_to_dict(m, i)
        for m in re.finditer('[^\d\.\s]', line)]

# Numbers with adjacent symbol
adjn = []

# See if they are adjacent (also diagonally).
for n in numbers:
    for s in symbols:
        ins, jns = n['i'], n['s'] # number start
        ine, jne = n['i'], n['e'] # number end
        iss, jss = s['i'], s['s'] # symbol start
        ise, jse = s['i'], s['e'] # symbol end

        if abs(ins-iss) > 1:
            continue

        print('%5s | s=%2s, e=%2s | i=%s, j=%s-%s' % (n['v'], n['s'], n['e'], ins, jns, jne))
        print('%5s | s=%2s, e=%2s | i=%s, j=%s-%s' % (s['v'], s['s'], s['e'], iss, jss, jse))

        if (jns-1<=jss) and (jse-1<=jne):
            adjn.append(n['v'])
            break

# Show numbers
print(adjn)

# Compute answer
answer = np.sum(np.array(adjn, int))

# Show answer
print(answer)


# -----------------------------
# Part 2
# -----------------------------
# Find gear
asterisks = [d for d in symbols if d['v'] == '*']

# Numbers with adjacent symbol
gears = 0

# See if they are adjacent (also diagonally).
for g in asterisks:
    number_list = []
    for n in numbers:
        ins, jns = n['i'], n['s']  # number start
        ine, jne = n['i'], n['e']  # number end
        iss, jss = g['i'], g['s']  # gear start
        ise, jse = g['i'], g['e']  # gear end

        if abs(ins-iss) > 1:
            continue

        if (jns - 1 <= jss) and (jse - 1 <= jne):
            number_list.append(n['v'])

    if len(number_list) == 2:
        gears += np.prod(np.array(number_list, int))

# Answer
print(gears)