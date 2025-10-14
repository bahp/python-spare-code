# Libraries
import re
import numpy as np
from pathlib import Path

path = Path('data/2023/day05')

with open(path / 'sample01.txt', 'r') as f:
    lines  = f.read() #.split("\n")

# ------------------------------------------
# Part 1
# ------------------------------------------
# Get seeds and maps
groups = lines.split("\n\n")
seeds = list(map(int, re.findall('\d+', groups[0])))

# Create walk through matrix
matrix = [[s] for s in seeds]

# Walk for each seed
for i, s in enumerate(seeds):
    for m in groups[1:]:
        aux = m.split("\n")[1:]
        done = False
        curr = matrix[i][-1]
        for j in aux:
            ds, ss, l = list(map(int, j.split(" ")))
            if ss <= curr <= ss+l:
                matrix[i].append(ds + abs((ss-curr)))
                done = True
                break
        if not done:
            matrix[i].append(curr)

# See results
#print(matrix)

# Show answer
print(min(np.array(matrix)[:,-1]))



# ------------------------------------------
# Part 2
# ------------------------------------------
# .. note: Go reverse mode

def get_numbers(l):
    return list(map(int, l.split(" ")))

"""
backwards = [groups[1:][-1].split("\n")[-1]]
for m in groups[1:][::-1]:
    aux = m.split("\n")[1]
    for r in m.split("\n")[2:]:
        d, s, l = get_numbers(r)
        if r[0] < aux[0]:
            if backwards[-1]
            aux = r
    backwards.append(aux)

# 56 93 4
# 93 93 4
# 45 77 23
# 49 53 8
# 37 52 2
# 52 50 48
# 55

print(backwards)
"""











# ---------------------------------------------------------------
# Solution 2:
# ---------------------------------------------------------------
# Ref: https://github.com/morgoth1145/advent-of-code/blob/5311ed667714398cf02d9b2bc2a4e88f53f7b0dc/2023/05/solution.py

def solve(s, seed_interpreter, verbose=10):
    groups = s.split('\n\n')

    seed_ranges = seed_interpreter(list(map(int, groups[0].split()[1:])))

    for g in groups[1:]:
        step_mapping = [tuple(map(int, l.split()))
                        for l in g.splitlines()[1:]]

        new_ranges = []

        print("\n")
        print(g)
        for start, r_len in seed_ranges:

            if verbose>1:
                print(" "*2 + "seed=%s, range=%s" % (start, r_len))

            while r_len != 0:
                found_match = False
                best_dist = r_len

                for dst, src, length in step_mapping:
                    if src <= start < src+length:
                        # Found a match
                        off = start - src
                        rem_length = min(length - off, r_len)
                        new_ranges.append((dst+off, rem_length))
                        start += rem_length
                        r_len -= rem_length
                        found_match = True
                        if verbose>1:
                            print(" "*4 + "start=%s, r_len=%s, dst=%s, src=%s, len=%s, off=%s, rem_length=%s found=%s" %
                                (start, r_len, dst, src, length, off, rem_length, found_match))
                        break
                    else:
                        if start < src:
                            best_dist = min(src - start, best_dist)
                            print(" "*4 + "start=%s, r_len=%s, dst=%s, src=%s, length=%s, best_dist=%s" %
                                (start, r_len, dst, src, length, best_dist))

                if not found_match:
                    handling_len = min(best_dist, r_len)
                    new_ranges.append((start, handling_len))
                    start += handling_len
                    r_len -= handling_len

                    print(" "*4 + "dst=%s, src=%s, len=%s, off=%s, rem_length=%s found=%s" %
                        (dst, src, length, off, rem_length, found_match))

        seed_ranges = new_ranges

    print(seed_ranges)

    return min(start for start, length in seed_ranges)

def part1(s):
    def seed_interpreter(nums):
        return [(n, 1) for n in nums]
    return solve(s, seed_interpreter)


def part2(s):
    def seed_interpreter(nums):
        return list(zip(nums[::2], nums[1::2]))
    return solve(s, seed_interpreter)

# Answer
#print(part1(lines))
#print("\n\n\n\n")
print(part2(lines))

































import sys
sys.exit()


# ------------------------------------------
# Part 2
# ------------------------------------------
# Get seeds and maps
groups = lines.split("\n\n")
seeds = list(map(int, re.findall('\d+', groups[0])))
cumu = []
for start, length in zip(seeds[::2], seeds[1::2]):
    print(start,length)
    cumu.extend(list(range(start, start+length)))
print(cumu)
print(len(cumu))


seeds = cumu

# Create walk through matrix
matrix = [[s] for s in seeds]

# Walk for each seed
for i, s in enumerate(seeds):
    for m in groups[1:]:
        aux = m.split("\n")[1:]
        done = False
        curr = matrix[i][-1]
        for j in aux:
            ds, ss, l = list(map(int, j.split(" ")))
            if ss <= curr <= ss+l:
                matrix[i].append(ds + abs((ss-curr)))
                done = True
                break
        if not done:
            matrix[i].append(curr)

# See results
print(matrix)

print("AHH")
# Show answer
print(min(np.array(matrix)[:,-1]))


"""
# Loop
for m in groups[1:]:
    aux = m.split("\n")[1:]
    for i,s in enumerate(seeds):
        done = False
        for j in aux:
            ds, ss, l = list(map(int, j.split(" ")))
            if ss <= s <= ss+l:
                print(s, ds, ss, l, ds + abs((ss-s)))
                seeds[i] = ds + abs((ss-s))
                matrix[i].append(ds + abs((ss-s)))
                done = True
        if done:
            continue
    print("===>", seeds)
"""






















import sys
sys.exit()

seeds = list(map(int, re.findall('\d+', lines[0])))

#walk = [[e] for e in seeds]

#print(walk)

for l in lines[1:]:
    if l=='':
        continue
    # Get destination/source range start and range length
    numbers = list(map(int, re.findall('\d+', l)))
    #
    if not numbers:
        continue
    print(seeds)
    ds, ss, l = numbers
    aux = seeds
    for i,s in enumerate(seeds):
        if ss <= s <= ss+l:
            print(s, ds, ss, l, ds + abs((ss-s)))
            aux[i] = ds + (ss-s)
            continue

    print("END LOOP")
    print(aux)


import sys
sys.exit()


seeds = list(map(int, re.findall('\d+', lines[0])))

D = {}

for l in lines[1:]:
    if l=='':
        continue

    # Get destination/source range start and range length
    numbers = list(map(int, re.findall('\d+', l)))
    if numbers:
        ds, ss, l = numbers
        for i in range(l):
            aux[ss+i] = ds+i
    else:
        dest, source = l.split(" ")[0].split("-to-")
        key = '%s2%s' % (dest, source)
        D[key] = {}
        aux = D[key]

# Show dictionary
print(D)

def walk(s, d):
    curr = None
    for key,value in d.items():
        a = s
        s = value.get(s, s)
        print(key, a, s)
    return s

locations = [
    walk(s, D) for s in seeds
]

# See locations
print(locations)

# Show answer
print(min(locations))
