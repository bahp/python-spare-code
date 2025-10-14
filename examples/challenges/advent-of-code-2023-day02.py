# Libraries
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def get_number_and_color(l):
    """"""
    regexp = '((?P<number>\d+)\s+(?P<color>\w+))'
    search = list(re.finditer(regexp, l))
    return search[0].groupdict()

def get_game_information_list(l):
    """"""
    s1 = l.split(':')
    subsets = s1[1].split(";")
    game_id = re.search('[0-9]+', s1[0]).group(0)

    result = []
    print("game: %s" % game_id)
    for i, subset in enumerate(subsets):
        print("subset (%s): %s" % (i, subset))

        # Get list with numbers and colors
        cumu = [
            get_number_and_color(cubes)
                for cubes in subset.split(",")
        ]

        # Include game and subset information
        for e in cumu:
            e['subset'] = i
            e['game'] = game_id

        # Append to result
        result.append(cumu)

    # Flatten array
    list_1D = [item for sub_list in result for item in sub_list]

    # Return
    return list_1D


def list_to_dataframe(results):
    """"""
    # Create DataFrame
    df = pd.DataFrame(results)

    # Pivot DataFrame
    df_aux = df.pivot(
        index=['game', 'subset'],
        columns=['color'],
        values=['number']
    )

    # Format
    df_aux.columns = df_aux.columns.droplevel(0)
    df_aux = df_aux.reset_index()
    df_aux = df_aux.astype(pd.Int64Dtype())

    # Return
    return df_aux



# -----------------------------
# Part 1
# -----------------------------
# Path
path = Path('./data/2023/day02/')

# Read document
with open(path / 'sample02.txt', 'r') as f:
    lines = f.readlines()

# Get games information
results = []
for l in lines:
    results = results + get_game_information_list(l)

# Create DataFrame
df = list_to_dataframe(results)

# Show dataframe
print(df)

# Constraint
red = 12
green = 13
blue = 14

# Get indices of impossible games
idxs = (df.red>red) | (df.green>green) | (df.blue>blue)

# Get possible and impossible games ids
impossible = df[idxs].game.unique()
possible = set(df.game.unique()).difference(impossible)

# Compute answer (possible games)
answer = np.sum(list(possible))

# Show
print(answer)

# -----------------------------
# Part 1
# -----------------------------
# Find minimum number of cubes required
df2 = df.groupby('game').max()
# Compute the power
df2['power'] = df2.green * df2.red * df2.blue
# Compute the answer
answer = df2.power.sum()
print(answer)
