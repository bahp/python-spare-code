# Libraries
import numpy as np

# Matrix
m = [
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ],
    [
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]
    ],

]

print(m)

# Convert to numpy array
np1 = np.array(m)
print(type(np1), np1.shape)

# Random matrix
mr = np.random.rand(12, 15, 400, 2)

# Save text
np.savetxt('matrix.txt', mr)

print(mr.shape)