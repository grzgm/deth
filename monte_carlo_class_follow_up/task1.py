import random
import numpy as np
import matplotlib.pyplot as plt

number_of_walks = 10000

for walk_length in range(1, 5):
    # visited_intersection_count = [[0 for _ in range(walk_length * 2 + 1)] for __ in range(walk_length * 2 + 1)]
    visited_intersection_count = np.zeros((walk_length * 2 + 1, walk_length * 2 + 1))
    for i in range(number_of_walks):
        x, y = walk_length, walk_length
        for j in range(walk_length):
            (dx, dy) = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            x += dx
            y += dy
            visited_intersection_count[y][x] += 1

    for y in range(walk_length * 2 + 1):
        for x in range(walk_length * 2 + 1):
            visited_intersection_count[y][x] /= number_of_walks * walk_length

    # Reverse the order of rows (swap values on the y-axis) for more clear graph
    visited_intersection_count = visited_intersection_count[::-1, :]

    extent = [-walk_length - 0.5, walk_length + 0.5, -walk_length - 0.5, walk_length + 0.5]
    plt.imshow(visited_intersection_count, extent=extent)
    plt.colorbar()
    plt.title("Probability of visiting intersection Heat Map")
    plt.xlabel("X coordinates of grid")
    plt.ylabel("Y coordinates of grid")
    plt.show()
