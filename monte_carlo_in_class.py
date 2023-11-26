import random

def rw(n):
    x, y = 0, 0
    for i in range(n):
        (dx, dy) = random.choice([(0,1), (0,-1), (1, 0), (-1, 0)])
        x += dx
        y += dy
    return (x, y)

data = []

for j in range(10000):
    x, y = rw(13)
    data.append(abs(x) + abs(y))

print(sum(data)/len(data))