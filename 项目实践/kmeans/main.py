
import math


class point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)


def get_center(plst):
    x = 0
    y = 0
    for i in plst:
        x = x + i.x
        y = y + i.y
    return (x/len(plst), y/len(plst))


data = [
    point(3, 4),
    point(3, 6),
    point(7, 3),
    point(4, 7),
    point(3, 8),
    point(8, 5),
    point(4, 5),
    point(4, 1),
    point(7, 4),
    point(5, 5)
]

tmp = [
    data[2],
    data[5],
    data[7],
    data[8],
]
print(get_center(tmp))

for i in data:
    print(get_distance(point(6, 2.67), i))
