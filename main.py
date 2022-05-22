import math

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

DATA = [
    (4., 3., 10., 4., 3., 5., 5.),
    (6., 3., 9., 4., 7., 2., 1.),
    (5., 4., 5., 4., 6., 4., 1.),
    (5., 3., 6., 3., 3., 7., 4.),
    (7., 10., 10., 8., 8., 1., 3.),
    (8., 4., 7., 5., 8., 2., 3.),
    (3., 2., 5., 2., 4., 8., 2.),
    (3., 2., 3., 3., 5., 10., 3.),
    (7., 3., 3., 3., 3., 7., 2.),
    (2., 2., 3., 1., 2., 10., 1.),
    (9., 3., 5., 4., 7., 2., 2.),
    (5., 5., 2., 3., 2., 5., 3.),
    (1., 5., 5., 2., 6., 6., 1.),
    (7., 5., 2., 7., 7., 1., 4.),
    (2., 2., 4., 2., 2., 1., 5.),
    (3., 3., 1., 3., 5., 1., 2.),
    (7., 3., 3., 4., 8., 5., 3.),
]


def to_distance_mx(data):
    entities = len(data)
    return [[math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[x], data[y])])) for y in range(entities)] for x in range(entities)]


def gather_pair(distance_mx, ignore=None):
    min_x = min_y = min_v = None

    for x, col in enumerate(distance_mx):
        c_min = min([x for x in col if x != 0])

        if min_v is not None and c_min >= min_v:
            continue

        if ignore and (x, col.index(c_min)) in ignore:
            continue

        min_x = x
        min_y = col.index(c_min)
        min_v = c_min

    return (min_x, min_y), min_v

def flatten(l):
    for x in l:
        yield from x


def avg_dist(data):
    return sum(list(flatten(to_distance_mx(data)))) / len(data) ** 2


def clusterize(data, dist):
    c_distance = 0
    d_mx = to_distance_mx(data)
    ignore = []

    while c_distance < dist:
        group, c_distance = gather_pair(d_mx, ignore=ignore)
        ignore.append(group)

    groups = []
    for a, b in ignore:
        for group in groups:
            if a in group:
                group.add(b)
                break
            if b in group:
                group.add(a)
                break
        else:
            groups.append({a, b})

    return groups


if __name__ == '__main__':
    print(clusterize(DATA, avg_dist(DATA) * 0.5))

    l = hierarchy.linkage(DATA, 'single')
    plt.figure()
    dendrogram(l)
    plt.show()
