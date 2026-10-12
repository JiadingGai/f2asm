"""Deterministic randomized single-cycle chains shared across benchmark families."""
import random


def cycle(nodes, seed=0):
    if type(nodes) is not int or not 1 <= nodes <= 1 << 24:
        raise ValueError("Invalid node count")
    order = list(range(nodes))
    random.Random(seed).shuffle(order)
    nxt = [0] * nodes
    for a, b in zip(order, order[1:] + order[:1]):
        nxt[a] = b
    return nxt
