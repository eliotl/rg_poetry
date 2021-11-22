import re
import numpy as np


def doppel_string(listA, listB):
    assert(len(listA) == len(listB))
    # Update each list item so that it's justified 
    # outList = [a.ljust(max(len(a), len(b))) + '\n' + b.ljust(max(len(a), len(b))) for a, b in zip(listA, listB)]
    outA = []
    outB = []
    for a, b in zip(listA, listB):
        m = max(len(a), len(b))
        outA.append(a.ljust(m))
        outB.append(b.ljust(m))
        # outList.append(a.ljust(m) + '\n' + b.ljust(m))
    outString = ' '.join(outA) + '\n' + ' '.join(outB)
    return outString

def set_intersection(listA, listB):
    return not set(listA).isdisjoint(listB)

def flatten_list(xlist):
    flat_list = [item for sublist in xlist for item in sublist]
    return flat_list


def get_endgrams(inpList, n):
    return [inpList[:n], inpList[-n:]]

def get_ngrams(inpList, n):
    return [inpList[i:i+n] for i in range(len(inpList)-n+1)]


def neighborhood(iterable):
    iterator = iter(iterable)
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (prev_item, current_item, next_item)
        prev_item = current_item
        current_item = next_item
    yield (prev_item, current_item, None)

def union_sets(sets):
    combo = set()
    for s in sets:
        combo = combo.union(s)
    return combo

def make_listMap(grafs, pad = 0):
    grafMap = np.cumsum([pad + len(graf) for graf in grafs])
    grafMap = np.insert(grafMap, 0, 0)
    return grafMap

def thing_to_map(sent_index, grafMap):
    mod = sent_index + 1
    idx = 0
    while idx < len(grafMap):
        if grafMap[idx] >= mod:
            break
        idx = idx + 1
    i1 = idx - 1
    i2 = max(0,mod - grafMap[idx-1] - 1)
    return [i1, i2]