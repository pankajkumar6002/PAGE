from collections import deque
import numpy as np
from typing import List

class _Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0]*(n+1)
    def add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i
    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s
    def range_sum(self, l: int, r: int) -> int:
        if r < l: return 0
        return self.sum(r) - self.sum(l-1)
    def find_by_prefix(self, target: int) -> int:
        idx, bitmask = 0, 1 << (self.n.bit_length()-1)
        while bitmask:
            t = idx + bitmask
            if t <= self.n and self.bit[t] < target:
                target -= self.bit[t]
                idx = t
            bitmask >>= 1
        return idx + 1

def binpack_bfd(items: List[int], capacity: int) -> List[List[int]]:
    """
    Best-Fit Decreasing bin packing (integer sizes), returning original indices.

    Args:
        items: positive integers (each <= capacity)
        capacity: positive integer

    Returns:
        bins_idx: List of bins; each bin is a list of 0-based original indices.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if any(x <= 0 or int(x) != x for x in items):
        raise ValueError("All item sizes must be positive integers.")
    if np.max(items) > capacity:
        raise ValueError(f"Found item larger than capacity. capacity={capacity}, max_item={np.max(items)}")

    # sort by size desc (stable -> preserves index order among ties)
    seq = sorted([(int(w), i) for i, w in enumerate(items)], key=lambda t: -t[0])

    buckets = [deque() for _ in range(capacity+1)]  # bins by exact residual r
    ft = _Fenwick(capacity)                          # counts of bins per residual r (1..capacity)

    bins_idx: List[List[int]] = []
    residuals: List[int] = []

    for w, i in seq:
        if ft.range_sum(w, capacity) == 0:
            # open a new bin
            bins_idx.append([i])
            r = capacity - w
            residuals.append(r)
            if r > 0:
                buckets[r].append(len(bins_idx)-1)
                ft.add(r, +1)
        else:
            # tightest fit: smallest residual r >= w that exists
            target = ft.sum(w-1) + 1
            r = ft.find_by_prefix(target)
            b = buckets[r].pop()
            ft.add(r, -1)

            bins_idx[b].append(i)
            new_r = r - w
            residuals[b] = new_r
            if new_r > 0:
                buckets[new_r].append(b)
                ft.add(new_r, +1)

    return bins_idx

if __name__ == "__main__":
    items = [7,3,6,2,5,5,5,1,1]
    print(binpack_bfd(items, 10))
    # Example: [[0, 1], [2, 3, 7, 8], [4, 5], [6]]

