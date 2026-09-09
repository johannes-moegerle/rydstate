import numpy as np


def build_segments(y_lists: list[list[float]]) -> list[list[tuple[int, int]]]:
    """Connect points of neighbouring x-columns into continuous line segments.

    Between two adjacent columns each point is matched to at most one point in
    the next column, greedily choosing the pairs with the smallest y difference
    first. The resulting one-to-one links form simple paths (segments); a new
    segment starts wherever a point has no match on its left.
    """
    cols = [np.asarray(ys, dtype=float) for ys in y_lists]
    edges = {}  # (i, j) -> (i + 1, k): links a point to its match in the next column
    for i in range(len(cols) - 1):
        left, right = cols[i], cols[i + 1]
        if left.size == 0 or right.size == 0:
            continue
        # |Δy| for every left/right pair at once, then visit pairs closest first
        dist = np.abs(left[:, None] - right[None, :])
        r = right.size
        used_left = np.zeros(left.size, dtype=bool)
        used_right = np.zeros(r, dtype=bool)
        remaining = min(left.size, r)
        for idx in np.argsort(dist, axis=None):
            j, k = divmod(int(idx), r)
            if used_left[j] or used_right[k]:
                continue
            used_left[j] = used_right[k] = True
            edges[(i, j)] = (i + 1, k)
            remaining -= 1
            if remaining == 0:  # every point in the smaller column is matched
                break

    targets = set(edges.values())
    segments = []
    for i, ys in enumerate(cols):
        for j in range(ys.size):
            if (i, j) in targets:
                continue  # not a segment start, it continues an earlier one
            seg = [(i, j)]
            while seg[-1] in edges:
                seg.append(edges[seg[-1]])
            segments.append(seg)
    return segments
