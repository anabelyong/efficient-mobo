import numpy as np

"""
Adapted from:
1) https://github.com/pytorch/botorch/blob/main/botorch/utils/multi_objective/hypervolume.py
2) https://github.com/msu-coinlab/pymoo/blob/main/pymoo/vendor/hv.py

"""
# Constants
MIN_Y_RANGE = 1e-7


# Hypervolume class to calculate the hypervolume
class Hypervolume:
    def __init__(self, ref_point):
        self.ref_point = -ref_point

    def compute(self, pareto_Y):
        if pareto_Y.shape[-1] != self.ref_point.shape[0]:
            raise ValueError("pareto_Y must have the same number of objectives as ref_point.")
        if pareto_Y.ndim != 2:
            raise ValueError("pareto_Y must have exactly two dimensions.")

        pareto_Y = -pareto_Y
        better_than_ref = (pareto_Y <= self.ref_point).all(axis=-1)
        pareto_Y = pareto_Y[better_than_ref]
        pareto_Y = pareto_Y - self.ref_point
        self._initialize_multilist(pareto_Y)
        bounds = np.full_like(self.ref_point, -np.inf)
        return self._hv_recursive(i=self.ref_point.shape[0] - 1, n_pareto=pareto_Y.shape[0], bounds=bounds)

    def _hv_recursive(self, i, n_pareto, bounds):
        """
        1) Sentinel: The sentinel node in the doubly linked list structure used to manage the Pareto points.
        2) self: Refers to the instance of the Hypervolume class.
        3) i: The current dimension being processed.
        4) n_pareto: The number of Pareto points currently considered.
        5) bounds: An array that keeps track of the bounds in each dimension.
        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if n_pareto == 0:
            return hvol
        elif i == 0:
            return -sentinel.next[0].data[0]
        elif i == 1:
            q = sentinel.next[1]
            h = q.data[0]
            p = q.next[1]
            while p is not sentinel:
                hvol += h * (q.data[1] - p.data[1])
                if p.data[0] < h:
                    h = p.data[0]
                q = p
                p = q.next[1]
            hvol += h * q.data[1]
            return hvol
        else:
            p = sentinel
            q = p.prev[i]
            while q.data is not None:
                if q.ignore < i:
                    q.ignore = 0
                q = q.prev[i]
            q = p.prev[i]
            while n_pareto > 1 and (q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]):
                p = q
                self.list.remove(p, i, bounds)
                q = p.prev[i]
                n_pareto -= 1
            q_prev = q.prev[i]
            if n_pareto > 1:
                hvol = q_prev.volume[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i])
            else:
                q.area[0] = 1
                q.area[1 : i + 1] = q.area[:i] * -q.data[:i]
            q.volume[i] = hvol
            if q.ignore >= i:
                q.area[i] = q_prev.area[i]
            else:
                q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                if q.area[i] <= q_prev.area[i]:
                    q.ignore = i
            while p is not sentinel:
                p_data = p.data[i]
                hvol += q.area[i] * (p_data - q.data[i])
                bounds[i] = p_data
                self.list.reinsert(p, i, bounds)
                n_pareto += 1
                q = p
                p = p.next[i]
                q.volume[i] = hvol
                if q.ignore >= i:
                    q.area[i] = q.prev[i].area[i]
                else:
                    q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                    if q.area[i] <= q.prev[i].area[i]:
                        q.ignore = i
            hvol -= q.area[i] * q.data[i]
            return hvol

    def _initialize_multilist(self, pareto_Y):
        m = pareto_Y.shape[-1]
        nodes = [Node(m=m, data=point) for point in pareto_Y]
        self.list = MultiList(m=m)
        for i in range(m):
            sort_by_dimension(nodes, i)
            self.list.extend(nodes, i)


def sort_by_dimension(nodes, i):
    decorated = [(node.data[i], index, node) for index, node in enumerate(nodes)]
    decorated.sort()
    nodes[:] = [node for (_, _, node) in decorated]


class Node:
    def __init__(self, m, data=None):
        self.data = data
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = np.zeros(m)
        self.volume = np.zeros_like(self.area)


class MultiList:
    def __init__(self, m):
        self.m = m
        self.sentinel = Node(m=m)
        self.sentinel.next = [self.sentinel] * m
        self.sentinel.prev = [self.sentinel] * m

    def append(self, node, index):
        last = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last
        self.sentinel.prev[index] = node
        last.next[index] = node

    def extend(self, nodes, index):
        for node in nodes:
            self.append(node, index)

    def remove(self, node, index, bounds):
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
        bounds[:] = np.minimum(bounds, node.data)
        return node

    def reinsert(self, node, index, bounds):
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
        bounds[:] = np.minimum(bounds, node.data)


# Function to infer the reference point
def infer_reference_point(pareto_Y, max_ref_point=None, scale=0.1, scale_max_ref_point=False):
    """
    1) Compute a suitable reference point for hypervolume calculations,
    handling edge cases such as empty Pareto sets and presence of NaN values.
    2) This is same function as the one in the Botorch implementation but we are not using torch functions like
    tensor.isnan(), uses arrays insteads of tensors, initializes non_nan_idx and better_than_ref with arrays.
    3) Change '.all(dim=-1), .clamp_min(MIN_Y_RANGE), .view(-1)' to '.all(axis=-1), np.clip(), .reshape(-1)' <- TODO: verify this
    does the same thing.
    4) useful for when we do not have a manually calculated reference point, rather this is found in the dataset.
    """
    if pareto_Y.shape[0] == 0:
        if max_ref_point is None:
            raise ValueError("Empty pareto set and no max ref point provided")
        if np.isnan(max_ref_point).any():
            raise ValueError("Empty pareto set and max ref point includes NaN.")
        return max_ref_point - scale * np.abs(max_ref_point) if scale_max_ref_point else max_ref_point

    if max_ref_point is not None:
        non_nan_idx = ~np.isnan(max_ref_point)
        better_than_ref = (pareto_Y[:, non_nan_idx] > max_ref_point[non_nan_idx]).all(axis=-1)
    else:
        non_nan_idx = np.ones(pareto_Y.shape[-1], dtype=bool)
        better_than_ref = np.ones(pareto_Y.shape[:1], dtype=bool)

    if max_ref_point is not None and better_than_ref.any() and non_nan_idx.all():
        Y_range = pareto_Y[better_than_ref].max(axis=0) - max_ref_point
        return max_ref_point - scale * Y_range if scale_max_ref_point else max_ref_point

    if pareto_Y.shape[0] == 1:
        Y_range = np.clip(np.abs(pareto_Y), MIN_Y_RANGE, None).reshape(-1)
        ref_point = pareto_Y.reshape(-1) - scale * Y_range
    else:
        nadir = pareto_Y.min(axis=0)
        if max_ref_point is not None:
            nadir[non_nan_idx] = np.minimum(nadir[non_nan_idx], max_ref_point[non_nan_idx])
        ideal = pareto_Y.max(axis=0)
        Y_range = np.clip(ideal - nadir, MIN_Y_RANGE, None)
        ref_point = nadir - scale * Y_range

    if non_nan_idx.any() and not non_nan_idx.all() and better_than_ref.any():
        ref_point[non_nan_idx] = (
            (max_ref_point - scale * Y_range)[non_nan_idx] if scale_max_ref_point else max_ref_point[non_nan_idx]
        )

    return ref_point


# test case values extracted from https://github.com/pytorch/botorch/blob/main/test/utils/multi_objective/test_hypervolume.py
if __name__ == "__main__":
    ref_point = np.array([0.0, 0.0])
    hv = Hypervolume(ref_point)
    pareto_Y = np.array([[1, 2], [2, 1]])
    volume = hv.compute(pareto_Y)
    print("Computed Hypervolume:", volume)

    test_cases = [
        {
            "ref_point": [0.0, 0.0],
            "pareto_Y": np.array([[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]]),
            "expected_volume": 37.75,
        },
        {
            "ref_point": [1.0, 0.5],
            "pareto_Y": np.array([[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]]),
            "expected_volume": 28.75,
        },
        {
            "ref_point": [-2.1, -2.5, -2.3],
            "pareto_Y": np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            "expected_volume": 11.075,
        },
        {
            "ref_point": [-2.1, -2.5, -2.3, -2.0],
            "pareto_Y": np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ]
            ),
            "expected_volume": 23.15,
        },
        {
            "ref_point": [-1.1, -1.1, -1.1, -1.1, -1.1],
            "pareto_Y": np.array(
                [
                    [-0.4289, -0.1446, -0.1034, -0.4950, -0.7344],
                    [-0.5125, -0.5332, -0.3678, -0.5262, -0.2024],
                    [-0.5960, -0.3249, -0.5815, -0.0838, -0.4404],
                    [-0.6135, -0.5659, -0.3968, -0.3798, -0.0396],
                    [-0.3957, -0.4045, -0.0728, -0.5700, -0.5913],
                    [-0.0639, -0.1720, -0.6621, -0.7241, -0.0602],
                ]
            ),
            "expected_volume": 0.42127855991587,
        },
    ]

    for i, case in enumerate(test_cases):
        ref_point = np.array(case["ref_point"])
        pareto_Y = case["pareto_Y"]
        expected_volume = case["expected_volume"]

        hv = Hypervolume(ref_point)
        computed_volume = hv.compute(pareto_Y)

        print(f"Test case {i+1}:")
        print(f"  Reference point: {ref_point}")
        print(f"  Pareto front: {pareto_Y}")
        print(f"  Expected volume: {expected_volume}")
        print(f"  Computed volume: {computed_volume}")

        if np.isclose(computed_volume, expected_volume, atol=1e-3):
            print(f"  Test case {i+1} passed.")
        else:
            print(f"  Test case {i+1} failed.")
