import random
from camel.array import CamelArray

class DataLoader:
    """iterates (X_batch, y_batch) pairs over paired CamelArrays, reshuffled each epoch"""

    def __init__(self, X: CamelArray, y: CamelArray, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows, got {X.shape[0]} and {y.shape[0]}")
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = self.X.shape[0]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size) # ceil div

    def __iter__(self):
        n = self.X.shape[0]
        order = list(range(n))
        if self.shuffle:
            random.shuffle(order)

        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            yield self.X[idx], self.y[idx]


def k_fold_split(X: CamelArray, y: CamelArray, k: int, shuffle: bool = True):
    """yields (X_train, y_train, X_val, y_val) for each of k folds; folds partition the dataset exactly"""
    if k < 2:
        raise ValueError(f"k_fold_split needs k >= 2, got {k}")

    n = X.shape[0]
    order = list(range(n))
    if shuffle:
        random.shuffle(order)

    # remainder rows go one each to the first (n % k) folds, standard k-fold sizing
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    boundaries = [0]
    for size in fold_sizes:
        boundaries.append(boundaries[-1] + size)

    for i in range(k):
        val_idx = order[boundaries[i]:boundaries[i + 1]]
        train_idx = order[:boundaries[i]] + order[boundaries[i + 1]:]
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]
