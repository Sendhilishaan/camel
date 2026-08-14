from camel import array as ca
from camel.data import DataLoader, k_fold_split


def check_row_slice():
    a = ca.CamelArray([[i, i * 10] for i in range(6)])
    original = a.tolist()
    s = a[2:5]
    assert s.shape == (3, 2)
    assert s.tolist() == [2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
    assert a.tolist() == original, "slicing mutated the original array"
    print("row slice: OK")


def check_row_gather():
    a = ca.CamelArray([[i, i * 10] for i in range(6)])
    g = a[[3, 0, 1]]
    assert g.shape == (3, 2)
    assert g.tolist() == [3.0, 30.0, 0.0, 0.0, 1.0, 10.0]
    print("row gather: OK")


def check_dataloader_coverage():
    """with shuffle off, batches cover every row in order, batch_size each except possibly the last"""
    n, batch_size = 23, 5
    X = ca.CamelArray([[float(i)] for i in range(n)])
    y = ca.CamelArray([[float(i)] for i in range(n)])
    loader = DataLoader(X, y, batch_size, shuffle=False)

    seen = []
    sizes = []
    for xb, yb in loader:
        assert xb.shape[0] == yb.shape[0]
        sizes.append(xb.shape[0])
        seen.extend(int(v) for v in xb.tolist())

    assert seen == list(range(n)), "not all rows covered in order"
    assert sizes[:-1] == [batch_size] * (len(sizes) - 1), "non-last batches should be full"
    assert sizes[-1] == n % batch_size or sizes[-1] == batch_size
    assert len(loader) == len(sizes)
    print("dataloader coverage (no shuffle): OK")


def check_dataloader_shuffle_coverage():
    """with shuffle on, every row still appears exactly once per epoch, just reordered"""
    import random
    random.seed(0)

    n, batch_size = 37, 8
    X = ca.CamelArray([[float(i)] for i in range(n)])
    y = ca.CamelArray([[float(i)] for i in range(n)])
    loader = DataLoader(X, y, batch_size, shuffle=True)

    seen = []
    for xb, _ in loader:
        seen.extend(int(v) for v in xb.tolist())

    assert sorted(seen) == list(range(n)), "shuffled epoch didn't cover every row exactly once"
    print("dataloader coverage (shuffle): OK")


def check_dataloader_drop_last():
    n, batch_size = 23, 5
    X = ca.CamelArray([[float(i)] for i in range(n)])
    y = ca.CamelArray([[float(i)] for i in range(n)])
    loader = DataLoader(X, y, batch_size, shuffle=False, drop_last=True)

    batches = list(loader)
    assert len(batches) == n // batch_size
    assert all(xb.shape[0] == batch_size for xb, _ in batches)
    assert len(loader) == len(batches)
    print("dataloader drop_last: OK")


def check_kfold_partition():
    """folds partition the dataset: val sets disjoint, their union is everything, sizes sum to n"""
    import random
    random.seed(1)

    n, k = 33, 5
    X = ca.CamelArray([[float(i)] for i in range(n)])
    y = ca.CamelArray([[float(i)] for i in range(n)])

    all_val = []
    for X_tr, y_tr, X_val, y_val in k_fold_split(X, y, k, shuffle=True):
        assert X_tr.shape[0] + X_val.shape[0] == n
        assert y_tr.shape[0] == X_tr.shape[0] and y_val.shape[0] == X_val.shape[0]
        val_rows = [int(v) for v in X_val.tolist()]
        all_val.extend(val_rows)

    assert sorted(all_val) == list(range(n)), "val folds don't exactly partition the dataset"
    print("k_fold_split partition: OK")


if __name__ == "__main__":
    check_row_slice()
    check_row_gather()
    check_dataloader_coverage()
    check_dataloader_shuffle_coverage()
    check_dataloader_drop_last()
    check_kfold_partition()
    print("all data checks passed")
