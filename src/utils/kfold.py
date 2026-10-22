import numpy as np


def get_kfold_indices(n_samples, n_splits=5, fold_index=0, shuffle=True, seed=0):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError("fold_index out of range")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)
    val_idx = folds[fold_index]
    train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_index])
    return train_idx, val_idx


def get_group_kfold_indices(groups, n_splits=5, fold_index=0, shuffle=True, seed=0):
    """Like get_kfold_indices, but all samples of a group land in the same fold.

    groups[i] is the group key (e.g. lesion id) of sample i. Unique groups are shuffled
    with `seed` and split into n_splits parts, so fold sizes differ slightly.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError("fold_index out of range")

    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_groups)

    val_groups = np.array_split(unique_groups, n_splits)[fold_index]
    in_val = np.isin(groups, val_groups)
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)


def get_dev_split(train_idx, fraction, seed=0, groups=None):
    """Split a fold's training indices into (development-train, development-validation).

    The development-validation set is used for method development instead of the fold's test
    set, which stays untouched. With `groups`, whole groups go to one side. The draw uses its own
    random stream (seed, 1), independent of the fold permutation, and does not depend on the
    order of train_idx.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    train_idx = np.sort(train_idx)
    rng = np.random.default_rng([seed, 1])
    if groups is None:
        order = rng.permutation(train_idx)
        n_dev = int(round(fraction * len(train_idx)))
        dev = np.sort(order[:n_dev])
    else:
        groups = np.asarray(groups)
        train_groups = np.unique(groups[train_idx])
        rng.shuffle(train_groups)
        dev_groups = train_groups[:int(round(fraction * len(train_groups)))]
        dev = np.sort(train_idx[np.isin(groups[train_idx], dev_groups)])
    rest = np.sort(np.setdiff1d(train_idx, dev))
    return rest, dev


def get_inner_split(train_idx, n_splits, fold_index, seed=0, groups=None):
    """Split a training set into (inner-train, inner-validation) for inner cross-validation.

    Used to predict every training image with a model that did not see it (out-of-fold priors).
    The inner folds of all fold_index values partition train_idx; with `groups`, whole groups
    stay in one inner fold. Uses its own random stream (seed, 2) and does not depend on the order
    of train_idx.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError("fold_index out of range")
    train_idx = np.sort(train_idx)
    rng = np.random.default_rng([seed, 2])
    if groups is None:
        val = np.array_split(rng.permutation(train_idx), n_splits)[fold_index]
    else:
        groups = np.asarray(groups)
        train_groups = np.unique(groups[train_idx])
        rng.shuffle(train_groups)
        val_groups = np.array_split(train_groups, n_splits)[fold_index]
        val = train_idx[np.isin(groups[train_idx], val_groups)]
    val = np.sort(val)
    return np.setdiff1d(train_idx, val), val
