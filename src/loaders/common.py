"""Shared wiring for the existing data protocol (not a new split policy)."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.kfold import get_dev_split, get_inner_split, get_kfold_indices, get_group_kfold_indices


def read_groups(ids, data_dir, group_cfg):
    """Group key of every sample id, from a metadata CSV (run.kfold.group_by).

    Samples with an empty group value form their own group.
    """
    path = Path(group_cfg["file"])
    if not path.is_absolute():
        path = Path(data_dir) / path
    id_column = group_cfg.get("id_column", "image_id")
    with open(path, newline="") as f:
        groups = {row[id_column]: row[group_cfg["group_column"]] for row in csv.DictReader(f)}
    missing = [i for i in ids if i not in groups]
    if missing:
        raise ValueError(f"{len(missing)} samples are not in {path}, e.g. {missing[:3]}")
    return [groups[i] or f"__single__{i}" for i in ids]


def prepare_kfold_data(preparer, config, logger=None):
    prep = preparer(data_dir=config["dataset"]["data_dir"],
                    image_size=config["dataset"]["input_size"], logger=logger)
    data = prep.get_data()
    shared = {"x": torch.tensor(data["x"]), "y": torch.tensor(data["y"])}
    cfg = config.get("run", {}).get("kfold", {})
    split = dict(n_splits=cfg.get("n_splits", 5), fold_index=cfg.get("fold", 0),
                 shuffle=cfg.get("shuffle", True), seed=cfg.get("seed", 0))
    if cfg.get("all_test"):
        # every image is a test image: models trained on another dataset (scripts/evaluate_per_case.py --eval_on)
        return shared, np.array([], dtype=int), np.arange(shared["x"].shape[0])
    group_cfg = cfg.get("group_by")
    groups = None
    if group_cfg:
        ids = prep.get_ids()
        if len(ids) != shared["x"].shape[0]:
            raise ValueError(f"{len(ids)} image files but {shared['x'].shape[0]} cached samples")
        groups = read_groups(ids, config["dataset"]["data_dir"], group_cfg)
        train_idx, val_idx = get_group_kfold_indices(groups, **split)
        if logger:
            logger.info(f"group k-fold by {group_cfg['group_column']}: {len(set(groups))} groups, "
                        f"{len(train_idx)} train / {len(val_idx)} held-out samples")
    else:
        train_idx, val_idx = get_kfold_indices(shared["x"].shape[0], **split)
    if cfg.get("dev_fraction"):
        # development mode: train on part of the fold's training set and hold out the rest
        # (development-validation) in place of the fold's test set, which is not used at all
        train_idx, val_idx = get_dev_split(train_idx, cfg["dev_fraction"], split["seed"], groups)
        if logger:
            logger.info(f"development split ({cfg['dev_fraction']} of the training set): "
                        f"{len(train_idx)} train / {len(val_idx)} development-validation samples; "
                        f"the fold's test set is not used")
    if cfg.get("inner_fold") is not None:
        # inner cross-validation within the (development) training set, for out-of-fold priors
        n_inner = cfg.get("inner_n_splits", 5)
        train_idx, val_idx = get_inner_split(train_idx, n_inner, cfg["inner_fold"], split["seed"], groups)
        if logger:
            logger.info(f"inner fold {cfg['inner_fold']}/{n_inner}: {len(train_idx)} train / "
                        f"{len(val_idx)} inner-validation samples")
    return shared, train_idx, val_idx


def build_loaders(config, train, validation, test):
    return {mode: {"dataset": dataset,
                   "loader": DataLoader(dataset, **config["data_loader"][key])}
            for mode, key, dataset in [("tr", "train", train),
                                       ("vl", "validation", validation),
                                       ("te", "test", test)] if dataset is not None}
