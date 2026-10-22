"""Which image each sample of a cached dataset is.

Caches (np/X_tr_*.npy, np/Y_tr_*.npy) built by older code follow the unsorted file listing, so the
order is stored next to them in np/ids_<size>x<size>.txt (one image id per line, cache order). It
is written whenever a cache is built and was recovered for older caches with
scripts/check_cache_order.py. Without the file, sorted file-name order is assumed.
"""
import glob
import os


def ids_path(npy_dir, image_size):
    return os.path.join(npy_dir, f"ids_{image_size}x{image_size}.txt")


def read_cache_ids(npy_dir, image_size, imgs_dir, input_fex):
    path = ids_path(npy_dir, image_size)
    if os.path.isfile(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(f"{imgs_dir}/*.{input_fex}"))]


def write_cache_ids(npy_dir, image_size, ids):
    with open(ids_path(npy_dir, image_size), "w") as f:
        f.write("\n".join(ids) + "\n")
