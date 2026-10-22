"""Segmentation priors for prior-guided refinement (dataset.prior in the config).

The prior of an image is the score map of a model that did not train on it (scripts/build_priors.py).
It is appended to the image as an extra channel, so the model gets channels_g = 3 + 1.
"""
import numpy as np
import torch
import torch.nn.functional as F


def load_priors(path, n_samples):
    """(n_samples, H, W) float tensor; images without a prior are NaN."""
    data = np.load(path)
    index, prior = data["index"], data["prior"].astype(np.float32)
    if index.max() >= n_samples:
        raise ValueError(f"{path}: index {index.max()} out of range for {n_samples} samples")
    priors = torch.full((n_samples, *prior.shape[1:]), float("nan"))
    priors[torch.as_tensor(index)] = torch.as_tensor(prior)
    return priors


def perturb_prior(prior, p=0.5, max_radius=5, p_box=0.25, max_box=0.4):
    """Random corruption of a (H, W) prior in [0, 1] during training, so the model has to check the
    prior against the image instead of copying it: with probability p dilate or erode by 1..max_radius
    px; with probability p_box clear or fill a random box of up to max_box of the image side."""
    x = prior[None, None]
    if torch.rand(()) < p:
        r = int(torch.randint(1, max_radius + 1, ()))
        if torch.rand(()) < 0.5:
            x = F.max_pool2d(x, 2 * r + 1, stride=1, padding=r)
        else:
            x = -F.max_pool2d(-x, 2 * r + 1, stride=1, padding=r)
    if torch.rand(()) < p_box:
        h, w = prior.shape
        bh = int(torch.randint(1, max(2, int(max_box * h)) + 1, ()))
        bw = int(torch.randint(1, max(2, int(max_box * w)) + 1, ()))
        top = int(torch.randint(0, h - bh + 1, ()))
        left = int(torch.randint(0, w - bw + 1, ()))
        x = x.clone()
        x[..., top:top + bh, left:left + bw] = float(torch.rand(()) < 0.5)
    return x[0, 0]
