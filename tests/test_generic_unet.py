import collections
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seg_models import MODELS  # noqa: E402
from seg_models.generic_unet import Generic_UNet  # noqa: E402


def layer_counts(model):
    return collections.Counter(type(m).__name__ for m in model.modules())


def test_generic_unet_uses_nnunet_training_settings():
    params = yaml.safe_load((ROOT / "configs/paper/isic2018/generic_unet.yaml").read_text())["model"]["params"]
    model = MODELS["GenericUNet"](**params)
    counts = layer_counts(model)
    assert counts["Dropout2d"] == 0 and counts["BatchNorm2d"] == 0 and counts["MaxPool2d"] == 0
    assert counts["InstanceNorm2d"] > 0 and counts["ConvTranspose2d"] == params["num_pool"]
    out = model(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 1, 128, 128)
    # the raw nnU-Net class keeps its constructor defaults
    raw = layer_counts(Generic_UNet(**params))
    assert raw["Dropout2d"] > 0 and raw["BatchNorm2d"] > 0 and raw["MaxPool2d"] == params["num_pool"]
