from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import MLFFMSeg  # noqa: E402
from seg_training import compute_loss  # noqa: E402


def _small_model():
    return MLFFMSeg(
        dim_x=8,
        dim_g=8,
        channels_x=1,
        channels_g=3,
        out_dim=1,
        dim_x_mults=[1, 1, 1, 1, 1, 1],
        dim_g_mults=[1, 1, 1, 1, 1, 1],
        resnet_block_groups=1,
    )


def test_direct_mask_forward_backward_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = _small_model()
    image = torch.randn(1, 3, 64, 64)
    target = (torch.rand(1, 1, 64, 64) > 0.5).float()

    logits = model(image)
    assert logits.shape == target.shape
    loss, terms = compute_loss(
        logits,
        target,
        {"bce": {"coefficient": 1.0}, "dice": {"coefficient": 1.0}},
    )
    assert set(terms) == {"bce", "dice", "total"}
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    checkpoint_path = tmp_path / "nondiff.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    restored = _small_model()
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"], strict=True)
    restored.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(image), model(image))


def test_direct_model_has_no_diffusion_time_embedding():
    model = _small_model()
    assert not hasattr(model, "time_mlp_x")
    assert not hasattr(model, "time_mlp_g")


def test_nondiffusion_train_and_test(tmp_path):
    """Run the PH2 non-diffusion config of the paper (small widths) on synthetic data."""
    import os
    import subprocess
    import numpy as np
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/paper/ph2/diffusion_vs_nondiffusion/nondiffusion.yaml").read_text())
    cfg["run"].update(device="cpu", writer_dir=str(tmp_path / "events"), log_dir=str(tmp_path / "logs"))
    cfg["run"]["kfold"] = {"enabled": True, "n_splits": 5, "fold": 0, "seed": 0, "metrics_dir": str(tmp_path / "metrics")}
    cfg["model"]["save_dir"] = str(tmp_path / "models")
    cfg["model"]["params"].update(dim_x=8, dim_g=8, dim_x_mults=[1] * 6, dim_g_mults=[1] * 6, resnet_block_groups=1)
    cfg["dataset"]["data_dir"] = str(tmp_path / "data")
    cfg["augmentation"]["p"] = 0
    cfg["training"]["epochs"] = 2
    for section in cfg["data_loader"].values():
        section.update(batch_size=2, num_workers=0, pin_memory=False)
    cache = tmp_path / "data/np"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(4)
    np.save(cache / "X_tr_128x128.npy", rng.random((10, 3, 128, 128), dtype=np.float32))
    np.save(cache / "Y_tr_128x128.npy", (rng.random((10, 1, 128, 128)) > .5).astype(np.float32))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for entry in ("seg_training", "seg_testing"):
        result = subprocess.run([sys.executable, str(ROOT / "src" / f"{entry}.py"), "-c", str(path)],
                                cwd=tmp_path, capture_output=True, text=True, env=env, timeout=300)
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]

    checkpoint = torch.load(next((tmp_path / "models").glob("*_last.pth")), map_location="cpu")
    assert checkpoint["ema"] is None and checkpoint["epoch"] == 1
    assert (tmp_path / "metrics/mlffm_nodiff_formal_metrics.json").exists()
