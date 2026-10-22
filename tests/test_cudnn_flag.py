import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cudnn", [True, None])
def test_seg_entrypoints_follow_run_cudnn(cudnn, tmp_path):
    """run.cudnn re-enables cuDNN for direct segmentation (importing the MLFFM models disables it)."""
    cfg = yaml.safe_load((ROOT / "configs/paper/isic2018/unet.yaml").read_text())
    cfg["run"].update(device="cpu", writer_dir=str(tmp_path / "runs"), log_dir=str(tmp_path / "logs"))
    if cudnn is not None:
        cfg["run"]["cudnn"] = cudnn
    cfg["run"]["kfold"] = {"enabled": True, "n_splits": 5, "fold": 0, "seed": 0, "metrics_dir": str(tmp_path / "metrics")}
    cfg["model"]["save_dir"] = str(tmp_path / "models")
    cfg["model"]["params"]["features"] = [8, 16]
    cfg["dataset"]["data_dir"] = str(tmp_path / "data")
    cfg["augmentation"]["p"] = 0
    cfg["training"]["epochs"] = 1
    for section in cfg["data_loader"].values():
        section.update(batch_size=2, num_workers=0, pin_memory=False)
    (tmp_path / "data/np").mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(tmp_path / "data/np/X_tr_128x128.npy", rng.random((10, 3, 128, 128), dtype=np.float32))
    np.save(tmp_path / "data/np/Y_tr_128x128.npy", (rng.random((10, 1, 128, 128)) > .5).astype(np.float32))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    for entry in ("seg_training", "seg_testing"):
        result = subprocess.run([sys.executable, str(ROOT / "src" / f"{entry}.py"), "-c", str(path)], cwd=tmp_path,
                                capture_output=True, text=True, timeout=300, env=dict(os.environ, OMP_NUM_THREADS="1"))
        assert result.returncode == 0, result.stderr[-3000:]
    expected = "cuDNN: on" if cudnn else "cuDNN: off"
    for log in (tmp_path / "logs").glob("*.log"):
        assert expected in log.read_text(), log.name
