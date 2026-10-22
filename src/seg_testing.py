import argparse
import json
from pathlib import Path
import torch

from common.logging import get_logger
from loaders.dataloaders import get_dataloaders
from metrics import get_binary_metrics
from seg_models import MODELS
from utils.config import load_config, print_config, get_model_path


def get_argparser():
    parser = argparse.ArgumentParser(
        prog="SegmentationTesting",
        description="Segmentation baselines testing",
        epilog="",
    )
    parser.add_argument("-c", "--config_file", type=str, required=True, help="")
    return parser


def prepare_targets(msks):
    if msks.shape[1] > 1:
        msks = torch.argmax(msks, dim=1, keepdim=True)
    return msks.float().clamp(0, 1)


@torch.no_grad()
def main():
    argparser = get_argparser()
    args = argparser.parse_args()

    config = load_config(args.config_file)
    logger = get_logger(
        filename=f"{config['model']['name']}_seg_test", dir=config.get("run", {}).get("log_dir", f"logs/{config['dataset']['name']}")
    )
    print_config(config, logger)

    device = torch.device(config["run"]["device"])
    logger.info(f"Device is <{device}>")
    torch.backends.cudnn.enabled = bool(config.get("run", {}).get("cudnn", False))  # importing models/mlffm_segdiff.py disables it
    logger.info(f"cuDNN: {'on' if torch.backends.cudnn.enabled else 'off'}")

    te_dataloader = get_dataloaders(config, "te", task="seg")

    model_cls = MODELS[config["model"]["class"]]
    model = model_cls(**config["model"]["params"]).to(device)

    testing_cfg = config.get("testing", {})
    weights_cfg = testing_cfg.get("model_weights", testing_cfg.get("model_weigths", {}))
    if weights_cfg.get("overload"):
        model_path = weights_cfg.get("file_path")
    else:
        model_path = get_model_path(name=config["model"]["name"], dir=config["model"]["save_dir"])

    if not model_path:
        raise ValueError("model weights path is empty")

    checkpoint = torch.load(model_path, map_location="cpu")
    ema_cfg = config.get("training", {}).get("ema", {})
    if ema_cfg.get("use"):
        from ema_pytorch import EMA
        ema = EMA(model=model, **ema_cfg["params"])
        ema.load_state_dict(checkpoint["ema"])
        model = ema.ema_model
        logger.info("ema loaded...")
    else:
        model.load_state_dict(checkpoint["model"])
        logger.info("simple model loaded...")
    model.to(device)
    model.eval()

    metrics = get_binary_metrics()
    for batch in te_dataloader:
        imgs = batch["image"].to(device)
        msks = prepare_targets(batch["mask"].to(device))
        logits = model(imgs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        preds = (torch.sigmoid(logits) > 0.5).int()
        targets = (msks > 0.5).int()
        metrics.update(preds, targets)

    result = metrics.compute()
    result_dict = {k: v.item() for k, v in result.items()}
    logger.info(json.dumps(result_dict, indent=4))

    metrics_dir = config.get("run", {}).get("kfold", {}).get("metrics_dir")
    if metrics_dir:
        Path(metrics_dir).mkdir(exist_ok=True, parents=True)
        metrics_path = Path(metrics_dir) / f"{config['model']['name']}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    main()
