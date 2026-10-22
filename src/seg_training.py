import argparse
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam, SGD, AdamW
from torch.utils.tensorboard import SummaryWriter

from common.logging import get_logger
from loaders.dataloaders import get_dataloaders
from metrics import get_binary_metrics
from seg_models import MODELS
from utils.config import load_config, print_config, get_model_path

import warnings
warnings.filterwarnings("ignore")
def get_argparser():
    parser = argparse.ArgumentParser(
        prog="SegmentationTraining",
        description="Segmentation baselines training",
        epilog="",
    )
    parser.add_argument("-c", "--config_file", type=str, required=True, help="")
    return parser


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def _coeff(cfg):
    return cfg.get("coefficient", cfg.get("cofficient", 1.0))


def compute_loss(logits, targets, loss_cfg):
    loss = 0.0
    losses = {}

    if "bce" in loss_cfg:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        losses["bce"] = bce
        loss += _coeff(loss_cfg["bce"]) * bce
    if "dice" in loss_cfg:
        dsc = dice_loss(logits, targets)
        losses["dice"] = dsc
        loss += _coeff(loss_cfg["dice"]) * dsc

    if not losses:
        raise ValueError("loss config must contain at least one of: bce, dice")

    losses["total"] = loss
    return loss, losses


def prepare_targets(msks):
    if msks.shape[1] > 1:
        msks = torch.argmax(msks, dim=1, keepdim=True)
    return msks.float().clamp(0, 1)


def train_one_epoch(model, dataloader, device, optimizer, loss_cfg, logger=None, ema=None):
    model.train()
    losses = []
    for batch in dataloader:
        imgs = batch["image"].to(device)
        msks = prepare_targets(batch["mask"].to(device))

        optimizer.zero_grad()
        logits = model(imgs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        loss, losses_dict = compute_loss(logits, msks, loss_cfg)
        loss.backward()
        optimizer.step()
        if ema:
            ema.update()

        losses.append(losses_dict)

    mean_losses = {
        k: float(np.mean([l[k].item() for l in losses if k in l]))
        for k in losses[0].keys()
    }
    if logger:
        logger.info("train losses: " + ", ".join([f"{k}={v:0.6f}" for k, v in mean_losses.items()]))
    return mean_losses


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, loss_cfg, logger=None):
    model.eval()
    losses = []
    metrics = get_binary_metrics()

    for batch in dataloader:
        imgs = batch["image"].to(device)
        msks = prepare_targets(batch["mask"].to(device))

        logits = model(imgs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        loss, losses_dict = compute_loss(logits, msks, loss_cfg)
        losses.append(losses_dict)

        preds = (torch.sigmoid(logits) > 0.5).int()
        targets = (msks > 0.5).int()
        metrics.update(preds, targets)

    mean_losses = {
        k: float(np.mean([l[k].item() for l in losses if k in l]))
        for k in losses[0].keys()
    }
    metrics_out = metrics.compute()
    if logger:
        logger.info(
            "val losses: " + ", ".join([f"{k}={v:0.6f}" for k, v in mean_losses.items()])
        )
        logger.info(
            "val metrics: " + ", ".join([f"{k}={v.item():0.6f}" for k, v in metrics_out.items()])
        )
    return mean_losses, metrics_out


def main():
    argparser = get_argparser()
    args = argparser.parse_args()

    config = load_config(args.config_file)
    logger = get_logger(filename=f"{config['model']['name']}_seg", dir=config.get("run", {}).get("log_dir", f"logs/{config['dataset']['name']}"))
    print_config(config, logger)

    set_seed(config.get("run", {}).get("seed", 0))

    device = torch.device(config["run"]["device"])
    logger.info(f"Device is <{device}>")
    torch.backends.cudnn.enabled = bool(config.get("run", {}).get("cudnn", False))  # importing models/mlffm_segdiff.py disables it
    logger.info(f"cuDNN: {'on' if torch.backends.cudnn.enabled else 'off'}")

    Path(config["model"]["save_dir"]).mkdir(exist_ok=True, parents=True)

    tr_dataloader = get_dataloaders(config, "tr", task="seg")

    model_cls = MODELS[config["model"]["class"]]
    model = model_cls(**config["model"]["params"]).to(device)

    optimizer = {"Adam": Adam, "SGD": SGD, "AdamW": AdamW}[config["training"]["optimizer"]["name"]](
        model.parameters(), **config["training"]["optimizer"]["params"]
    )
    logger.info("Protocol: fixed epochs, fixed learning rate, last-epoch checkpoint; no validation/test during training.")

    # same EMA as the diffusion entrypoint (training.py); off unless training.ema.use is set
    ema = None
    ema_cfg = config["training"].get("ema", {})
    if ema_cfg.get("use"):
        from ema_pytorch import EMA
        ema = EMA(model=model, **ema_cfg["params"])
        ema.to(device)
    logger.info(f"EMA: {'on' if ema else 'off'}")

    writer = SummaryWriter(f'{config["run"]["writer_dir"]}/{config["model"]["name"]}')

    epochs = config["training"]["epochs"]
    loss_cfg = config["training"]["loss"]
    model_path = get_model_path(name=config["model"]["name"], dir=config["model"]["save_dir"])

    if Path(model_path).exists():
        writer.close()
        raise FileExistsError(f"Refusing to overwrite {model_path}; use a new experiment directory.")

    save_every = config["training"].get("save_every", 1)
    for epoch in range(epochs):
        tr_losses = train_one_epoch(
            model, tr_dataloader, device, optimizer, loss_cfg, logger=logger, ema=ema
        )
        writer.add_scalars("Loss/train", tr_losses, epoch)
        if (epoch + 1) % save_every != 0 and epoch + 1 != epochs:
            continue
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "epochs": epochs,
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "vl_loss": None,
            "selection_policy": "last_epoch",
            "training_loss": tr_losses["total"],
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Saved last-epoch checkpoint to {model_path}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
