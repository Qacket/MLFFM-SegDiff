from pathlib import Path
import numpy as np
import random
import torch
from torch.optim import Adam, SGD, AdamW
from utils.config import load_config, get_model_path, get_conf_name, print_config
from models import MODELS
from forward.forward_schedules import ForwardSchedule
from forward.forward_process import ForwardProcess
from train_validate import train
from loaders.dataloaders import get_dataloaders
from torch.utils.tensorboard import SummaryWriter
import os
from common.logging import get_logger
from argument import get_argparser, sync_config
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_training(config):
    # ------------------- params --------------------
    print(config)
    logger = get_logger(filename=f"{config['model']['name']}", dir=config.get("run", {}).get("log_dir", f"logs/{config['dataset']['name']}"))
    print_config(config, logger)

    # create the writer for tensorboard
    writer = SummaryWriter(f'{config["run"]["writer_dir"]}/{config["model"]["name"]}')

    # variables
    timesteps = config["diffusion"]["schedule"]["timesteps"]
    epochs = config["training"]["epochs"]
    input_size = config["dataset"]["input_size"]
    batch_size = config["data_loader"]["train"]["batch_size"]
    img_channels = config["dataset"]["img_channels"]
    msk_channels = config["dataset"]["msk_channels"]
    ID = get_conf_name(config)

    # device
    device = torch.device(config["run"]["device"])
    logger.info(f"Device is <{device}>")


    start_epoch = 0

    set_seed(config.get("run", {}).get("seed", 0))


    # --------- check required dirs --------------------
    Path(config["model"]["save_dir"]).mkdir(exist_ok=True, parents=True)


    forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])
    forward_process = ForwardProcess(forward_schedule)

    # --------------- Datasets and Dataloaders -----------------
    tr_dataloader = get_dataloaders(config, "tr")


    Net = MODELS[config["model"]["class"]]
    model = Net(**config["model"]["params"])

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of model parameters: {total_params}")


    tr_prms = config["training"]
    optimizer = {"Adam": Adam, "SGD": SGD, "AdamW": AdamW}[tr_prms["optimizer"]["name"]](
        model.parameters(), **tr_prms["optimizer"]["params"]
    )
    logger.info("Protocol: fixed epochs, fixed learning rate, last-epoch checkpoint; no validation/test during training.")

    # ------------------------ EMA -------------------------------
    # https://github.com/lucidrains/ema-pytorch
    from ema_pytorch import EMA

    try:
        if config["training"]["ema"]["use"]:
            ema = EMA(model=model, **config["training"]["ema"]["params"])
            ema.to(device)
        else:
            ema = None
    except KeyError:
        logger.exception("You need to determine the EMA parameters at <config.training>!")


    if config["run"]["continue_training"] or config["training"]["intial_weights"]["use"]:
        if config["run"]["continue_training"]:
            model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
        else:
            model_path = config["training"]["intial_weights"]["file_path"]
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            if config["run"]["continue_training"]:
                if epochs > checkpoint["epoch"] + 1:
                    model.load_state_dict(checkpoint["model"])
                    start_epoch = checkpoint["epoch"] + 1
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    for group in optimizer.param_groups:
                        group["lr"] = tr_prms["optimizer"]["params"]["lr"]
                    if ema:
                        ema.load_state_dict(checkpoint["ema"])

                    logger.info(f"Loaded the model state (ep:{checkpoint['epoch']+1}/{checkpoint['epochs']}) to continue training from the following path:")
                    logger.info(f" -> {model_path}\n")
                else:
                    logger.warning("the net already trained!")
                    writer.close()
                    return
            else:
                model.load_state_dict(checkpoint["model"])
                if ema:
                    ema = EMA(model=model, **config["training"]["ema"]["params"])
                    ema.to(device)
                logger.info(f"Loaded initial weights")
                logger.info(f" -> {model_path}\n")

        except:
            logger.warning("There is a problem with loading the previous model to continue training.")
            logger.warning(" --> Do you want to train the model from the beginning? (y/N):")
            user_decision = input()
            if (user_decision != "y"):
                exit()
    else:
        model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
        if os.path.isfile(model_path):
            logger.warning(f"There is a model weights at determind directory with desired name: {ID}")
            logger.warning(" --> Do you want to train the model from the beginning? It will overwrite the current weights! (y/N):")
            user_decision = input()
            if (user_decision != "y"):
                exit()


    save_every = config["training"].get("save_every", 1)
    keep_every = config["training"].get("keep_every")  # extra checkpoint per keep_every epochs, kept separately
    for epoch in range(start_epoch, epochs):

        tr_losses, model = train(
            model,
            tr_dataloader,
            forward_process,
            device,
            optimizer,
            ema=ema,
            cfg=config,
            extra={"skip_steps": 10, "prefix": f"ep:{epoch+1}/{epochs}"},
            logger=logger
        )

        tr_loss = float(np.mean([l[0] for l in tr_losses]))
        tr_losses_dict = {
            n: float(np.mean([d[1][n] for d in tr_losses]))
            for n in tr_losses[0][1]
        }
        writer.add_scalars("Loss/train", tr_losses_dict, epoch)
        logger.info(f"ep:{epoch+1}/{epochs}, tr_loss: {tr_loss:.8f}")
        keep = bool(keep_every) and (epoch + 1) % keep_every == 0
        if (epoch + 1) % save_every != 0 and epoch + 1 != epochs and not keep:
            continue
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "epochs": epochs,
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "vl_loss": None,
            "selection_policy": "last_epoch",
            "training_loss": tr_loss,
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Saved last-epoch checkpoint to {model_path}")
        if keep:
            snapshot = model_path.replace("_last.pth", f"_ep{epoch + 1:04d}.pth")
            torch.save(checkpoint, snapshot)
            logger.info(f"Kept snapshot {snapshot}")


    writer.flush()
    writer.close()


def main():
    args = get_argparser().parse_args()
    config = sync_config(load_config(args.config_file), args)
    run_training(config)


if __name__ == "__main__":
    main()
