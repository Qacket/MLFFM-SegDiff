from datasets.isic2018 import ISIC2018DatasetFast, PrepareISIC2018
from modules.transforms import DiffusionTransform, SegmentationTransform, DataAugmentationTransform
import albumentations as A
import glob
from loaders.common import prepare_kfold_data, build_loaders
from datasets.prior import load_priors


def get_isic2018(config, logger=None, verbose=False, task="diffusion", modes=("tr", "vl", "te")):

    if logger: print = logger.info

    INPUT_SIZE = config["dataset"]["input_size"]
    task = (task or "diffusion").lower()
    if task in ("seg", "segmentation"):
        DT = SegmentationTransform((INPUT_SIZE, INPUT_SIZE))
    else:
        DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))
    AUGT = DataAugmentationTransform((INPUT_SIZE, INPUT_SIZE))

    img_dir = "ISIC2018_Task1-2_Training_Input"
    msk_dir = "ISIC2018_Task1_Training_GroundTruth"
    img_path_list = glob.glob(f"{config['dataset']['data_dir']}/{img_dir}/ISIC*.jpg")

    pixel_level_transform = AUGT.get_pixel_level_transform(config["augmentation"], img_path_list=img_path_list)
    spacial_level_transform = AUGT.get_spacial_level_transform(config["augmentation"])
    tr_aug_transform = A.Compose([
        A.Compose(pixel_level_transform, p=config["augmentation"]["levels"]["pixel"]["p"]),
        A.Compose(spacial_level_transform, p=config["augmentation"]["levels"]["spacial"]["p"])
    ], p=config["augmentation"]["p"])

    # ----------------- dataset --------------------
    kfold_cfg = config.get("run", {}).get("kfold", {})
    use_kfold = bool(kfold_cfg.get("enabled"))
    if config["dataset"]["class_name"] == "ISIC2018DatasetFast":
        if use_kfold:
            shared_data, train_idx, val_idx = prepare_kfold_data(PrepareISIC2018, config, logger)
            prior_cfg = config["dataset"].get("prior")
            priors = load_priors(prior_cfg["file"], shared_data["x"].shape[0]) if prior_cfg else None
            prior_kwargs = {"priors": priors}
            tr_dataset = ISIC2018DatasetFast(
                mode="tr",
                data=shared_data,
                indices=train_idx,
                **prior_kwargs,
                prior_perturb=prior_cfg.get("perturb") if prior_cfg else None,
                one_hot=False,
                image_size=config["dataset"]["input_size"],
                aug=tr_aug_transform,
                img_transform=DT.get_forward_transform_img(),
                msk_transform=DT.get_forward_transform_msk(),
                add_boundary_mask=config["dataset"]["add_boundary_mask"],
                add_boundary_dist=config["dataset"]["add_boundary_dist"],
                logger=logger,
                data_scale=config["dataset"].get("data_scale", "full"),
            ) if "tr" in modes else None
            vl_dataset = ISIC2018DatasetFast(
                mode="vl",
                data=shared_data,
                indices=val_idx,
                **prior_kwargs,
                one_hot=False,
                image_size=config["dataset"]["input_size"],
                img_transform=DT.get_forward_transform_img(),
                msk_transform=DT.get_forward_transform_msk(),
                add_boundary_mask=config["dataset"]["add_boundary_mask"],
                add_boundary_dist=config["dataset"]["add_boundary_dist"],
                logger=logger,
                data_scale=config["dataset"].get("data_scale", "full"),
            ) if "vl" in modes else None
            te_dataset = ISIC2018DatasetFast(
                mode="te",
                data=shared_data,
                indices=val_idx,
                **prior_kwargs,
                one_hot=False,
                image_size=config["dataset"]["input_size"],
                img_transform=DT.get_forward_transform_img(),
                msk_transform=DT.get_forward_transform_msk(),
                add_boundary_mask=config["dataset"]["add_boundary_mask"],
                add_boundary_dist=config["dataset"]["add_boundary_dist"],
                logger=logger,
                data_scale=config["dataset"].get("data_scale", "full"),
            ) if "te" in modes else None
        else:
            # preparing training dataset
            tr_dataset = ISIC2018DatasetFast(
                mode="tr",
                data_dir=config["dataset"]["data_dir"],
                one_hot=False,
                image_size=config["dataset"]["input_size"],
                aug=tr_aug_transform,
                img_transform=DT.get_forward_transform_img(),
                msk_transform=DT.get_forward_transform_msk(),
                add_boundary_mask=config["dataset"]["add_boundary_mask"],
                add_boundary_dist=config["dataset"]["add_boundary_dist"],
                logger=logger,
                data_scale=config["dataset"].get("data_scale", "full")
            ) if "tr" in modes else None
            vl_dataset = ISIC2018DatasetFast(
                mode="vl",
                data_dir=config["dataset"]["data_dir"],
                one_hot=False,
                image_size=config["dataset"]["input_size"],
                img_transform=DT.get_forward_transform_img(),
                msk_transform=DT.get_forward_transform_msk(),
                add_boundary_mask=config["dataset"]["add_boundary_mask"],
                add_boundary_dist=config["dataset"]["add_boundary_dist"],
                logger=logger,
                data_scale=config["dataset"].get("data_scale", "full")
            ) if "vl" in modes else None
            te_dataset = ISIC2018DatasetFast(
                mode="te",
                data_dir=config["dataset"]["data_dir"],
                one_hot=False,
                image_size=config["dataset"]["input_size"],
                img_transform=DT.get_forward_transform_img(),
                msk_transform=DT.get_forward_transform_msk(),
                add_boundary_mask=config["dataset"]["add_boundary_mask"],
                add_boundary_dist=config["dataset"]["add_boundary_dist"],
                logger=logger,
                data_scale=config["dataset"].get("data_scale", "full")
            ) if "te" in modes else None

    else:
        message = "In the config file, `dataset>class_name` should be in: ['ISIC2018DatasetFast']"
        if logger:
            logger.exception(message)
        else:
            raise ValueError(message)

    if verbose and all(d is not None for d in (tr_dataset, vl_dataset, te_dataset)):
        print("ISIC 2018:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t   {len(te_dataset)}")

    return build_loaders(config, tr_dataset, vl_dataset, te_dataset)
