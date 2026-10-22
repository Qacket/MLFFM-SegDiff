import os
from datasets.cache_ids import read_cache_ids, write_cache_ids
from datasets.prior import perturb_prior
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F
from PIL import Image
from utils.boundary import calc_edge, calc_distance_map
from utils.tensors import normalize

np_normalize = lambda x: (x-x.min())/(x.max()-x.min())


class PH2DatasetFast(Dataset):
    def __init__(self,
                 mode,
                 data_dir=None,
                 one_hot=True,
                 image_size=224,
                 indices=None,
                 data=None,
                 aug=None,
                 aug_empty=None,
                 transform=None,
                 img_transform=None,
                 msk_transform=None,
                 add_boundary_mask=False,
                 add_boundary_dist=False,
                 logger=None,
                 priors=None,
                 prior_perturb=None,
                 **kwargs):
        self.print=logger.info if logger else print

        # pre-set variables
        self.data_dir = data_dir if data_dir else "data/ph2"

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.aug = aug
        self.aug_empty = aug_empty
        self.transform = transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode

        self.add_boundary_mask = add_boundary_mask
        self.add_boundary_dist = add_boundary_dist
        # dataset.prior, as in datasets/isic2018.py
        self.priors = priors
        self.prior_perturb = prior_perturb


        if data is None:
            data_preparer = PreparePH2(
                data_dir=self.data_dir, image_size=self.image_size, logger=logger
            )
            data = data_preparer.get_data()
        X, Y = data["x"], data["y"]

        if not torch.is_tensor(X):
            X = torch.tensor(X)
        if not torch.is_tensor(Y):
            Y = torch.tensor(Y)

        self.imgs = X
        self.msks = Y

        if indices is not None:
            self.indices = np.array(indices)
        else:
            if mode == "tr":
                self.indices = np.arange(0, 80)
            elif mode == "vl":
                self.indices = np.arange(80, 80 + 20)
            elif mode == "te":
                self.indices = np.arange(80 + 20, 200)
            elif mode == "all":
                self.indices = np.arange(len(self.imgs))
            else:
                raise ValueError()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        data_id = real_idx
        img = self.imgs[real_idx]
        msk = self.msks[real_idx]
        prior = None
        if self.priors is not None:
            prior = self.priors[real_idx]
            if torch.isnan(prior).any():
                raise ValueError(f"no prior for sample {real_idx}")

        if self.one_hot:
            msk = (msk - msk.min()) / (msk.max() - msk.min())
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        if self.aug:
            if self.mode == "tr":
                img_ = np.uint8(torch.moveaxis(img*255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy())
                if prior is not None:  # spatial transforms move the prior with the mask
                    msk_ = np.concatenate([msk_, np.uint8(np.round(prior.numpy()[..., None] * 255))], axis=-1)
                augmented = self.aug(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
                if prior is not None:
                    msk, prior = msk[:-1], msk[-1] / 255
            elif self.aug_empty: # "tr", "vl", "te"
                img_ = np.uint8(torch.moveaxis(img*255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy())
                augmented = self.aug_empty(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
            img = img.nan_to_num(127)
            img = normalize(img)
            msk = msk.nan_to_num(0)
            msk = normalize(msk)


        if self.add_boundary_mask or self.add_boundary_dist:
            msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy())

        if self.add_boundary_mask:
            boundary_mask = calc_edge(msk_, mode='canny')
            boundary_mask = np_normalize(boundary_mask)
            msk = torch.concatenate([msk, torch.tensor(boundary_mask).unsqueeze(0)], dim=0)

        if self.add_boundary_dist:
            boundary_mask = boundary_mask if self.add_boundary_mask else calc_edge(msk_, mode='canny')
            distance_map = calc_distance_map(boundary_mask, mode='l2')
            distance_map = np_normalize(distance_map)
            msk = torch.concatenate([msk, torch.tensor(distance_map).unsqueeze(0)], dim=0)


        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform:
            msk = self.msk_transform(msk)

        img = img.nan_to_num(0.5)
        msk = msk.nan_to_num(-1)

        if prior is not None:
            if self.mode == "tr" and self.prior_perturb:
                prior = perturb_prior(prior, **self.prior_perturb)
            img = torch.cat([img, prior[None].to(img.dtype)], dim=0)

        sample = {"image": img, "mask": msk, "id": data_id}
        return sample


class PreparePH2:
    def __init__(self, data_dir, image_size, logger=None, **kwargs):
        self.print = logger.info if logger else print

        self.data_dir = data_dir
        self.image_size = image_size
        # preparing input info.
        self.data_prefix = "IMD"
        self.target_postfix = "_lesion"
        self.target_fex = "bmp"
        self.input_fex = "bmp"
        self.data_dir = self.data_dir
        self.npy_dir = os.path.join(self.data_dir, "np")

    def __get_data_path(self):
        x_path = f"{self.npy_dir}/X_tr_{self.image_size}x{self.image_size}.npy"
        y_path = f"{self.npy_dir}/Y_tr_{self.image_size}x{self.image_size}.npy"
        return {"x": x_path, "y": y_path}

    def __get_img_by_id(self, id):
        img_dir = os.path.join(
            self.imgs_dir, f"{self.data_prefix}{id}.{self.input_fex}"
        )
        img = torch.moveaxis(torch.tensor(np.asarray(Image.open(img_dir))), -1, 0)
        return img

    def __get_msk_by_id(self, id):
        msk_dir = os.path.join(
            self.msks_dir,
            f"{self.data_prefix}{id}{self.target_postfix}.{self.target_fex}",
        )
        msk = torch.tensor(np.asarray(Image.open(msk_dir))).unsqueeze(0).to(torch.uint8)
        return msk

    def __get_transforms(self):
        # transform for image
        img_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=[self.image_size, self.image_size],
                    interpolation=transforms.functional.InterpolationMode.BILINEAR,
                ),
            ]
        )
        # transform for mask
        msk_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=[self.image_size, self.image_size],
                    interpolation=transforms.functional.InterpolationMode.NEAREST,
                ),
            ]
        )
        return {"img": img_transform, "msk": msk_transform}

    def get_ids(self):
        """Image ids in the order of the cached arrays (see datasets/cache_ids.py)."""
        imgs_dir = os.path.join(self.data_dir, "trainx")
        return read_cache_ids(self.npy_dir, self.image_size, imgs_dir, self.input_fex)

    def is_data_existed(self):
        for k, v in self.__get_data_path().items():
            if not os.path.isfile(v):
                return False
        return True

    def prepare_data(self):
        data_path = self.__get_data_path()

        # Parameters
        self.transforms = self.__get_transforms()

        self.imgs_dir = os.path.join(self.data_dir, "trainx")
        self.msks_dir = os.path.join(self.data_dir, "trainy")

        self.img_dirs = sorted(glob.glob(f"{self.imgs_dir}/*.{self.input_fex}"))
        self.data_ids = [
            d.split(self.data_prefix)[1].split(f".{self.input_fex}")[0]
            for d in self.img_dirs
        ]

        # gathering images
        imgs = []
        msks = []
        for data_id in tqdm(self.data_ids):
            img = self.__get_img_by_id(data_id)
            msk = self.__get_msk_by_id(data_id)

            img = self.transforms["img"](img)
            img = (img - img.min()) / (img.max() - img.min())

            msk = self.transforms["msk"](msk)
            msk = (msk - msk.min()) / (msk.max() - msk.min())

            imgs.append(img.numpy())
            msks.append(msk.numpy())

        X = np.array(imgs)
        Y = np.array(msks)

        # check dir
        Path(self.npy_dir).mkdir(exist_ok=True)

        self.print("Saving data...")
        np.save(data_path["x"].split(".npy")[0], X)
        np.save(data_path["y"].split(".npy")[0], Y)
        write_cache_ids(self.npy_dir, self.image_size, [f"{self.data_prefix}{i}" for i in self.data_ids])
        self.print(f"Saved at:\n  X: {data_path['x']}\n  Y: {data_path['y']}")
        return

    def get_data(self):
        data_path = self.__get_data_path()

        self.print("Checking for pre-saved files...")
        if not self.is_data_existed():
            self.print("There are no pre-saved files.")
            self.print("Preparing data...")
            self.prepare_data()
        else:
            self.print(f"Found pre-saved files at {self.npy_dir}")

        self.print("Loading...")
        X = np.load(data_path["x"])
        Y = np.load(data_path["y"])
        self.print("Loaded X and Y npy format")

        return {"x": X, "y": Y}
