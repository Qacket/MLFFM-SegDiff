from .unet import UNet
from .generic_unet import GenericUNet, Generic_UNet
from .swin_unetr_resswin_skipadd import SwinUNETR
from models.mlffm_segdiff import MLFFMSeg

MODELS = {"UNet": UNet, "GenericUNet": GenericUNet,
          "Generic_UNet": Generic_UNet, "SwinUNETR": SwinUNETR,
          "MLFFMSeg": MLFFMSeg}
__all__ = ["UNet", "GenericUNet", "Generic_UNet", "SwinUNETR", "MLFFMSeg", "MODELS"]
