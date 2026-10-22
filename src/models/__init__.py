from .segdiffs import DermoSegDiff
from .mlffm_segdiff import MLFFMSegDiff, MLFFMSeg

MODELS = {"DermoSegDiff": DermoSegDiff, "MLFFMSegDiff": MLFFMSegDiff,
          "MLFFMSeg": MLFFMSeg}
__all__ = ["DermoSegDiff", "MLFFMSegDiff", "MLFFMSeg", "MODELS"]
