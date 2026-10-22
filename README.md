# MLFFM-SegDiff

Code and results for the paper **"MLFFM-SegDiff: A multi-level feature fusion diffusion model for skin lesion
segmentation"** (J. Gu, C. Shen, Y. Cao, W. Zhang, Y. Li, A. Fan; PLOS ONE).

MLFFM-SegDiff (`src/models/mlffm_segdiff.py`) is a conditional diffusion model for skin lesion segmentation built
on the official [DermoSegDiff](https://github.com/mindflow-institue/DermoSegDiff) code. It adds a multi-level
feature fusion module (MLFFM) to the skip connections of the DermoSegDiff denoiser and batch normalization after its
linear attention layers. The repository also contains the comparison methods of the paper (DermoSegDiff, U-Net,
SwinUNETR and Generic U-Net), the scripts for the nnU-Net baseline, the per-image evaluation with boundary metrics
(HD95, ASSD), the paired statistical tests and the efficiency measurements, and, in [`results/`](results/), the
per-image and per-fold results behind Tables 4–9 of the paper.

## Repository structure

```text
configs/paper/<dataset>/<model>.yaml   settings of every model in the paper (ISIC2018, PH2, HAM10000)
configs/paper/ph2/diffusion_vs_nondiffusion/   PH2 comparison with the non-diffusion counterpart
splits/<dataset>_5fold.csv             the five folds of the paper, in the order of the data cache
src/                                   models, data loading, training and testing
  models/          MLFFMSegDiff, MLFFMSeg (non-diffusion counterpart), DermoSegDiff
  seg_models/      U-Net, SwinUNETR, Generic U-Net
  run_kfold.py     five-fold training and testing
  per_case_metrics.py   per-image metrics, including HD95 and ASSD
scripts/
  order_cache.py           puts the data cache in the image order of splits/ (run once per dataset)
  evaluate_per_case.py     per-image evaluation of the fold checkpoints
  compare_per_case.py      paired image-level comparison (Wilcoxon, Holm, bootstrap CIs)
  measure_efficiency.py    parameters, FLOPs, training and inference time
  nnunet/                  export to nnU-Net v2 with the paper's folds, evaluation, efficiency
results/                               results of the paper and reproduce_tables.py (see results/README.md)
tests/                                 tests on small synthetic data
```

## Environment

Python 3.8 or newer and PyTorch with CUDA for training. `requirements.txt` pins the package versions
(PyTorch 2.0.1); the tests also pass with Python 3.11 and PyTorch 2.2.

```bash
pip install -r requirements.txt
```

Install the `torch`/`torchvision` build that matches your CUDA version first if needed. FLOP counting in
`scripts/measure_efficiency.py` needs PyTorch 2.1 or newer. nnU-Net v2 (2.8.1 in the paper) is installed in a
separate environment (`pip install nnunetv2`); its scripts are described below. The tests need `pytest`:

```bash
python -m pytest tests -q
```

## Data

The datasets are not included. Download them from their original sources and arrange them as follows (the
`dataset.data_dir` of the configs):

```text
data/
  ISIC2018/
    ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg ...             2,594 images
    ISIC2018_Task1_Training_GroundTruth/ISIC_0000000_segmentation.png ...
  ph2/
    trainx/IMD002.bmp ...                                            200 images
    trainy/IMD002_lesion.bmp ...
  ham10000/
    images/ISIC_0024306.jpg ...                                      10,015 images
    masks/ISIC_0024306_segmentation.png ...
    HAM10000_metadata                                                CSV with image_id and lesion_id
```

- **ISIC2018**: the training data of Task 1 of the ISIC 2018 challenge,
  `ISIC2018_Task1-2_Training_Input.zip` and `ISIC2018_Task1_Training_GroundTruth.zip` from
  <https://challenge.isic-archive.com/data/#2018>. The validation and test sets of the challenge are not used.
- **PH2**: <https://www.fc.up.pt/addi/ph2%20database.html> (Mendonça et al., EMBC 2013). The archive has one
  folder per image; copy the dermoscopic images and lesion masks into `trainx/` and `trainy/`:

  ```bash
  mkdir -p data/ph2/trainx data/ph2/trainy
  for d in "PH2Dataset/PH2 Dataset images"/IMD*; do
    i=$(basename "$d")
    cp "$d/${i}_Dermoscopic_Image/$i.bmp" data/ph2/trainx/
    cp "$d/${i}_lesion/${i}_lesion.bmp" data/ph2/trainy/
  done
  ```

- **HAM10000**: Harvard Dataverse, [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
  (CC BY-NC 4.0): the images of `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip` go to `images/`,
  the lesion masks of `HAM10000_segmentations_lesion_tschandl.zip` to `masks/`. The metadata
  (`HAM10000_metadata.tab`; download it as comma-separated values, the original file format) is used only for the
  lesion-clustered bootstrap intervals and is saved as `data/ham10000/HAM10000_metadata`.

### Data cache and folds

On first use, the images are resized to 128 × 128 and cached in `<data_dir>/np/` (`X_tr_128x128.npy`,
`Y_tr_128x128.npy`, and `ids_128x128.txt` with the image id of every cached sample). The five folds are positions
in this cache: the samples are permuted with NumPy seed 0 and split into five folds of nearly equal size
(`src/utils/kfold.py`), the same for all models. A new cache is built in sorted file-name order, but the ISIC2018
and PH2 caches of the paper followed an unsorted file listing. `splits/<dataset>_5fold.csv` lists the image of
every cache position in the paper's order together with its fold, and `scripts/order_cache.py` builds the cache
(if needed) and puts it in that order. Run it once per dataset before training or evaluation:

```bash
python scripts/order_cache.py --dataset isic2018 --data_dir data/ISIC2018
python scripts/order_cache.py --dataset ph2 --data_dir data/ph2
python scripts/order_cache.py --dataset ham10000 --data_dir data/ham10000   # already in order
```

The fold of every image is also given in `results/folds/<dataset>_folds.csv`.

## Reproducing the experiments

Run all commands from the repository root. Outputs (checkpoints, logs, metrics) are written to
`outputs/paper/<dataset>/<model>/`, as set in the configs.

### Main comparison (Table 4)

Each config in `configs/paper/<dataset>/` holds the settings of one model in Table 4 (`mlffm`, `dermosegdiff`,
`unet`, `swinunetr`, `generic_unet`), taken from the configuration printed in the test log of the original
experiment. `src/run_kfold.py` trains and tests the five folds:

```bash
python src/run_kfold.py -c configs/paper/isic2018/mlffm.yaml                 # all five folds
python src/run_kfold.py -c configs/paper/isic2018/mlffm.yaml --folds 0,1     # some folds, e.g. one GPU each
python src/run_kfold.py -c configs/paper/isic2018/mlffm.yaml --aggregate_only   # summary of all folds
```

Per-fold metrics (accumulated over all pixels of the test fold) and their summary are written to
`outputs/paper/<dataset>/<model>/metrics/`. Notes:

- **Checkpoint selection.** As stated in the paper, Table 4 used, for every fold, the checkpoint with the lowest
  loss on the held-out fold (`*_best.pth`), which therefore served for both checkpoint selection and testing. The
  training code in this repository was restructured after the original experiments and saves the checkpoint of
  the final epoch (`*_last.pth`); it does not evaluate the held-out fold during training. Retraining with these
  configs therefore follows the original settings except for the checkpoint selection, and the results will also
  differ because of random initialization and diffusion sampling.
- Generic U-Net is the `Generic_UNet` class of nnU-Net v1 with its constructor defaults, as in the original
  experiments. (`GenericUNet` in `src/seg_models/generic_unet.py`, with nnU-Net's training settings, was not used
  in the paper.)
- `ham10000/dermosegdiff.yaml` tests with batches of 128 images (about 35 GB of GPU memory); use
  `ham10000/dermosegdiff_bs64.yaml` on smaller GPUs. It gives the same predictions up to the order of the sampling
  noise.
- Importing `src/models/mlffm_segdiff.py` disables cuDNN, as in the original code, so the diffusion models run
  without cuDNN; the feed-forward models also do unless `run.cudnn: true` is set.

### Per-image evaluation with HD95 and ASSD (Tables 6 and 7)

`scripts/evaluate_per_case.py` predicts the held-out images of every fold with the fold checkpoints and writes one
row per image (Dice, Jaccard, precision, recall, specificity, accuracy, HD95, ASSD, and pixel counts; conventions
in `src/per_case_metrics.py`) to `outputs/paper/<dataset>/<model>/per_case/` (`oof_cases.csv`, `summary.json`).
Inference is the same as in `src/testing.py` / `src/seg_testing.py`. By default it loads the checkpoints written by
`run_kfold.py`; `--checkpoint` selects others, with `{fold}` for the fold index (for the paper, the original
`*_best.pth` checkpoints, e.g. `n-mlffm_i01_f{fold}_s-128_b-32_t-250_sc-linear_best.pth`, named in the header of
each config):

```bash
python scripts/evaluate_per_case.py -c configs/paper/isic2018/mlffm.yaml
python scripts/evaluate_per_case.py -c configs/paper/isic2018/mlffm.yaml --checkpoint "path/to/model_f{fold}_best.pth"
python scripts/evaluate_per_case.py -c configs/paper/isic2018/mlffm.yaml --folds 0,1     # some folds
python scripts/evaluate_per_case.py -c configs/paper/isic2018/mlffm.yaml --summary_only  # combine fold files
```

### nnU-Net baseline

nnU-Net v2 (2.8.1) with the 2D configuration, trained from scratch for 100 epochs (`nnUNetTrainer_100epochs`, the
trainer distributed with nnU-Net; otherwise default settings) on the same five folds. Its final-epoch checkpoint
predicts the held-out images of each fold. With `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` set:

```bash
# export (repository environment), plan and preprocess (nnU-Net environment), copy the paper's folds
bash scripts/nnunet/prepare_paper_kfold.sh         # datasets 521 (ISIC2018), 522 (PH2), 523 (HAM10000)

# train every fold k = 0..4 (nnU-Net environment)
nnUNetv2_train 521 2d 0 -tr nnUNetTrainer_100epochs

# per-image evaluation of the held-out predictions (repository environment)
python scripts/nnunet/evaluate_nnunet.py --dataset isic2018 --paper_kfold --trainer nnUNetTrainer_100epochs \
  --results_dir $nnUNet_results/Dataset521_ISIC2018KF/nnUNetTrainer_100epochs__nnUNetPlans__2d \
  --out_dir outputs/paper/isic2018/nnunet/per_case --nnunet_version 2.8.1 \
  --plans_file $nnUNet_preprocessed/Dataset521_ISIC2018KF/nnUNetPlans.json
```

`prepare_paper_kfold.sh` uses `python` and the `nnUNetv2_*` commands on the `PATH`; set `PYTHON` and `NNBIN` to use
other environments. The export images are the cached 128 × 128 arrays that all other models use.

### Paired comparisons (Table 7)

```bash
O=outputs/paper/isic2018
python scripts/compare_per_case.py --ref MLFFM=$O/mlffm/per_case \
  --compare DermoSegDiff=$O/dermosegdiff/per_case UNet=$O/unet/per_case SwinUNETR=$O/swinunetr/per_case \
            GenericUNet=$O/generic_unet/per_case nnUNet=$O/nnunet/per_case \
  --metrics dice jaccard hd95 assd precision recall --out_dir $O/comparison
```

For HAM10000, add `--cluster_file data/ham10000/HAM10000_metadata` to resample whole lesions in the bootstrap. The
`Favours` column of the output follows the sign of the mean difference; when the Wilcoxon test and the mean point
in different directions, read both.

### Computational efficiency (Table 9)

```bash
python scripts/measure_efficiency.py -c configs/paper/isic2018/*.yaml --out_dir outputs/paper/efficiency/isic2018
python scripts/nnunet/measure_nnunet.py --fold 0 --batch_size 32 \
  --model_folder $nnUNet_results/Dataset521_ISIC2018KF/nnUNetTrainer_100epochs__nnUNetPlans__2d \
  --out_file outputs/paper/efficiency/nnunet_isic2018.json      # in the nnU-Net environment
```

The paper reports measurements on one NVIDIA RTX A6000.

### Diffusion model vs. non-diffusion counterpart on PH2 (Ablation study)

The non-diffusion counterpart (`MLFFMSeg`) uses the same network, receives an all-zero mask input and predicts the
mask directly. Both models are trained for 1,000 epochs with batch size 12, a constant learning rate and no EMA,
and the final-epoch checkpoint is tested:

```bash
python src/run_kfold.py -c configs/paper/ph2/diffusion_vs_nondiffusion/diffusion.yaml
python src/run_kfold.py -c configs/paper/ph2/diffusion_vs_nondiffusion/nondiffusion.yaml
```

### Tables from the published results

`results/` contains the per-image and per-fold results of the paper; this recomputes Tables 4–8 and the PH2
comparison from them (no data or GPU needed):

```bash
python results/reproduce_tables.py --ham_metadata data/ham10000/HAM10000_metadata
```

See [`results/README.md`](results/README.md) for the files and notes on the results.

## Notes

- The folds are formed at the image level. In HAM10000, images of the same lesion can therefore be in different
  folds; the paper reports a sensitivity analysis.
- Some options of the code were not used in the paper and are off in all configs, for example `run.kfold.group_by`
  (lesion-grouped folds), `run.kfold.dev_fraction`, `dataset.prior`, `testing.clip_x0`, the ablation switches of
  `MLFFMSegDiff` and the `GenericUNet` class.
- Model weights are not included.

## License and acknowledgements

The code is released under the MIT License (see `LICENSE`), as the DermoSegDiff code it builds on. The Generic
U-Net (`src/seg_models/generic_unet.py`, `src/seg_models/gunterpp/`) is taken from nnU-Net and the SwinUNETR
implementation (`src/seg_models/swin_unetr_resswin_skipadd.py`) from MONAI; both are under the Apache License 2.0,
as stated in their file headers. The diffusion utilities follow
[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

## Citation

If you use this code, please cite the paper (citation to be completed on publication) and DermoSegDiff:

```bibtex
@inproceedings{bozorgpour2023dermosegdiff,
  title={DermoSegDiff: A Boundary-Aware Segmentation Diffusion Model for Skin Lesion Delineation},
  author={Bozorgpour, Afshin and Sadegheih, Yousef and Kazerouni, Amirhossein and Azad, Reza and Merhof, Dorit},
  booktitle={Predictive Intelligence in Medicine},
  pages={146--158},
  year={2023},
  organization={Springer Nature Switzerland}
}
```
