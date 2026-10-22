# Results underlying the paper

This folder contains the per-image and per-fold results behind Tables 4–9 of the paper and the PH2 comparison between the diffusion model and its non-diffusion counterpart. It does not contain dataset images or masks, which are available from their original sources (ISIC 2018 Challenge Task 1, PH2, and HAM10000 on Harvard Dataverse, doi:10.7910/DVN/DBW86T), or model weights.

To recompute the table values:

```bash
python results/reproduce_tables.py --ham_metadata path/to/HAM10000_metadata.csv
```

The metadata file (columns `image_id` and `lesion_id`) is distributed with HAM10000 and is needed only for the lesion-clustered bootstrap intervals of HAM10000 in Table 7.

## Files

| Path | Content | Used for |
|---|---|---|
| `folds/<dataset>_folds.csv` | Fold (0–4) of every image. The folds were formed at the image level (random permutation, NumPy seed 0) and are the same for all models. | All tables |
| `per_image/<dataset>/<model>.csv` | Per-image results of the re-evaluation: Dice, Jaccard, precision, recall, specificity, accuracy, HD95, and ASSD (pixels at 128 × 128), pixel counts (`tp`, `fp`, `fn`, `gt_area`, `pred_area`), and `empty_pred` | Tables 6, 7, and 8B |
| `fold_level/main_comparison.csv` | Per-fold metrics of the original experiments, accumulated over all pixels of each test fold | Tables 4, 5, and 8A |
| `fold_level/diffusion_vs_nondiffusion_ph2.csv` | Per-fold metrics of the matched comparison on PH2 (1,000 epochs, batch size 12, no EMA, final epoch) | Ablation study, non-diffusion comparison |
| `efficiency/` | Parameters and FLOPs (`flops.json`), and measured latency, memory, and training time per iteration on one NVIDIA RTX A6000 (`efficiency_<dataset>.json`, `nnunet_<dataset>.json`) | Table 9 |
| `reproduce_tables.py` | Recomputes Tables 4–8 and the non-diffusion comparison from these files | |

Model names: `mlffm_segdiff`, `dermosegdiff`, `unet`, `swinunetr`, `generic_unet`, `nnunet`. Rows of the per-image files are paired across models by `fold` and `index`; `scripts/compare_per_case.py` reads them directly.

## Notes

- **Diffusion models.** Sampling is stochastic, so the per-image re-evaluation of MLFFM-SegDiff and DermoSegDiff differs slightly from the original experiments (pooled Dice by at most 0.0076), as stated in the caption of Table 6.
- **HAM10000, fold 0 of the feed-forward baselines.** The fold-0 checkpoints of U-Net, SwinUNETR, and Generic U-Net used for Table 4 were overwritten by a retraining with the same configuration. `fold_level/main_comparison.csv` contains the original fold-0 results (taken from the test log of the original run); the per-image files contain the retrained models.
- **HAM10000, two full-image lesions.** In ISIC_0029819 and ISIC_0026042, the lesion covers the whole image. The data loader inherited from DermoSegDiff normalizes masks by min–max scaling, which turned these masks into empty masks. In the per-image files, these two images were re-scored against full masks for all models. In the original experiments behind Table 4 they were scored as empty masks, which changes pooled metrics by about 10⁻⁴.
- **Rounding in Table 4.** The published values of Table 4 differ from the values recomputed from `fold_level/main_comparison.csv` by at most 0.0002, because the original table was prepared with inconsistent rounding of the fourth decimal place. Table 5 is the mean of the published Table 4 values.
- **Ablation variants.** Per-fold records are not provided for the ablation variants in Table 10, except for the last row of each dataset, which is the MLFFM-SegDiff configuration of Table 4.
