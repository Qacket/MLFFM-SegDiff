#!/bin/bash
# Export ISIC2018 / PH2 / HAM10000 with the manuscript's 5 folds (export_nnunet.py --paper_kfold),
# plan and preprocess them for nnU-Net, and put the folds where nnUNetv2_train reads them.
# Run once from the repository root (see README.md). PYTHON is the environment of this repository,
# NNBIN the bin directory of the nnU-Net environment; nnUNet_raw, nnUNet_preprocessed and
# nnUNet_results must be set as nnU-Net requires.
set -euo pipefail
PYTHON=${PYTHON:-python}
NNBIN=${NNBIN:-$(dirname "$(command -v nnUNetv2_plan_and_preprocess)")}
: "${nnUNet_raw:?set nnUNet_raw}" "${nnUNet_preprocessed:?set nnUNet_preprocessed}" "${nnUNet_results:?set nnUNet_results}"
export nnUNet_raw nnUNet_preprocessed nnUNet_results
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

for spec in isic2018:521:ISIC2018KF ph2:522:PH2KF ham10000:523:HAM10000KF; do
  IFS=: read -r dataset id name <<< "$spec"
  "$PYTHON" scripts/nnunet/export_nnunet.py --dataset "$dataset" --dataset_id "$id" --raw_dir "$nnUNet_raw" --paper_kfold
  "$NNBIN/nnUNetv2_plan_and_preprocess" -d "$id" -c 2d -np 4 --verify_dataset_integrity
  cp "$nnUNet_raw/Dataset${id}_${name}/splits_final.json" "$nnUNet_preprocessed/Dataset${id}_${name}/"
  echo "Dataset${id}_${name}: exported, preprocessed, splits_final.json in place"
done
