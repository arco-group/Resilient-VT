# Resilient VT

[![arXiv](https://img.shields.io/badge/arXiv-<ARXIV_ID>-b31b1b.svg)](https://arxiv.org/abs/<ARXIV_ID>)


This repository contains the PyTorch/Hydra implementation used for the experiments in **"Resilient Vision-Tabular Multimodal Learning under Modality Missingness"**. 
The project studies multilabel chest X-ray classification from paired **MIMIC-CXR images** and **MIMIC-IV structured clinical variables**, with explicit stress tests in which one modality is partially or entirely unavailable during training and/or inference.

Resilient VT extends the NAIM-style masked self-attention mechanism from tabular data to vision-tabular multimodal learning. 
The proposed pipeline uses modality-specific encoders for clinical and imaging data, then fuses their representations with a multimodal transformer that uses modality tokens and masked attention to ignore missing modalities instead of imputing them.

---

## Contents

1. [Installation](#installation)
2. [Repository structure](#repository-structure)
3. [Data expected by the configs](#data-expected-by-the-configs)
4. [Hydra configuration](#hydra-configuration)
5. [Multimodal pipelines](#multimodal-pipelines)
6. [Running experiments](#running-experiments)
7. [Outputs](#outputs)
8. [Contact](#contact)
9. [Citation](#citation)

---

## Installation

The code is configured through [Hydra](https://hydra.cc/) and uses PyTorch models.
A typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then install a PyTorch/torchvision build compatible with your hardware. The SLURM scripts in
[`bash/`](./bash) were used with a CUDA PyTorch bundle, while local runs can set `device` in
[`confs/config.yaml`](./confs/config.yaml) to one of:

```yaml
device: cuda # cuda, cpu, or mps
```

---

## Repository structure

```text
.
├── main.py                         # Hydra entry point
├── model_selection_approach.py     # Post-processing script for model-selection experiments
├── plot_results_REMUS.py           # Plot/table generation utilities
├── confs/
│   ├── config.yaml                 # Root Hydra configuration
│   └── experiment/                 # Experiment, model, dataset, preprocessing configs
├── CMC_utils/
│   ├── datasets/                   # Tabular, imaging and multimodal dataset wrappers
│   ├── models/                     # NAIM, MARIA, ResNet, fusion wrappers, train/test utilities
│   ├── pipelines/                  # Simple, missing-data, joint/early/late multimodal pipelines
│   ├── preprocessing/              # Missingness generation and preprocessing utilities
│   └── metrics/                    # Performance computation
└── bash/                           # SLURM launch scripts used for large experiment grids
```

The executable pipeline is selected by the `pipeline` key in each experiment config. In
[`main.py`](./main.py), the available options are:

| `pipeline` value | Function called | Purpose |
|---|---|---|
| `simple` | `supervised_learning_main` | Standard unimodal training/evaluation. |
| `missing` | `supervised_tabular_missing_main` | Unimodal experiments with generated missingness. |
| `multimodal_early_fusion` | `multimodal_early_fusion_supervised_learning_main` | Concatenates modalities at input/table level. |
| `multimodal_joint_fusion` | `multimodal_joint_fusion_supervised_learning_main` | Trains modality-specific encoders plus a shared fusion network. Used by the proposed approach and most baselines. |
| `multimodal_late_fusion` | `multimodal_late_fusion_supervised_learning_main` | Trains one predictor per modality and fuses predictions. |

---

## Data expected by the configs

The MIMIC configs use two aligned datasets with the same multilabel targets:

| Modality | Config | Expected file |
|---|---|---|
| Clinical/tabular | [`MIMIC_clinical.yaml`](./confs/experiment/databases/MIMIC_clinical.yaml) | `${data_path}/tabular/classification/MIMIC/processed/MIMIC_clinical_data.csv` |
| Chest X-ray/image | [`MIMIC_CXR.yaml`](./confs/experiment/databases/MIMIC_CXR.yaml) | `${data_path}/imaging/classification/MIMIC/MIMIC_img_data.csv` |

By default, local paths are declared in
[`confs/experiment/paths/system/local.yaml`](./confs/experiment/paths/system/local.yaml):

```yaml
data_path: ./datasets
output_path: ./outputs
```

The imaging CSV must contain:

- `ID`: sample identifier;
- `folder`: relative folder containing the image;
- `img`: image filename;
- the 14 multilabel target columns.

The actual image root is configured through `data_folder` in
[`MIMIC_CXR.yaml`](./confs/experiment/databases/MIMIC_CXR.yaml). Update this path before running
experiments on your machine/cluster.

The clinical CSV must contain:

- `ID`: sample identifier;
- clinical variables declared in [`MIMIC_clinical.yaml`](./confs/experiment/databases/MIMIC_clinical.yaml)
  currently 7 continuous features and 374 categorical features;
- the same 14 multilabel target columns.

The 14 targets are:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum,
Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion,
Pleural Other, Pneumonia, Pneumothorax, Support Devices
```

Labels with value `-1` are configured as missing labels via `replace_classes`.

> MIMIC-CXR and MIMIC-IV are controlled-access datasets. This repository contains the code and
> configuration expected by the experiments, but not the data.

---

## Hydra configuration

The root config is [`confs/config.yaml`](./confs/config.yaml):

```yaml
device: cuda
python_version: ${python_version:micro}

defaults:
  - _self_
  - experiment: MIMIC_multimodal_model_selection
  - experiment/paths/system@: local
```

Override the experiment from the command line, for example:

```bash
python3 main.py experiment=MIMIC_multimodal device=cuda
```

The MIMIC multimodal configs share the following defaults:

- datasets: `MIMIC_clinical` and `MIMIC_CXR`;
- test CV: `multilabelstratifiedkfold`, 5 folds by default;
- validation split: `multilabelholdout`, `test_size: 0.2`;
- task: multilabel classification;
- loss: multilabel binary cross-entropy;
- metrics: AUC, accuracy, recall, precision, F1, MCC, and G-mean;
- image preprocessing: channels-first, normalization, resize, gray-to-RGB;
- image augmentation: random-order augmentation with horizontal flip and rotation;
- tabular preprocessing: numerical normalization, categorical encoding, no imputation.

Training missingness is configured through `missing_percentages`, while inference-time stress tests
use `test_missing_percentages`. For multimodal configs these are pairs:

```text
[clinical_missing_fraction, CXR_missing_fraction]
```

For example, `[0.0, 0.75]` means clinical data are kept available and 75% of CXR modalities are
masked. The standard grid is:

```yaml
missing_percentages:
  - [0.0, 0.0]
  - [0.0, 0.25]
  - [0.0, 0.5]
  - [0.0, 0.75]
  - [0.75, 0.0]
  - [0.5, 0.0]
  - [0.25, 0.0]

test_missing_percentages:
  - [0.0, 0.0]
  - [0.0, 0.25]
  - [0.0, 0.5]
  - [0.0, 0.75]
  - [0.0, 1.0]
  - [1.0, 0.0]
  - [0.75, 0.0]
  - [0.5, 0.0]
  - [0.25, 0.0]
```

The generation mode `multimodal_missing_generation: separate_modalities` masks each modality
independently. The `separate_modalities_complete` variant is used by model-selection experiments
where records with any missing modality are handled by a different predictor.

---

## Multimodal pipelines

### Proposed model: 

Config: [`MIMIC_multimodal.yaml`](./confs/experiment/MIMIC_multimodal.yaml)

```yaml
pipeline: multimodal_joint_fusion
model@ms_models.0: naim_pretrained      # clinical encoder
model@ms_models.1: resnet_pretrained    # CXR encoder
model@shared_net: maria                 # multimodal masked-attention fusion transformer
model: multimodal_learner
missing_augmentation_probability: 0.3
```

This is the main pipeline. It builds a [`MultimodalLearner`](./CMC_utils/models/generic/multimodal_model.py)
composed of:

1. a tabular encoder, `NAIM`, for structured clinical variables;
2. an imaging encoder, `ResNet-50`, for chest X-rays;
3. a shared [`MARIA`](./CMC_utils/models/tabular/naim/naim_maria.py) fusion network.

The modality-specific models are used as extractors. Their hidden representations are concatenated
and passed to MARIA, which reshapes them into tokens, applies learnable modality tokens, and uses
masked self-attention so missing modalities/features do not contribute to attention aggregation.
During training, `missing_augmentation_probability: 0.3` enables modality-dropout style
regularization at the multimodal dataset level.

### Frozen-encoder / early-fusion ablation

Config: [`MIMIC_multimodal_early.yaml`](./confs/experiment/MIMIC_multimodal_early.yaml)

This uses the same joint-fusion pipeline, but freezes both pretrained unimodal encoders:

```yaml
overrides@ms_models.0: freeze_unimodal_module
overrides@ms_models.1: freeze_unimodal_module
```

It isolates the effect of fine-tuning the clinical and imaging branches jointly with the multimodal
fusion network.

### Zero-composition baselines

Configs:

- [`MIMIC_multimodal_zeros.yaml`](./confs/experiment/MIMIC_multimodal_zeros.yaml)
- [`MIMIC_multimodal_zeros_maria.yaml`](./confs/experiment/MIMIC_multimodal_zeros_maria.yaml)

These baselines represent missing images/modalities with a constant value:

```yaml
images_missing_value: 0.0
missing_augmentation_probability: 0.0
```

`MIMIC_multimodal_zeros.yaml` uses pretrained modality encoders followed by an MLP fusion head.
`MIMIC_multimodal_zeros_maria.yaml` uses the fusion module with masking disabled through
[`maria_no_mask`](./confs/experiment/overrides/maria_no_mask.yaml).

### Pooling fusion baseline

Config: [`MIMIC_multimodal_pooling.yaml`](./confs/experiment/MIMIC_multimodal_pooling.yaml)

This baseline replaces attention-based fusion with feature pooling. It uses pretrained NAIM and
ResNet encoders, a custom MLP shared head, and applies:

```yaml
overrides@model: max_pooling_fusion
```

The corresponding [`max_pooling_fusion`](./confs/experiment/overrides/max_pooling_fusion.yaml)
override sets the multimodal learner fusion mode to max pooling.

### Late-fusion baseline

Config: [`MIMIC_multimodal_late.yaml`](./confs/experiment/MIMIC_multimodal_late.yaml)

```yaml
pipeline: multimodal_late_fusion
model@model.0: naim
model@model.1: resnet

late_fusion_approaches:
  - max
  - min
  - mean
  - majority_voting
```

This pipeline trains one predictor for each modality and combines predictions only at decision
time. It is useful to compare our intermediate fusion approach against prediction-level fusion.

### Model-selection baseline

Configs/scripts:

- [`MIMIC_multimodal_model_selection.yaml`](./confs/experiment/MIMIC_multimodal_model_selection.yaml)
- [`MIMIC_multimodal_model_selection_maria.yaml`](./confs/experiment/MIMIC_multimodal_model_selection_maria.yaml)
- [`model_selection_approach.py`](./model_selection_approach.py)

These experiments emulate a heuristic model-switching strategy. The configs generate multimodal
predictions under `separate_modalities_complete`, while `model_selection_approach.py` post-processes
predictions by selecting:

- the multimodal predictor when both modalities are available;
- the clinical unimodal predictor when CXR is missing;
- the CXR unimodal predictor when clinical data are missing.

The `*_maria` variant keeps MARIA as the fusion module but disables the missing-value mask, providing
an additional model-selection/no-mask ablation.

### Unimodal references

Configs:

- [`MIMIC_unimodal_tab.yaml`](./confs/experiment/MIMIC_unimodal_tab.yaml): clinical NAIM baseline;
- [`MIMIC_unimodal_img.yaml`](./confs/experiment/MIMIC_unimodal_img.yaml): CXR ResNet baseline;
- [`MIMIC_unimodal_tab_w_tr.yaml`](./confs/experiment/MIMIC_unimodal_tab_w_tr.yaml): clinical baseline trained with modality/sample dropping;
- [`MIMIC_unimodal_img_w_tr.yaml`](./confs/experiment/MIMIC_unimodal_img_w_tr.yaml): imaging baseline trained with modality/sample dropping.

The pretrained multimodal configs expect unimodal checkpoint files to be available under
`./checkpoints/<model-name>/<missing-percentage>/<test-fold>_<val-fold>.pth`, where
`<model-name>` matches the config name such as `naimcat` or `resnet`.

---

## Running experiments

### Single experiment

Run the proposed MARIA-VT pipeline:

```bash
python3 main.py experiment=MIMIC_multimodal
```

Run a baseline:

```bash
python3 main.py experiment=MIMIC_multimodal_zeros
python3 main.py experiment=MIMIC_multimodal_pooling
python3 main.py experiment=MIMIC_multimodal_model_selection
python3 main.py experiment=MIMIC_multimodal_early
python3 main.py experiment=MIMIC_multimodal_late
```

### Run only one training-missingness condition

Hydra overrides can restrict the missingness grid. Quote list values in shells such as zsh:

```bash
python3 main.py experiment=MIMIC_multimodal 'missing_percentages=[[0.0,0.75]]'
```

### Run only one outer fold

The multimodal pipelines support the optional `fold_to_do` key:

```bash
python3 main.py experiment=MIMIC_multimodal 'missing_percentages=[[0.0,0.75]]' +fold_to_do=2
```

### Model-selection post-processing

After the required unimodal and multimodal prediction folders exist, run:

```bash
python3 model_selection_approach.py experiment=MIMIC_multimodal_model_selection
```

### SLURM launch scripts

The [`bash/`](./bash) directory contains cluster-oriented launch scripts, including:

- [`launch_experiment.bash`](./bash/launch_experiment.bash): one experiment;
- [`launch_experiment_with_separate_folds.bash`](./bash/launch_experiment_with_separate_folds.bash): split folds across array jobs;
- [`launch_experiments_with_separate_parameters.bash`](./bash/launch_experiments_with_separate_parameters.bash): split missingness settings across array jobs;
- [`launch_experiments_with_separate_parameters_and_folds.bash`](./bash/launch_experiments_with_separate_parameters_and_folds.bash): split both folds and missingness settings.

Before using them, update the account, partition, virtual environment, project path, and selected
experiment grid for your cluster.

---

## Outputs

Hydra writes outputs under `${output_path}`; by default this is `./outputs`.
For missing-modality multimodal experiments, the path structure is organized by:

```text
outputs/
└── <experiment_name>/
    └── <test_cv>_<val_cv>/
        ├── cross_validation/
        ├── missing_masks/
        ├── preprocessing/
        ├── saved_models/
        ├── predictions/
        │   └── <model_and_preprocessing>/<train_missing_percentage>/<test_missing_percentage>/
        ├── results/
        │   └── <model_and_preprocessing>/<train_missing_percentage>/<test_missing_percentage>/
        ├── logs/
        └── configs/
```

For paired multimodal missingness, folder names such as `0-75` mean:

```text
0% clinical missing, 75% CXR missing
```

Predictions are saved as CSV files per fold and split, for example:

```text
<test_fold>_<val_fold>_train.csv
<test_fold>_<val_fold>_val.csv
<test_fold>_<val_fold>_test.csv
```

Results are computed for both unbalanced and class-balanced variants where applicable, and include
per-class, micro-average, macro-average, and weighted-average summaries.

---

## Contact 

For any questions, please contact [camillomaria.caruso@unicampus.it](mailto:camillomaria.caruso@unicampus.it) and [valerio.guarrasi@unicampus.it](mailto:valerio.guarrasi@unicampus.it).

---

## Citation

If you use this code, please cite the associated paper once available:


```bibtex
@article{caruso2026resilient,
  title={Resilient Vision-Tabular Multimodal Learning under Modality Missingness},
  author={Caruso, Camillo Maria and Soda, Paolo and Guarrasi, Valerio},
  journal={arXiv preprint arXiv:},
  year={2026}
} 

```