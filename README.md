# How Many Dimensions Are Enough? Investigating Bottleneck Sufficiency in Vision-Language Knowledge Distillation

<!-- **Experiment runner and reproducibility guide** -->

This repository supports experiments examining how bottleneck width affects classification accuracy when transferring representations from a frozen vision-language teacher to CNN students. It evaluates multiple width candidates across datasets, student architectures, and training objectives, including labels-only learning and knowledge distillation (KD). The experiment runner launches each selected teacher × dataset × student × ablation combination sequentially, records logs, and can resume incomplete experiments.


## 1. Repository layout

Place `run_experiments.py` and the enabled dataset training scripts in the repository root:

```text
CVPR2027/
├── README.md
├── run_experiments.py
├── test_performance.py             # Evaluate trained models
├── fmnist.py
├── svhn.py
├── cifar10.py
├── cifar100.py
├── data/                           # Dataset files
├── results/                        # Experiment results
└── saved_models/                   # Saved model checkpoints

```

Optional scripts named in the runner, currently **disabled**, are `kmnist.py` and `tinyimagenet200.py`. Include them if you enable those datasets.

## 2. Environment and prerequisites

- Python 3 with packages required by the dataset-specific training scripts.
- A working machine-learning environment with the appropriate GPU framework and drivers if GPU training is enabled.
- Training scripts that accept the flags passed by the runner (`--teacher`, `--pretrained`, `--student`, `--epochs`, `--seeds`, `--modes`, `--dims`, `--gpus`, `--num-workers`, `--data-root`, `--out`, and optionally `--resume`, `--max-train`, and ablation flags).
- Dataset files accessible under `DATA_ROOT` (default `./data`) in the format expected by each training script.
- Access to the specified pretrained teacher checkpoints where required.

The runner itself uses Python's standard library; **the exact package versions for training are not available from this file**. Add a verified `requirements.txt` or environment file based on the training scripts, rather than assuming unverified versions.

## 3. Quick start

From the repository root:

```bash
# Inspect the full experiment matrix without training
python run_experiments.py --dry-run

# Run all experiments enabled in SETTINGS
python run_experiments.py
```

The runner has no command-line switches for changing the model matrix; edit the `SETTINGS` section in `run_experiments.py` before execution. A dry run still creates the results root and writes a run summary, but does not launch the training scripts.

## 4. Current settings in the supplied runner

| Item | Active setting |
| --- | --- |
| Teacher | `vitl14` → OpenCLIP `ViT-L-14`, pretrained `laion2b_s32b_b82k` (768-dimensional embedding) |
| Datasets | Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100 |
| Students | EfficientNet-B0, MobileNetV2, ResNet-18 |
| Objectives | `labels`, `feat`, `rkd`, `crd`, `dkd`, `vl2lite`, `sskd`, `clipkd` |
| Seeds | `42` |
| Epochs | `5` |
| Ablation | `main` |
| Training data limit | None (full training set as handled by each training script) |
| GPU selection | `0` |
| Data-loader workers | `4` |
| Dataset root | `./data` |
| Results root | `results` |

**With these active settings, the runner schedules 1 teacher × 4 datasets × 3 students × 1 ablation = 12 sequential training-script invocations.** Each invocation handles its enabled modes and width grid internally. This is not a count of individual trained models.

**Important:** `EPOCHS_DEFAULT = 5` and `SEEDS = '42'` are quick-test settings, **not** a multi-seed, full-length paper reproduction. The runner itself warns when epochs are below 20 or there is only one seed; standard deviations across independent seeds are unavailable with the current configuration.

### Bottleneck-width candidates

The default `DIMS` string is:

```text
2,4,8,12,16,24,32,48,64,96,128,
mle*0.5,mle*0.75,mle,pca,erank,cap,mle*1.25,mle*1.5,full
```

The CIFAR-100 entry overrides the general grid with additional explicit widths `192` and `256`. The tokens `mle`, `pca`, `erank`, `cap`, their MLE-scaled variants, and `full` are interpreted by the dataset-specific training scripts; the runner only forwards them.

### Alternative teacher configurations

The supplied runner includes the following teacher definitions; only ViT-L/14 is enabled:

| Tag | Teacher | Pretrained identifier | Embedding dim. | Default status |
| --- | --- | --- | ---: | --- |
| `vitb32` | `ViT-B-32` | `laion2b_s34b_b79k` | 512 | Enabled |
| `vitl14` | `ViT-L-14` | `laion2b_s32b_b82k` | 768 |  Enabled |
| `vith14` | `ViT-H-14` | `laion2b_s32b_b79k` | 1024 | Disabled |

To enable a teacher, uncomment its entry in `TEACHERS`. **Verify that your installed OpenCLIP version offers the exact model/pretrained pairing and that each training script uses the corresponding teacher embedding dimension.** The runner passes the model and pretrained identifier, but does not pass the `dim` field as a separate argument.

To add Tiny ImageNet or KMNIST, uncomment their entries in `DATASETS` and include the corresponding training scripts. To run a matched-view ablation, uncomment `matched_views` in `ABLATIONS`; it passes `--no-augment` to the training scripts.

## 5. Running and resuming experiments

The runner executes one script at a time using the same Python interpreter used for the runner. Before each invocation, it prints the dataset, teacher, student, epochs, destination folder, detected state, and full command.

The resulting path follows:

```text
results/<teacher>/<dataset>_<student>_e<epochs>/
```

Example under the active configuration:

```text
results/vitl14/cifar100_resnet18_e5/
```

Non-default ablations append their name to the folder; setting `MAX_TRAIN` adds an `_n<number>` suffix.

The runner checks `selection.json`, `missing_runs.json`, `pareto_summary.csv`, and `results_per_seed.csv` to decide whether a run is new, resumable, or complete. When it detects a resumable run, it passes `--resume`. If a run appears finished and `SKIP_FINISHED=True`, it skips the run.

> **Caution:** Completion detection compares reported seed count and modes, but does not exhaustively verify every requested width, seed identity, checkpoint, or experimental artifact. Recheck completion before publishing results.

If an individual run fails, the runner continues to the next one. At the end, it reports all statuses and returns a nonzero exit code if a subprocess failed. If interrupted, rerunning can continue experiments for which the training scripts properly implement `--resume`.

## 6. Output files

The runner directly creates or maintains:

```text
results/
├── runs_summary.csv
└── vitl14/
    ├── fmnist_efficientnet_b0_e5/
    │   └── run_log.txt
    ├── ...
    └── cifar100_resnet18_e5/
        └── run_log.txt
```

It also expects the dataset-specific scripts to produce some of the following files for its completion/resumption logic:

| File | Role |
| --- | --- |
| `run_log.txt` | Stdout/stderr captured and appended by the runner for each invocation |
| `runs_summary.csv` | Runner-generated job metadata, status, and elapsed minutes |
| `pareto_summary.csv` | Expected training-script summary used for completion checks |
| `results_per_seed.csv` | Expected per-seed training results used to detect resumable work |
| `selection.json` | Optional training-script completion metadata (`complete`) |
| `missing_runs.json` | Optional training-script report of incomplete work |

Detailed metrics, checkpoint filenames, plots, and schema depend on the training scripts; they cannot be inferred conclusively from this runner alone.

## 7. Experimental methodology and reporting

The experiment matrix supports comparison of student bottleneck widths derived from fixed candidates, MLE intrinsic-dimensionality estimates, PCA, effective rank, class-aware capacity, and full-width representations. Eight training objectives are available, including a labels-only control and seven KD modes.

For publication, distinguish **teacher intrinsic dimension** from **empirically sufficient student bottleneck width**. An empirically sufficient width depends on the measured accuracy, the candidate grid, the specified tolerance, and the training protocol. It is not a guaranteed globally minimal width.

The supplied runner does **not** implement the Random Forest/XGBoost strategy-feasibility predictor or generate the paper's predictive generalization tables by itself. Those analyses require separate scripts and outputs.

To reproduce the numerical results in the paper, ensure the configuration matches the reported teacher, dataset, seed list, training epochs, preprocessing, candidate width grid, and evaluation protocol. Do not label results from the current five-epoch, single-seed settings as a full multi-seed reproduction.

## 8. Troubleshooting

| Symptom | Check |
| --- | --- |
| `Training script(s) not found` | Place every enabled dataset script in the current working directory, or update the configured paths. |
| Unrecognized teacher flags | Verify each training script accepts `--teacher` and `--pretrained` and constructs matching-size projection layers. |
| GPU out of memory | Review teacher size, batch size in training scripts, and available GPU memory. The runner does not set batch size. |
| `--resume` error | Check that the dataset training script supports the flag and its saved state is valid. |
| Run skipped incorrectly | Inspect the generated summary and completion metadata; set `SKIP_FINISHED=False` only when rerunning is intentional. |
| Different numerical results | Confirm epochs, seed list, checkpoints, data processing, and exact pretrained weights. |


## 9. CIFAR-100 held-out feasibility prediction (RQ2)

`test_performance.py` is a separate **meta-classifier experiment**. It does not retrain the CLIP teachers or CNN students. Instead, it learns to classify whether each candidate student bottleneck width satisfies a specified accuracy-regret tolerance, using results already recorded in Pareto summary CSV files.

### Train/test split

| Split | Dataset | Role |
| --- | --- | --- |
| Train | Fashion-MNIST (`fmnist`) | Feasibility-classifier training |
| Train | SVHN (`svhn`) | Feasibility-classifier training |
| Train | CIFAR-10 (`cifar10`) | Feasibility-classifier training |
| Train | Tiny ImageNet-200 (`tinyimagenet`) | Feasibility-classifier training, **only if corresponding CSV files are available** |
| Test only | CIFAR-100 (`cifar100`) | Held-out evaluation; excluded from classifier training |

This is a **single CIFAR-100 held-out split**, not a complete five-fold leave-one-dataset-out (LODO) experiment. Missing datasets are not silently generated; at least one non-CIFAR-100 training dataset and CIFAR-100 test data must be available. Both target classes must occur in the training records.

### Requirements

In addition to the dataset-training dependencies, the feasibility script requires `numpy`, `pandas`, `scikit-learn`, `joblib`, and `xgboost`.
```bash
pip install numpy pandas scikit-learn joblib xgboost
```

### Input files

Place `test_performance.py` and the flattened Pareto summary CSV files in the **same working directory**. Files must follow the naming pattern:
```text
<teacher>_<dataset>_<student>_e<epochs>_pareto_summary.csv
```

For example:
```text
vitb32_fmnist_resnet18_e5_pareto_summary.csv
vitb32_svhn_mobilenetv2_e5_pareto_summary.csv
vitl14_cifar10_efficientnet_b0_e5_pareto_summary.csv
vitb32_cifar100_resnet18_e5_pareto_summary.csv
vitl14_tinyimagenet_mobilenetv2_e5_pareto_summary.csv
```

A suffix such as `(1)` before `.csv` is also accepted. Valid teacher and student identifiers are parsed from filenames. The script reads CSV files from `Path.cwd()` **without recursively searching the `results/` directory**. Therefore, summary files stored inside nested `results/` directories must first be collected in the working directory or the script must be adapted.

Each CSV requires the columns `mode`, `bottleneck_kind`, `bottleneck_d`, and `test_accuracy_pct`. Teacher geometry is derived from optional `mle_selected`, `pca_selected`, and `erank_selected` flags. Rows are restricted to **linear bottlenecks** and repeated rows with the same objective and width are averaged.

### Run

From the directory containing the script and CSV files:
```bash
python test_performance.py
```

In a Jupyter notebook:
```python
%run test_performance.py
```

Default configuration in the script:
```python
DATA_DIR = Path.cwd()
OUT_DIR = DATA_DIR / "saved_models" 
DELTA = 1.0
SEED = 42
```

The value `DELTA = 1.0` means a tolerance of **one percentage point**, not a relative 1% reduction in accuracy.

### Learning target and models

For each dataset, teacher, student, and distillation objective, the script sets the best observed accuracy across evaluated linear widths to `A_max`. For candidate width `d`, it calculates:
```text
regret_pp = A_max - accuracy_at_d
feasible = 1 if regret_pp <= DELTA else 0
```

The model inputs include log-scaled candidate width, class count, teacher geometric dimension estimates and width ratios, teacher identity, student identity, and objective (`mode`). The script trains five classifiers with the hyperparameters defined in the Python source: Gradient Boosting, AdaBoost, Random Forest, RBF-SVM, and XGBoost. Each pipeline includes its own fitted preprocessing steps.

Each classifier is **trained on the four non-CIFAR-100 datasets that are actually present**, saved with `joblib.dump()`, **reloaded with `joblib.load()`**, and then evaluated on the CIFAR-100 records. The script does **not** reuse previously trained `saved_models/*.joblib` models; doing so could invalidate the held-out split if those models were trained using CIFAR-100.

### Saved outputs
```text
saved_models/
    ├── gradient_boosting.joblib
    ├── adaboost.joblib
    ├── random_forest.joblib
    ├── rbf_svm.joblib
    ├── xgboost.joblib
```

### Interpretation and limitations

**Important:** The current code computes training and held-out feasibility labels from the CSV field `test_accuracy_pct`. Thus, the split prevents CIFAR-100 *rows* from being used to fit the classifiers, but the target definition is **retrospective and test-accuracy-derived**. It is not a leakage-free prospective bottleneck-selection evaluation. The reported scores measure prediction of retrospective feasibility labels, not demonstrated deployment-time width selection or reduced training-search cost. All measurements are conditional on the available width grid, accuracy measurements, and training settings used to create each source CSV. The script does not perform multi-seed uncertainty analysis, feature ablation, or a full LODO evaluation by itself.

## 10. Citation

If you use this repository, please cite the accompanying manuscript once its bibliographic details are available:

> *How Many Dimensions Are Enough? Investigating Bottleneck Sufficiency in Vision-Language Knowledge Distillation.* Manuscript.


