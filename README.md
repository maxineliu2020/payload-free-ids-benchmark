# Explainable Machine Learning for Payload-Free Intrusion Detection

[![DOI](https://zenodo.org/badge/1220463905.svg)](https://doi.org/10.5281/zenodo.19748632)

This repository contains the reproducible experiment package for the AIFE 2026 manuscript:

**Explainable Machine Learning for Payload-Free Intrusion Detection: A Reproducible Benchmark for Cybersecurity Education**

## Authors

- Yuanyuan Liu (Maxine), Nova Southeastern University; Johns Hopkins University
- Wei Li, Nova Southeastern University

## Project origin

This work was developed from the CISC 650 Computer Networks course project at Nova Southeastern University and revised into a reproducible conference research artifact.

## Repository structure

```text
payload-free-ids-benchmark/
├── code/                  # Modular Python experiment script
├── data/                  # Placeholder for optional public CICIDS2017 CSV files
├── docs/                  # Modular workflow documentation
├── figures/               # Figures generated from the validated run
├── outputs/               # Complete generated outputs from a validation run
├── results/               # CSV result tables used for reporting
├── CITATION.cff           # Citation metadata
├── LICENSE                # MIT License
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Scope note

The experiments use payload-free flow-level features. This repository does **not** claim to process authentic encrypted healthcare packet captures, and CICIDS2017 is **not** described as a healthcare dataset. The package is intended for reproducible academic review and cybersecurity education, not clinical deployment.

The public repository contains code, generated figures, result tables, and documentation. The submitted manuscript files are intentionally not included.

## What changed in v1.1.0

This version responds to Prof. Wei Li's feedback by strengthening the repository as a modular cybersecurity-education benchmark rather than a fixed three-model IDS experiment.

Key updates:

- Expanded supervised classifiers:
  - Logistic Regression
  - KNN
  - Naive Bayes
  - Linear SVM
  - MLP
  - Random Forest
  - XGBoost
- Added optional unsupervised baselines:
  - PCA Reconstruction
  - K-Means Distance
- Added modular pipeline structure:
  - dataset input
  - preprocessing
  - feature handling
  - classifier execution
  - metric generation
  - plotting
  - SHAP explainability
- Added support for user-supplied flow-level CSV files.
- Added validated output tables and figures.
- Added modular workflow documentation.

## Installation

Create a clean Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Validated reproducibility run

The following command was tested successfully and produced the included tables and figures:

```bash
python code/cisc650_payload_free_ids_benchmark.py --dataset synthetic --synthetic-samples 6000 --include-unsupervised --run-shap --output-dir outputs/validated_run_2026_04_27
```

The validation run completed successfully and generated:

- `results/synthetic_model_metrics.csv`
- `results/synthetic_feature_ablation.csv`
- `results/synthetic_robustness.csv`
- `figures/figure2_synthetic_roc_pr_curves.png`
- `figures/figure3_synthetic_model_f1.png`
- `figures/figure4_synthetic_feature_ablation_f1.png`
- `figures/figure5_synthetic_robustness_f1.png`
- `figures/figure6_synthetic_shap_summary.png`

## Optional CICIDS2017 benchmark

Large CICIDS2017 CSV files are not bundled in this repository because of file size. To run the optional CICIDS-style benchmark, place selected public CICIDS2017 CSV files in the `data/` folder and run:

```bash
python code/cisc650_payload_free_ids_benchmark.py --dataset cicids --data-dir data --output-dir outputs/cicids_run --max-rows-per-file 50000
```

## User-supplied CSV benchmark

Users may also provide their own flow-level CSV file:

```bash
python code/cisc650_payload_free_ids_benchmark.py --dataset user --input-csv data/my_flows.csv --label-column Label --output-dir outputs/user_run
```

## Main files

- `code/cisc650_payload_free_ids_benchmark.py`: modular reproducible experiment script.
- `docs/modular_workflow.md`: explanation of the modular workflow.
- `figures/figure1_modular_workflow.png`: modular workflow diagram.
- `figures/figure2_synthetic_roc_pr_curves.png`: ROC and precision-recall curves from the validated run.
- `figures/figure6_synthetic_shap_summary.png`: SHAP summary plot from the validated run.
- `results/synthetic_model_metrics.csv`: validated model comparison table.
- `results/synthetic_feature_ablation.csv`: validated feature-group ablation table.
- `results/synthetic_robustness.csv`: validated perturbation robustness table.

## License

The source code is released under the MIT License.

## Citation

Please cite this repository using the metadata in `CITATION.cff`.

Version-specific archived release DOI:

```text
10.5281/zenodo.19748633
```
