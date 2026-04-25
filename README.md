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
├── code/          # Python experiment script
├── data/          # Placeholder for optional public CICIDS2017 CSV files
├── figures/       # Figures used in the manuscript
├── outputs/       # Generated outputs when users rerun the experiment
├── results/       # Result tables used in the manuscript
├── CITATION.cff   # Citation metadata
├── LICENSE        # MIT License
├── README.md      # Project documentation
└── requirements.txt
```

## Scope note

The experiments use payload-free flow-level features. This repository does not claim to process authentic encrypted healthcare packet captures, and CICIDS2017 is not described as a healthcare dataset. The package is intended for reproducible academic review and cybersecurity education, not clinical deployment.

## Installation

Create a clean Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce the synthetic stress-test experiment

```bash
python code/cisc650_payload_free_ids_benchmark.py --synthetic-only --output-dir outputs
```

The script will generate synthetic IDS data, train Logistic Regression, Random Forest, and XGBoost models, and save result tables and figures to `outputs/`.

## Optional CICIDS2017 benchmark

Large CICIDS2017 CSV files are not bundled in this repository because of file size. To run the optional CICIDS2017 benchmark, place selected public CICIDS2017 CSV files in the `data/` folder and run:

```bash
python code/cisc650_payload_free_ids_benchmark.py --data-dir data --output-dir outputs --max-rows-per-file 50000
```

## Main files

- `code/cisc650_payload_free_ids_benchmark.py`: reproducible experiment script.
- `figures/figure1_roc_pr_curves.png`: ROC and precision-recall curves.
- `figures/figure2_shap_summary.png`: SHAP summary plot.
- `results/synthetic_results_table.csv`: synthetic stress-test result table.
- `paper/`: ACM-style manuscript files.

## License

The source code is released under the MIT License.

## Citation

Please cite this repository using the information in `CITATION.cff`. A Zenodo DOI can be added after the first archived release.
