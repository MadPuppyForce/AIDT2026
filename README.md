# Reproduction code for

**“A relational evaluation framework for recommender systems: assessing usefulness and coherence through item relationships”**
Marie Griffon, Nicolas Delestre, Maxime Gueriau
Artificial Intelligence for Digital Transformations, 2026

[Paper link]

---

# Overview

This repository contains the code used to reproduce the experiments presented in the paper:

> *A relational evaluation framework for recommender systems: assessing usefulness and coherence through item relationships.*

The goal of this work is to evaluate recommender systems through **item relationships**, using metrics based on:

* **Substitutability**
* **Complementarity**
* **Usefulness**
* **Intra-List Relationships (ILR)**

The repository provides scripts to:

1. Build train/test splits from raw BundleRec datasets
2. Compute substitutability and complementarity scores
3. Compute the IC score introduced in Sun, Z., Yang, J., Zhang, J., Bozzon, A.: Exploiting both vertical and horizontaldimensions of feature hierarchy for effective recommendation. Proc. AAAI Conf. Artif. Intell. 31(1) (2017)
4. Evaluate the substitutability and complementarity scores
5. Evaluate the ILR and Usefulness metrics

---

# Repository structure

```
mon-papier-repro/
│
├── pyproject.toml
├── uv.lock
├── README.md
│
├── data/
│   ├── raw/              # original datasets
│   ├── processed/        # processed datasets (train/test splits)
│   └── README.md         # dataset download instructions
│
├── src/
│   ├── metrics/
│   │   └── usefulness_ilr.py
│   │
│   └── scores/
│       ├── ic.py
│       └── substitution_complementarity_scores.py
│
├── scripts/
│   ├── build_splits.py
│   ├── run_scores_experiment.py
│   └── run_metrics_experiment.py
│
└── results/
```

---

# Environment setup

The project uses **uv** to manage the Python environment.

## Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create the virtual environment

```
uv venv
```

## Install dependencies

```
uv sync
```

---

# Dataset installation

The datasets used in this work come from the **BundleRec dataset** introduced in:

* Sun et al., Revisiting Bundle Recommendation: Datasets, Tasks, Challenges and Opportunities for Intent-Aware Product Bundling, SIGIR 2022.
* Sun et al., Revisiting Bundle Recommendation for Intent-aware Product Bundling, ACM TORS 2024.

See:

https://github.com/BundleRec/bundle_recommendation

Detailed instructions are provided in:

```
data/README.md
```

The expected directory structure is:

```
data/raw/
    clothing/
        session_item.csv
        bundle_item.csv
        session_bundle.csv

    electronic/
        session_item.csv
        bundle_item.csv
        session_bundle.csv

    food/
        session_item.csv
        bundle_item.csv
        session_bundle.csv
```

---

# Reproducing the experiments

All random seeds are fixed to **42** to ensure reproducibility.

Experiments should be run in the following order.

---

## 1. Build train/test splits

```
uv run python scripts/build_splits.py
```

This script:

* splits sessions into **train (80%)** and **test (20%)**
* applies a swap-based optimization to improve the balance of unique item occurrences across splits
* saves the processed data into:

```
data/processed/<domain>/
```

---

## 2. Run scores experiments

```
uv run python scripts/run_scores_experiment.py
```
This scrip evaluates the ability of our substitutability and complementarity scores to identify meaningful relationships.
The identified relationships are compared against a ground truth constructed from the annotated bundles.

Outputs include:

* substitutability and complementarity score tables
* precision and recall summary

Outputs are saved in:

```
results/<domain>/
results/
```

---

## 3. Run ILR and Usefulness experiments

```
uv run python scripts/run_metrics_experiment.py
```

This script evaluates the behavior of the metrics under controlled corruption.

Outputs include:

* aggregated metric tables
* plots of metric degradation vs corruption level

Results are saved in:

```
results/<domain>/
results/
```

---

# Hardware configuration

All experiments were run on:

* **OS:** Ubuntu 22.04
* **CPU:** Intel Core i9-12900K
* **RAM:** 64 GB
* **Python:** 3.10.12

---

# Citation

If you use this code, please cite:

```
@article{griffon2026relational,
  title={A relational evaluation framework for recommender systems: assessing usefulness and coherence through item relationships},
  author={Griffon, Marie and Delestre, Nicolas and Gueriau, Maxime},
  journal={Artificial Intelligence for Digital Transformations},
  year={2026}
}
```

You should also cite the dataset:

```
@inproceedings{sun2022revisiting,
  title={Revisiting Bundle Recommendation: Datasets, Tasks, Challenges and Opportunities for Intent-Aware Product Bundling},
  author={Sun, Zhu and Yang, Jie and Feng, Kaidong and Fang, Hui and Qu, Xinghua and Ong, Yew Soon},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2022}
}

@article{sun2024revisiting,
  title={Revisiting Bundle Recommendation for Intent-aware Product Bundling},
  author={Sun, Zhu and Feng, Kaidong and Yang, Jie and Fang, Hui and Qu, Xinghua and Ong, Yew-Soon and Liu, Wenyuan},
  journal={ACM Transactions on Recommender Systems},
  year={2024},
  publisher={ACM New York, NY}
}
```

---

# License

This project is released under the MIT License.
