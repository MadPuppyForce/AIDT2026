## Dataset Installation

The datasets used in this project originate from the official Bundle Recommendation repository:

https://github.com/BundleRec/bundle_recommendation

If you use these datasets, please cite:

* Sun et al., Revisiting Bundle Recommendation: Datasets, Tasks, Challenges and Opportunities for Intent-Aware Product Bundling, SIGIR 2022.
* Sun et al., Revisiting Bundle Recommendation for Intent-aware Product Bundling, ACM TORS 2024.

(See full BibTeX entries below.)

---

## Required Files

For each domain, you must download the following three files:

* `session_item.csv`
* `bundle_item.csv`
* `session_bundle.csv`

There are three available domains:

* `food`
* `clothing`
* `electronics`

In the official repository, these files are located in:

```
bundle_recommendation/dataset/<domain>/
```

Each domain has its own dedicated folder containing the three `.csv` files.

---

## Download Procedure

1. Go to the official repository:
   https://github.com/BundleRec/bundle_recommendation

2. Navigate to the `dataset/` directory.

3. For each domain you intend to use (`food`, `clothing`, `electronics`):

   * Open the corresponding folder.
   * Download the three required `.csv` files:

     * `session_item.csv`
     * `bundle_item.csv`
     * `session_bundle.csv`

---

## Local Directory Structure

After downloading, place the files in the following structure:

```
project_root/
│
├── data/
│   ├── raw/
│   |    ├── clothing/
│   |    │   ├── session_item.csv
│   |    │   ├── bundle_item.csv
│   |    │   └── session_bundle.csv
│   |    │
│   |    ├── electronic/
│   |    │   ├── session_item.csv
│   |    │   ├── bundle_item.csv
│   |    │   └── session_bundle.csv
│   |    │
│   |    └── food/
│   |        ├── session_item.csv
│   |        ├── bundle_item.csv
│   |        └── session_bundle.csv
│   |
|   └── ...
|
└── ...
```

Important:
* Create the `data/raw/` folder manually before placing the files (it is not included in the repository).
* Do not rename the files.
* Keep the directory structure exactly as shown to ensure reproducibility.

## Citation

```bibtex
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
