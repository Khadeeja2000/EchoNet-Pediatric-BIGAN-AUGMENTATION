# EchoNet-Pediatric-BIGAN-AUGMENTATION

## Preprocessing (data-preprocessing branch)

This branch adds a lightweight preprocessing pipeline that samples a balanced subset of the EchoNet Pediatric dataset and outputs a manifest CSV for downstream augmentation (e.g., BiGAN). The balance is done by sex and age bins using metadata provided in each view's `FileList.csv`.

Important: We do not commit raw videos. See `.gitignore`. The manifest references local file paths for reproducibility.

### Files added
- `preprocessing/config.yaml` — configure subset size, views, balance strategy, and paths
- `preprocessing/preprocess.py` — generates a balanced manifest CSV of selected samples
- `data/processed/manifest.csv` — generated list of selected samples with metadata
- `requirements.txt` — Python dependencies

### Quick start
1. Create a virtualenv and install deps:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Ensure the dataset exists at the path configured in `preprocessing/config.yaml` under `paths.dataset_root`.
3. Run preprocessing:
   ```bash
   python preprocessing/preprocess.py --config preprocessing/config.yaml
   ```
4. Output: `data/processed/manifest.csv` with columns: view, file_name, file_path, ef, sex, age, weight, height, split, age_bin.

### Notes
- The dataset CSVs do not include subject names; balancing is done using `Sex` and `Age` only.
- Adjust `subset.total_samples`, `subset.views`, and `subset.age_bins` in the config as needed.