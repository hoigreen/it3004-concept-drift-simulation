# IT3004 - Intrusion Detection System under Concept Drift

**Course:** IT3004.CH191 - Advanced Topics in Intrusion Detection Systems  
**Assignment:** Reproducing IDS Performance Degradation and Mitigation (Lecture 06)

This project builds a **full end-to-end experimental pipeline** to:

1. Reproduce IDS performance degradation caused by **concept drift**, and
2. Mitigate the degradation using **drift-aware retraining**.

---

## Objectives

Real-world network traffic is a **non-stationary data stream**. As behaviors evolve and **new attack types emerge**, models trained on historical data can suffer severe performance drops.

This project focuses on:

- **Concept Drift (Novel Attacks)**: new attack types appear in later periods and are **absent from initial training data**.
- **Performance degradation reproduction**: Logistic Regression shows sharp drops in **Recall, F1, and PR-AUC** for attack detection.
- **Mitigation strategy**: **sliding window retraining**, triggered **only when drift is detected**.

---

## Datasets (Kaggle via `kagglehub`)

| Dataset | Kaggle Slug |
|---|---|
| NSL-KDD | `hassan06/nslkdd` |
| UNSW-NB15 | `harshwardhanbhangale/unsw-complete-dataset` |
| CSE-CIC-IDS2018 | `dhoogla/csecicids2018` |

All datasets are automatically downloaded and cached locally using **`kagglehub`**.

> Note: CIC-IDS2018 is large; sampling is recommended.

---

## Experimental Design

### Task Definition

- **Binary classification**
  - `0` -> Benign
  - `1` -> Attack
- **Model**: Logistic Regression
- **Preprocessing**:
  - `StandardScaler` for numeric features
  - One-Hot Encoding for categorical features

### Drift Simulation

Data is split into sequential **time chunks**:

| Chunk | Content |
|---|---|
| Chunk 1-2 | Benign + Known attacks |
| Chunk 3-5 | Benign + Known attacks + **New (unseen) attacks** |

New attack types are **excluded from early chunks**, forcing the model to generalize poorly once drift occurs.

### Baseline IDS

- Trained **once** on Chunk 1
- Evaluated sequentially on later chunks
- **No retraining**

Expected behavior:

- Attack Recall drops sharply
- F1-Score drops sharply

---

## Drift Detection

### KS-Test (Feature Distribution Drift)

- Two-sample Kolmogorov-Smirnov test
- Applied per numeric feature
- Drift is detected when:

```
(# drifted features) / (total features) > threshold
```

### Error-Based Drift (river)

Implemented via **`river`**:

- **ADWIN**
- **DDM**
- **EDDM**

Input stream:

```
error_t = 1 if y_true != y_pred else 0
```

Drift is detected when the error distribution changes significantly.

---

## Mitigation Strategy: Sliding Window Retraining

- **Window size (W)**: number of most recent chunks
- **Retraining is triggered only if drift is detected**
- Model is retrained from scratch using:

```
chunks[t-W ... t-1]
```

This balances:

- Adaptability to new attacks
- Computational efficiency

---

## Evaluation Metrics

Metrics are computed **per chunk**:

- Accuracy
- Precision (Attack class)
- **Recall (Attack class)**
- **F1-Score (Attack class)**
- ROC-AUC
- **PR-AUC**
- Confusion Matrix (TN, FP, FN, TP)

> Due to class imbalance, **Recall, F1, and PR-AUC** are emphasized.

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Project

### 1. (Optional) Pre-download Datasets

```bash
python scripts/download_kaggle.py --all
```

Or download a single dataset:

```bash
python scripts/download_kaggle.py --only nslkdd
```

### 2. Exploratory Data Analysis (EDA)

```bash
python run.py eda --dataset nslkdd --out outputs/nslkdd_eda
python run.py eda --dataset unsw --out outputs/unsw_eda
python run.py eda --dataset cicids2018 --sample 300000 --out outputs/cic_eda
```

**EDA outputs**

- `eda_summary.json`
- `numeric_describe.csv`
- Label distribution plots
- Top attack type plots
- Correlation heatmaps

### 3. Concept Drift Experiment + Mitigation

**KS-Test Drift Detection**

```bash
python run.py exp --dataset nslkdd --detector ks --window 2 --chunks 5 --out outputs/nslkdd_exp
```

**ADWIN Drift Detection**

```bash
python run.py exp --dataset unsw --detector adwin --window 2 --chunks 5 --out outputs/unsw_exp
```

**CIC-IDS2018 (sampling)**

```bash
python run.py exp --dataset cicids2018 --detector ks --window 2 --chunks 5 --sample 300000 --out outputs/cic_exp
```

---

## Outputs

Each experiment produces:

- `drift_meta.json`
- `metrics_by_chunk.csv`

The `figures/` directory includes:

- `recall_attack_over_chunks.png`
- `f1_attack_over_chunks.png`
- `pr_auc_over_chunks.png`
- `roc_auc_over_chunks.png`
- `accuracy_over_chunks.png`

`drift_meta.json` records high-level information about the generated stream:

- Number of chunks
- Approximate chunk size
- Benign fraction in the first and last chunks
- Drift strength used to vary class balance over time

---
