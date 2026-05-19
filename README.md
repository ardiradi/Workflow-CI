# 🔄 Heart Disease MLOps — CI/CD Workflow

<p align="center">
  <img src="https://img.shields.io/badge/MLflow-2.19.0-blue?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=github-actions&logoColor=white" alt="GitHub Actions">
  <img src="https://github.com/ardiradi/Workflow-CI/actions/workflows/ci.yml/badge.svg" alt="CI Status">
</p>

## 📋 Overview

Tahap **CI/CD Workflow** dari end-to-end MLOps pipeline. Mengimplementasikan automated training pipeline menggunakan **GitHub Actions** dan **MLflow Projects** yang di-trigger setiap kali ada push ke branch `main`.

## 🗂️ Struktur Direktori

```
├── MLProject                       # MLflow Project definition
├── modelling.py                    # Training script
├── conda.yaml                     # Conda environment specification
├── Tautan ke Docker Hub.txt       # Docker Hub reference
├── heart_disease_preprocessing/   # Preprocessed dataset
│   ├── X_train.csv
│   ├── X_val.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_val.csv
│   └── y_test.csv
└── .github/workflows/
    └── ci.yml                     # GitHub Actions CI pipeline
```

## ⚡ CI Pipeline

### Trigger
- Push ke branch `main` (file: `modelling.py`, `MLProject`, `conda.yaml`, `heart_disease_preprocessing/`)
- Manual trigger via `workflow_dispatch`

### Pipeline Steps

```mermaid
graph LR
    A[📥 Checkout] --> B[🐍 Setup Python 3.12]
    B --> C[📦 Install Dependencies]
    C --> D[🚀 mlflow run .]
    D --> E[📤 Upload Artifacts]
    E --> F[📦 Package]
```

1. **Checkout** — Clone repository
2. **Setup Python** — Python 3.12
3. **Install Dependencies** — MLflow, scikit-learn, pandas, numpy, matplotlib, seaborn
4. **Run MLflow Project** — `mlflow run . --env-manager=local` dengan parameter tuning
5. **Upload Artifacts** — Model, metrics, plots ke GitHub Artifacts
6. **Package** — Compress artifacts untuk distribusi

### MLflow Project Configuration

```yaml
# MLProject
name: Heart_Disease_MLOps
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 200}
      max_depth: {type: int, default: 10}
      min_samples_split: {type: int, default: 5}
      min_samples_leaf: {type: int, default: 2}
    command: "python modelling.py --n_estimators {n_estimators} ..."
```

### Training Parameters (CI)

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |

## 📊 CI Output

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **91.80%** |
| Test Precision | 91.87% |
| Test Recall | 91.80% |
| Test F1-Score | 91.81% |

### Artifacts Generated
- `mlruns/` — MLflow tracking data
- `artifacts/confusion_matrix.png` — Confusion matrix visualization
- `artifacts/classification_report.json` — Detailed classification metrics

## 🚀 Cara Menjalankan Lokal

```bash
# Clone repository
git clone https://github.com/ardiradi/Workflow-CI.git
cd Workflow-CI

# Install dependencies
pip install mlflow==2.19.0 scikit-learn pandas numpy matplotlib seaborn joblib

# Jalankan via MLflow Project
mlflow run . --env-manager=local \
  -P n_estimators=200 \
  -P max_depth=10

# Atau jalankan langsung
python modelling.py --n_estimators 200 --max_depth 10
```

## 🔗 Related Repositories

| Component | Repository |
|-----------|------------|
| 🔬 Experimentation | [Eksperimen_SML_ardir](https://github.com/ardiradi/Eksperimen_SML_ardir) |
| 📦 Model Building | [Membangun-Model-SML](https://github.com/ardiradi/Membangun-Model-SML) |
| 📊 Monitoring | [Monitoring-Logging-SML](https://github.com/ardiradi/Monitoring-Logging-SML) |

---

<p align="center">
  <b>Part of the Heart Disease MLOps Pipeline</b><br>
  Built as part of Dicoding — Membangun Sistem Machine Learning
</p>
