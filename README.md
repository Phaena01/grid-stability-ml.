# 🔌 Predictive Stability Classification for Smart Electrical Grids
## Using Supervised Machine Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)]()

**COEN807 – Machine Learning | Term Project**

*Ahmadu Bello University (ABU), Zaria · Department of Computer Engineering*

[Problem Statement](#-problem-statement) · [Dataset](#-dataset) · [Installation](#️-installation) · [Quickstart](#-quickstart) · [Results](#-results) · [Workflow](#-execution-workflow) · [Limitations](#️-limitations--ethical-notes) · [References](#-references)

</div>

---

## 📋 Project Overview

Nigeria's national electricity grid operates under chronic instability — with actual average generation of only **4,000–5,000 MW against 13,000 MW installed capacity**, and the 330 kV transmission network experiencing frequent cascading outages. As smart grid technologies are adopted and price-responsive demand management is introduced, the dynamic interactions between producers and consumers create complex, non-linear stability behaviour that conventional eigenvalue analysis cannot assess in real time.

This project presents an **end-to-end supervised machine learning pipeline** to classify the stability status of a simulated four-node smart electrical grid. Given 12 measurable electrical parameters — nodal reaction times (τ), power generated and consumed (p), and price elasticity coefficients (g) — six classifiers are trained and rigorously compared to predict whether a grid configuration is:

| Label | Class | Physical Meaning |
|-------|-------|-----------------|
| `0` | **UNSTABLE** | Eigenvalues of the linearised Jacobian have **positive** real parts → perturbations grow → potential fault |
| `1` | **STABLE** | All eigenvalues have **negative** real parts → perturbations decay → grid self-corrects |

### 🏆 Best Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **90.00%** |
| **Weighted F1-Score** | **0.9001** |
| **ROC-AUC** | **0.9717** |
| **Best Model** | Gradient Boosting (Tuned via GridSearchCV) |
| **False Negative Rate** | 7.70% *(requires human verification layer)* |

---

## 🎯 Problem Statement

> **Binary Classification Task:**  
> Given a set of 12 measurable electrical grid parameters, train a supervised classifier to predict whether the resulting grid state will be **STABLE (1)** or **UNSTABLE (0)**.

The critical performance requirement is **high recall for the Unstable class**, since a missed unstable prediction (false negative) corresponds to an undetected fault that could trigger a cascading blackout. This makes the choice of evaluation metric critical — `Accuracy` alone is misleading under the 63/37 class imbalance; `Weighted F1-Score` and `ROC-AUC` are the primary metrics.

### Project Objectives

1. Acquire, explore, and characterise the Electrical Grid Stability Simulated Dataset
2. Engineer six domain-informed features from the 12 raw variables
3. Implement, cross-validate, and compare six supervised learning classifiers
4. Apply GridSearchCV hyperparameter optimisation to the best-performing model
5. Evaluate using metrics appropriate for imbalanced binary classification
6. Interpret results in the Nigerian power system context and critically evaluate limitations

---

## 📊 Dataset

**Electrical Grid Stability Simulated Dataset** — Arzamasov et al. (2018)

| Property | Value |
|----------|-------|
| **Source** | [UCI ML Repository (ID 471)](https://archive.ics.uci.edu/dataset/471) · [Kaggle](https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability) |
| **Reference** | Arzamasov, V., Böhm, K., Jochem, P. (2018). DOI: [10.24432/C5PP49](https://doi.org/10.24432/C5PP49) |
| **Instances** | 10,000 |
| **Raw Features** | 12 (all continuous numerical) |
| **Engineered Features** | 6 (domain-informed aggregates) |
| **Total Features Used** | **18** |
| **Target Column** | `stabf` |
| **Class Distribution** | Unstable (0): 6,300 (63.0%) · Stable (1): 3,700 (37.0%) |
| **Missing Values** | **None** |

### Raw Feature Descriptions

The dataset models a **star-topology 4-node grid** — one producer (Node 1) and three consumers (Nodes 2–4):

| Feature Group | Variables | Physical Range | Description |
|--------------|-----------|---------------|-------------|
| **Reaction Times** | `tau1` – `tau4` | 0.5 – 10.0 s | Time for each node to adjust power consumption/generation in response to a price deviation signal. Shorter τ = faster response = more adaptive network. |
| **Producer Power** | `p1` | 0.5 – 2.0 p.u. | Power generated by Node 1 (positive = supplying the network). |
| **Consumer Loads** | `p2`, `p3`, `p4` | −2.0 – −0.5 p.u. | Power consumed at each load node (negative = drawing from network). |
| **Price Elasticity** | `g1` – `g4` | 0.05 – 1.00 | Degree to which each node adjusts demand per unit of price change. Higher g = more elastic = stronger demand response. |
| **Target** | `stabf` | 0 or 1 | Binary stability label derived from eigenvalue analysis of the linearised system Jacobian. |

### Engineered Features (12 → 18)

Six domain-informed features were constructed to give models higher-level physical abstractions:

| Feature | Formula | MDI Importance | Physical Rationale |
|---------|---------|---------------|--------------------|
| `total_demand` | p₂ + p₃ + p₄ | **0.2903 (29.0%)** ★ | Total consumer load — primary driver of generation stress. Top predictor. |
| `net_power` | p₁ + total\_demand | **0.1191 (11.9%)** | Net generation–demand balance. Negative = shortfall = instability signal. |
| `mean_tau` | mean(τ₁–τ₄) | 0.0490 | Average network response speed. High mean_tau = sluggish adaptation. |
| `tau_std` | std(τ₁–τ₄) | — | Heterogeneity of response times. High std = uneven adaptation capacity. |
| `mean_g` | mean(g₁–g₄) | — | Average price elasticity across all nodes. |
| `response_ratio` | mean\_g ÷ (mean\_τ + ε) | **0.0847 (8.5%)** | Elasticity-to-speed ratio — captures how quickly AND how strongly the network compensates for price shocks. |

> **Key finding:** 3 of the top 7 features by importance are engineered, collectively contributing **49.4%** of Random Forest MDI importance. This validates domain-informed feature engineering as the highest-impact design decision — outperforming hyperparameter tuning.

---

## 🏗️ Repository Structure

```
grid-stability-ml/
│
├── 📄 README.md                           ← This file — project overview
├── 📄 requirements.txt                    ← Python package dependencies
├── 📄 setup.py                            ← Package installation
├── 📄 .gitignore                          ← Version control exclusions
├── 📄 LICENSE                             ← MIT License
│
├── 📁 src/                                ← Modular source code
│   ├── main.py                            ← Pipeline entry point ← START HERE
│   ├── config.py                          ← Central config (paths, hyperparams, seed)
│   │
│   ├── 📁 data/
│   │   ├── loader.py                      ← Dataset loading & synthetic generation
│   │   └── preprocessor.py               ← Stratified split & StandardScaler
│   │
│   ├── 📁 features/
│   │   └── engineer.py                   ← Domain feature engineering (12 → 18)
│   │
│   ├── 📁 models/
│   │   ├── train.py                      ← Model definitions & 5-fold CV
│   │   ├── tune.py                       ← GridSearchCV hyperparameter tuning
│   │   └── evaluate.py                   ← Metrics, reports & CSV export
│   │
│   └── 📁 visualization/
│       └── plots.py                      ← EDA & results dashboards (PNG)
│
├── 📁 notebooks/
│   └── grid_stability_analysis.ipynb     ← Interactive Jupyter walkthrough
│
├── 📁 data/
│   ├── README.md                         ← Dataset download instructions
│   └── raw/                              ← Place downloaded CSV here
│
├── 📁 outputs/                           ← Auto-generated (not committed)
│   ├── figures/                          ← EDA & results plots (PNG)
│   └── results/
│       └── results_summary.csv          ← Model comparison table
│
├── 📁 docs/
│   └── data_description.md              ← Full feature documentation
│
├── 📁 tests/
│   └── test_pipeline.py                 ← Unit tests (pytest)
│
└── 📁 scripts/
    ├── run_pipeline.sh                  ← One-command full pipeline (Linux/Mac)
    └── download_data.sh                 ← Kaggle dataset download helper
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.10** or higher
- `pip` (comes with Python) or `conda`
- ~500 MB disk space (for dependencies)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/grid-stability-ml.git
cd grid-stability-ml
```

### Step 2 — Create a Virtual Environment *(strongly recommended)*

```bash
# Using venv (standard)
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows CMD
```

```bash
# OR using conda
conda create -n grid-ml python=3.10 -y
conda activate grid-ml
```

### Step 3 — Install All Dependencies

```bash
pip install -r requirements.txt
```

> **requirements.txt includes:** scikit-learn 1.3.0, numpy 1.24.4, pandas 2.0.3, matplotlib 3.7.2, seaborn 0.12.2, jupyter 1.0.0, pytest 7.4.3

### Step 4 — Install Package in Development Mode *(optional)*

```bash
pip install -e .
```

---

## 📥 Dataset Download

### Option A — Kaggle CLI *(recommended)*

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Place your Kaggle API key at ~/.kaggle/kaggle.json
#    (Generate at: kaggle.com → Account → API → Create New Token)

# 3. Download and extract
kaggle datasets download pcbreviglieri/smart-grid-stability
unzip smart-grid-stability.zip -d data/raw/
```

Expected file location: `data/raw/smart_grid_stability_augmented.csv`

### Option B — UCI ML Repository *(manual)*

1. Visit: [https://archive.ics.uci.edu/dataset/471](https://archive.ics.uci.edu/dataset/471)
2. Download `smart_grid_stability_augmented.csv`
3. Place in `data/raw/smart_grid_stability_augmented.csv`

### Option C — No Download Required *(synthetic replica)*

```bash
# No setup needed — just run the pipeline directly
python src/main.py
```

If no CSV is found in `data/raw/`, the pipeline **automatically generates a synthetic 10,000-instance replica** matching the published feature distributions and 63/37 class ratio (Arzamasov et al., 2018). The synthetic dataset is fully sufficient for demonstrating the complete pipeline methodology.

---

## 🚀 Quickstart

```bash
# Run the complete pipeline with synthetic data (no download required)
python src/main.py

# Run with real dataset
python src/main.py --data data/raw/smart_grid_stability_augmented.csv
```

All outputs are written to `outputs/figures/` (PNG plots) and `outputs/results/results_summary.csv`.

---

## 🔄 Execution Workflow

The pipeline runs in eight sequential stages, each handled by a dedicated module:

```
INPUT
  └── CSV file  OR  synthetic replica (auto-generated)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 │ Data Loading & Quality Check          src/data/loader.py │
│  ─────── │ • Load CSV or generate synthetic data                    │
│           │ • Verify: 0 missing values, 0 duplicates, class balance  │
└───────────┴────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 │ Feature Engineering (12 → 18)    src/features/engineer.py│
│  ─────── │ • Add: total_demand, net_power, mean_tau                 │
│           │ • Add: tau_std, mean_g, response_ratio                   │
└───────────┴────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 │ Preprocessing                  src/data/preprocessor.py  │
│  ─────── │ • Stratified 80/20 train/test split (RANDOM_STATE=42)   │
│           │ • StandardScaler: FIT on train → APPLY to both sets     │
│           │   (no data leakage)                                      │
└───────────┴────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 │ EDA Visualisations               src/visualization/plots │
│  ─────── │ • Class distribution, correlation heatmap                │
│           │ • Feature KDE by class, boxplots, variance bar chart     │
│           │ OUTPUT → outputs/figures/eda_dashboard.png               │
└───────────┴────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5 │ Model Training + 5-Fold Stratified CV  src/models/train  │
│  ─────── │ • Logistic Regression  (linear baseline)                 │
│           │ • Decision Tree        (depth-limited, non-linear)       │
│           │ • Random Forest        (bagging ensemble)                │
│           │ • SVM (RBF Kernel)     (kernel-based)                    │
│           │ • K-Nearest Neighbours (instance-based)                  │
│           │ • Gradient Boosting    (boosting ensemble)               │
│           │ • Reports CV accuracy ± std deviation per model          │
└───────────┴────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6 │ Hyperparameter Tuning (GridSearchCV)   src/models/tune   │
│  ─────── │ • Best base model: Gradient Boosting                     │
│           │ • Grid: n_estimators=[100,200], max_depth=[3,5],         │
│           │         learning_rate=[0.05, 0.10, 0.15]                 │
│           │ • cv=3, scoring=f1_weighted, 12 total configurations     │
│           │ • Best: n_estimators=200, max_depth=5, lr=0.10           │
└───────────┴────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 7 │ Evaluation & Reporting          src/models/evaluate.py   │
│  ─────── │ • Test set metrics: Accuracy, Precision, Recall, F1, AUC │
│           │ • Per-class classification report (best model)           │
│           │ • Confusion matrix with FNR/FPR analysis                 │
│           │ OUTPUT → outputs/results/results_summary.csv             │
└───────────┴────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 8 │ Results Visualisations          src/visualization/plots  │
│  ─────── │ • Model comparison bar chart (Accuracy vs F1)            │
│           │ • ROC curves (all models + tuned GB)                     │
│           │ • Confusion matrix heatmap (best model)                  │
│           │ • 5-fold CV boxplots, feature importance, AUC bars       │
│           │ OUTPUT → outputs/figures/model_results_dashboard.png     │
└───────────┴────────────────────────────┘

OUTPUT SUMMARY
  outputs/figures/eda_dashboard.png
  outputs/figures/model_results_dashboard.png
  outputs/results/results_summary.csv
```

---

## 🖥️ Running the Project

### Full Pipeline

```bash
python src/main.py
```

### With Real Dataset

```bash
python src/main.py --data data/raw/smart_grid_stability_augmented.csv
```

### Run Individual Stages

```bash
python src/main.py --stage eda      # EDA visualisations only
python src/main.py --stage train    # Model training & cross-validation only
python src/main.py --stage tune     # Hyperparameter tuning only
python src/main.py --stage eval     # Evaluation & results CSV only
```

### Shell Script (Linux/macOS — sets up environment and runs everything)

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

### Jupyter Notebook (interactive, step-by-step)

```bash
jupyter notebook notebooks/grid_stability_analysis.ipynb
```

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 📈 Results

### Model Comparison — Held-Out Test Set (N = 2,000)

> All metrics computed exclusively on the 2,000-instance test partition, never seen during training or tuning.  
> **Weighted averaging** applied to Precision, Recall, F1 to account for 63/37 class imbalance.

| Model | CV Acc ± σ | Test Acc | Precision (W) | Recall (W) | F1 (W) | AUC |
|-------|:----------:|:--------:|:-------------:|:----------:|:------:|:---:|
| Logistic Regression | 0.8731 ± 0.006 | 0.8580 | 0.8574 | 0.8580 | 0.8576 | 0.9401 |
| Decision Tree | 0.8561 ± 0.008 | 0.8595 | 0.8586 | 0.8595 | 0.8588 | 0.8728 |
| Random Forest | 0.8860 ± 0.006 | 0.8865 | 0.8862 | 0.8865 | 0.8863 | 0.9599 |
| SVM (RBF Kernel) | 0.8881 ± 0.005 | 0.8905 | 0.8901 | 0.8905 | 0.8903 | 0.9627 |
| KNN (k=5) | 0.8489 ± 0.009 | 0.8410 | 0.8399 | 0.8410 | 0.8386 | 0.9131 |
| Gradient Boosting | 0.9021 ± 0.005 | 0.9030 | 0.9027 | 0.9030 | 0.9028 | 0.9704 |
| **GB Tuned ★** | — | **0.9000** | **0.9002** | **0.9000** | **0.9001** | **0.9717** |

> **Why GB wins:** Boosting's sequential residual correction outperforms bagging (RF) when the stability boundary involves moderate non-linearity. The 2.3-pt AUC gap between SVM (0.9627) and LR (0.9401) confirms the boundary is non-linear — the RBF kernel correctly maps features to a linearly separable manifold.

### Best Model — Class-Level Report (GB Tuned ★)

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Unstable (0) | 0.9226 | 0.9183 | 0.9204 | 1,260 |
| Stable (1) | 0.8619 | 0.8689 | 0.8654 | 740 |
| **Weighted Avg** | **0.9002** | **0.9000** | **0.9001** | **2,000** |

### Confusion Matrix — GB Tuned ★

```
                   Predicted: Unstable    Predicted: Stable
                  ┌──────────────────────┬─────────────────┐
Actual Unstable   │   TN = 1,157  (91.8%)│  FP = 103  (8.2%)│
                  ├──────────────────────┼─────────────────┤
Actual Stable     │   FN =    97  (13.1%)│  TP = 643 (86.9%)│
                  └──────────────────────┴─────────────────┘

  ⚠ False Negative Rate (Unstable missed as Stable) = 97 / 1,260 = 7.70%
     → Requires a human verification layer before any operational deployment
```

### Top Feature Importances (Random Forest MDI)

| Rank | Feature | Importance | Type |
|:----:|---------|:----------:|:----:|
| 1 ★ | `total_demand` | **0.2903 (29.0%)** | Engineered |
| 2 ★ | `net_power` | **0.1191 (11.9%)** | Engineered |
| 3 ★ | `response_ratio` | **0.0847 (8.5%)** | Engineered |
| 4 | `p4` | 0.0602 (6.0%) | Raw |
| 5 | `p3` | 0.0561 (5.6%) | Raw |
| 6 | `p2` | 0.0529 (5.3%) | Raw |
| 7 ★ | `mean_tau` | 0.0490 (4.9%) | Engineered |

---

## 🔁 Reproducibility

All experiments are **fully deterministic** with the following fixed configuration:

| Setting | Value |
|---------|-------|
| `RANDOM_STATE` | **42** (set in `src/config.py`, propagated to all modules) |
| Train/test split | 80 % / 20 %, **stratified** by class label |
| CV strategy | `StratifiedKFold(n_splits=5, shuffle=True)` |
| Scaler fit | Train set only → no data leakage |
| Tuning scorer | `f1_weighted` |
| Tuning CV folds | 3 |
| Python | 3.10+ |
| scikit-learn | 1.3.0 |

To reproduce all results from scratch:

```bash
git clone https://github.com/<user>/grid-stability-ml.git
cd grid-stability-ml
pip install -r requirements.txt
python src/main.py               # ← single command reproduces everything
```

---

## ⚠️ Limitations & Ethical Notes

### Technical Limitations

| Limitation | Detail |
|-----------|--------|
| **Synthetic data** | Not validated on real TCN SCADA measurements. Real grid accuracy is unknown without empirical testing. |
| **Steady-state only** | Models capture static stability snapshots. Transient fault dynamics (fault-induced oscillations) require time-series models (LSTM, TCN). |
| **4-node topology** | The star topology is a major simplification of Nigeria's real 330 kV ring/mesh network with hundreds of nodes. |
| **No uncertainty quantification** | Models produce point predictions only. Safety-critical deployment requires calibrated confidence intervals (Bayesian inference, conformal prediction). |
| **Partial black box** | Gradient Boosting requires SHAP (SHapley Additive exPlanations) analysis for per-prediction interpretability before regulatory deployment. |

### Ethical Considerations

> ⚠️ **No Autonomous Grid Actuation:**  
> This model is a **decision-support tool only**. The 7.70% false negative rate (unstable grids predicted as stable) is operationally significant. **Human verification is mandatory** before any protective action is triggered by model output. No automated grid control should be based solely on this classifier.

> ⚠️ **Load-Shedding Equity:**  
> If model predictions influence load-shedding decisions, fairness constraints must be applied across LGAs and demographic groups to prevent algorithmic bias against rural or low-income communities.

> ⚠️ **Synthetic Data Bias:**  
> The dataset encodes the assumptions of the mathematical simulation model as implicit ground truth. If the eigenvalue stability criterion is an imperfect proxy for real grid behaviour, the classifier will inherit those biases systematically.

---

## 🔮 Future Work

- [ ] **LSTM / Temporal CNNs** for transient stability time-series dynamics
- [ ] **Graph Neural Networks (GNNs)** for complex real-world mesh grid topologies
- [ ] **SHAP explainability** values for NERC regulatory compliance and operator trust
- [ ] **Empirical validation** on real PMU (Phasor Measurement Unit) SCADA data from TCN
- [ ] **Online incremental learning** to adapt to evolving grid configurations and growing renewable penetration
- [ ] **Class-imbalance correction** (SMOTE, cost-sensitive learning) for more severe real-world imbalance scenarios

---

## 📚 References

1. Arzamasov, V., Böhm, K., & Jochem, P. (2018). *Towards Concise Models of Grid Stability*. Proc. IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm). DOI: [10.24432/C5PP49](https://doi.org/10.24432/C5PP49)

2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

4. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

5. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273–297.

6. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS), 30.

7. Akorede, M. F., Abubakar, A. S., Aliyu, U. O., & Jimoh, A. A. (2017). *Present challenges and the future of electric power industry in Nigeria*. International Journal of Electrical Power & Energy Systems, 76(1), 1–12.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for full terms.

---

<div align="center">

*COEN807 Term Project · Ahmadu Bello University (ABU), Zaria · Department of Computer Engineering*  
*Dataset: Arzamasov et al. (2018) DOI: 10.24432/C5PP49*

</div>
