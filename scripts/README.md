# Scripts

This directory contains the Python scripts used for the machine-learning
analysis of inorganic phosphor emission wavelength.

The scripts implement:

- data preprocessing
- 10-fold cross-validation
- feature selection
- feature-selection stability analysis
- Random Forest regression
- Gradient Boosting regression
- model performance evaluation
- optimal feature-number selection
- generation of numerical output files and figures

---

## Main Analysis Scripts

### 1. `cv_first_threading_no_temp_disk.py`

This script performs the main feature-selection and model-evaluation workflow
using 10-fold cross-validation.

Key characteristics:

- 10-fold cross-validation
- `shuffle=True`
- `random_state=42`
- feature selection performed independently inside each training fold
- fold-wise feature scaling using `StandardScaler`
- Random Forest and Gradient Boosting regression
- evaluation using R², MAE, and RMSE
- fold-level selected feature subsets saved for reproducibility
- feature-selection stability calculated across CV folds
- identification of fully stable 10/10 feature subsets

The script evaluates the following 12 feature-selection methods:

#### Filter methods

- Pearson correlation
- Mutual Information
- ANOVA F-test
- t-test
- Wilcoxon rank-sum test

#### Wrapper methods

- Recursive Feature Elimination (RFE)
- Sequential Forward Selection (SFS)
- Sequential Backward Selection (SBS)

#### Embedded / importance-based methods

- LASSO
- Ridge
- Random Forest feature importance
- Gradient Boosting feature importance

---

### 2. `revised_all_k_stability_pipeline.py`

This script performs the all-k feature-selection sweep and calculates
fold-level feature-selection stability across the complete range of selected
feature-set sizes.

Key characteristics:

- evaluates feature-set sizes over a range of `k`
- uses the same 10-fold CV partitioning
- performs feature selection inside each CV training fold
- preserves the original selected feature subset from every fold
- calculates stability directly from the raw fold-level subsets
- evaluates Random Forest and Gradient Boosting performance
- saves performance, feature-frequency, and stability results
- verifies Pearson and ANOVA-F feature subsets at `k = 7`

The stability analysis includes:

- Jaccard similarity
- Dice similarity
- Hamming similarity
- Kuncheva index
- Spearman correlation
- Kendall correlation
- Pearson correlation
- Nogueira stability

---

## Cross-Validation Configuration

The main scripts use:

```python
KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
