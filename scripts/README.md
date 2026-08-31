# Analysis Scripts

This directory contains the Python scripts used for feature selection,
cross-validation, model evaluation, stability analysis, descriptor correlation
analysis, and error analysis for the inorganic phosphor emission-wavelength
prediction study.

## 1. Feature Selection and Model Evaluation

**Recommended file name:** `feature_selection_model_evaluation.py`

This script performs the main feature-selection and machine-learning analysis.

### Main workflow

- Loads and preprocesses the single-dopant phosphor dataset.
- Uses `Emission max. (nm)` as the prediction target.
- Removes metadata, identifier, non-numeric, and constant columns.
- Handles missing numerical values using median imputation.
- Uses 10-fold cross-validation with:
  - `n_splits = 10`
  - `shuffle = True`
  - `random_state = 42`
- Fits `StandardScaler` within each training fold.
- Performs feature selection independently inside each CV training fold.
- Evaluates Random Forest (RF) and Gradient Boosting (GB) regressors.

### Feature-selection methods

Twelve feature-selection methods are evaluated:

**Filter methods**
- Pearson correlation
- Mutual Information
- ANOVA F-test
- t-test
- Wilcoxon rank-sum

**Wrapper methods**
- Recursive Feature Elimination (RFE)
- Sequential Forward Selection (SFS)
- Sequential Backward Selection (SBS)

**Embedded / importance-based methods**
- LASSO
- Ridge
- Random Forest importance
- Gradient Boosting importance

The number of selected descriptors (`k`) is systematically varied to evaluate
model performance and feature-selection stability.

### Model-performance metrics

For every feature-selection method and value of `k`, predictive performance is
evaluated using:

- R²
- MAE
- RMSE

The script identifies the performance-optimal `k` using the following priority:

1. Highest mean R²
2. Lowest mean RMSE
3. Lowest mean MAE
4. Fewer descriptors as the final tie-breaker

Feature-selection stability is reported separately and is not used as an
eligibility criterion for the performance-optimal solution.

### Feature-selection stability

Stability is calculated directly from the feature subsets selected independently
in the 10 CV folds.

Eight stability metrics are calculated:

- Jaccard
- Dice
- Hamming similarity
- Kuncheva index
- Spearman correlation
- Kendall correlation
- Pearson correlation
- Nogueira stability

The script also identifies **fully stable 10/10 subsets**, where all 10 CV folds
select exactly the same descriptor subset.

A separate strict analysis selects the best-performing solution only among these
fully stable 10/10 candidates.

The script additionally verifies the Pearson and ANOVA-F seven-descriptor
solutions (`k = 7`) across all 10 folds.

---

## 2. Descriptor Correlation Analysis

**File name:** `descriptor_correlation_analysis.py`

This script evaluates inter-descriptor correlations among the seven selected
descriptors:

- 1st dopant valency
- EN_ligand_avg
- Excitation source
- Ionization Energy_sum
- avg_d_electrons
- ionic_radius_emission_center
- ionic_radius_substituted

### Analysis

The script calculates:

- Pearson correlation coefficients
- Spearman rank-correlation coefficients
- Pairwise comparison of all descriptor pairs
- Absolute Pearson correlations
- Absolute Spearman correlations
- Maximum absolute correlation for each descriptor pair

The pairwise relationships are ranked according to their maximum absolute
correlation.

### Missing values

Descriptor values are converted to numeric format and infinite values are
treated as missing values (`NaN`).

Correlation coefficients are calculated using the available valid observations
for each descriptor pair.

### Figures

Publication-quality Pearson and Spearman correlation heatmaps are generated
using a fixed correlation scale from -1 to +1.


- `Pearson_Correlation_Seven_Descriptors.csv`
- 'Spearman_Correlation_Seven_Descriptors.csv`
- 'Pearson_Spearman_Pairwise_Comparison.csv'
- 'Pearson correlation heatmap'
- 'Spearman correlation heatmap'

---

## 3. Emission-Region Error Analysis

**File name:** `emission_region_error_analysis.py`

This script evaluates model prediction errors across different spectral regions
using the final seven-descriptor feature set.

Random Forest and Gradient Boosting models are evaluated using 10-fold
out-of-fold predictions.

### Spectral regions

Predictions are grouped according to the true emission wavelength into:

- UV
- Violet/Blue
- Green
- Yellow
- Red
- Near-infrared (NIR)
- Mid-infrared (MIR)

### Analysis

For each model and spectral region, the script calculates the mean absolute
error (MAE).

Median imputation is fitted only on the training portion of each CV fold and
then applied to the corresponding validation fold.

### Main outputs

- MAE by emission region
- Complete 10-fold out-of-fold prediction/error table


---

## Reproducibility

The main feature-selection workflow uses a fixed 10-fold cross-validation
configuration:

```python
KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
