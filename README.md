# Emission-ML

Machine-learning analysis for predicting the emission wavelength of inorganic phosphor materials.

## Dataset

The dataset used in this study contains inorganic phosphor materials and their physicochemical descriptors.

**Target variable:** 'Emission max. (nm)'

## Cross-Validation

The analysis uses 10-fold cross-validation with:

- 'n_splits = 10'
- 'shuffle = True'
- 'random_state = 42'

The exact sample-to-fold assignments are provided in:

`data/reproducibility/CV_FOLD_ASSIGNMENTS.csv`

## Feature Selection

Twelve feature-selection methods were investigated:

- Pearson correlation
- Mutual Information
- ANOVA F-test
- t-test
- Wilcoxon rank-sum test
- RFE
- SFS
- SBS
- LASSO
- Ridge
- Random Forest Importance
- Gradient Boosting Importance

## Machine-Learning Models

- Random Forest
- Gradient Boosting

## Evaluation Metrics

- R²
- MAE
- RMSE
