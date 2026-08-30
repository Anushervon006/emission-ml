# 📊 Dataset

This directory contains the dataset and reproducibility files used in the
machine-learning analysis of inorganic phosphor emission wavelengths.

## 🔬 Main Dataset

**File:** `group_one_dopant_without_exact_duplicates.csv`

The dataset contains inorganic phosphor materials together with
physicochemical descriptors used as input variables for machine-learning
models.

**Target variable:** `Emission max. (nm)`

The dataset represents the single-dopant subset after removal of exact
duplicate entries.

## 📋 Descriptor Definitions

**File:** `descriptor_definitions.csv`

This file provides the exact names, physical definitions, and units of the
descriptors used in the machine-learning analysis.

## 🔁 Cross-Validation Assignments

**File:** `CV_FOLD_ASSIGNMENTS.csv`

This file provides the exact sample-to-fold assignments used for the
10-fold cross-validation analysis.

Cross-validation settings:

- `n_splits = 10`
- `shuffle = True`
- `random_state = 42`

## ⚙️ Data Preprocessing

Before model training:

- Rows with missing target values were removed.
- Non-numerical metadata columns were excluded from the ML feature matrix.
- Constant numerical features were removed.
- Missing descriptor values were imputed using the median.
- Feature scaling was performed within each cross-validation training fold.
