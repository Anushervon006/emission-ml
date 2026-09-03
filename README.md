# From Explainable Machine Learning to Physics-Guided Design Rules for Inorganic Phosphors
 

Machine-learning analysis for predicting the emission wavelength of inorganic
phosphor materials from physicochemical descriptors.

## Dataset

The dataset used in this study contains inorganic phosphor materials and their
physicochemical descriptors.

The machine-learning analysis is based on the **Optical Property Database of
Inorganic Phosphor (IPOP)** dataset.

The original IPOP dataset is publicly available through Figshare:

[Optical property database of inorganic phosphor (IPOP dataset)](https://figshare.com/articles/dataset/Optical_property_database_of_inorganic_phosphor_IPOP_dataset_/21766916)

For the present study, the database was processed to obtain the single-dopant
subset used for machine-learning analysis. Exact duplicate entries were removed
during dataset preparation.

**Dataset used in this repository:**

`data/group_one_dopant_without_exact_duplicates.csv`

**Target variable:**

`Emission max. (nm)`

The remaining numerical physicochemical descriptors were used as candidate
predictor variables for feature-selection and machine-learning analysis.

Before model development:

- Rows without a valid target value were excluded.
- Metadata and identifier columns were excluded from the predictor matrix.
- Non-numeric columns were removed from the candidate feature set.
- Constant numerical descriptors were removed.
- Missing numerical descriptor values were handled during preprocessing.

The processed dataset included in this repository therefore represents the
specific data subset used for the reported machine-learning experiments.


## Cross-Validation

Model development and feature selection are performed using **10-fold
cross-validation (CV)**.

The CV configuration is:

```python
KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
