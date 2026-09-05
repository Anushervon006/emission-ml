# From Explainable Machine Learning to Physics-Guided Design Rules for Inorganic Phosphors

Machine-learning analysis for predicting the emission wavelength of inorganic
phosphor materials from physicochemical descriptors, with emphasis on robust
feature selection, feature-selection stability, model interpretation, and the
identification of physically meaningful descriptor relationships.

## Main Goal

The main objective of this study is to identify a **small, stable, and
predictive set of physicochemical descriptors** for machine-learning prediction
of the emission wavelength of inorganic phosphor materials.

Rather than relying on a single feature-selection algorithm, we systematically
compare multiple feature-selection strategies to determine:

- which descriptors are repeatedly identified as informative;
- how many descriptors are required for accurate prediction;
- whether the selected descriptors remain stable when the training data change;
- whether different feature-selection methods converge toward the same
  descriptor subset;
- how the selected subsets affect Random Forest and Gradient Boosting
  performance; and
- whether the final descriptors can be interpreted in terms of the underlying
  physics and chemistry of phosphor emission.

The broader purpose of the workflow is to provide a reproducible strategy for
descriptor selection that can potentially be adapted to other regression
problems in materials science and chemistry.

---

## Path to the Final Descriptor Subset

The final descriptors are not selected in a single step. Instead, the study
follows a multi-stage procedure combining **feature selection, cross-validation,
stability analysis, and predictive-performance evaluation**.

The overall path is:

```text
Candidate physicochemical descriptors
                ↓
      12 feature-selection methods
                ↓
     Feature count k = 1, 2, ... 30
                ↓
         10-fold cross-validation
                ↓
Feature selection inside each training fold
                ↓
   10 selected subsets for each Method × k
                ↓
      Feature-selection stability analysis
                ↓
   Search for fully stable 10/10 subsets
                ↓
If multiple stable candidates remain:
compare predictive performance
                ↓
        R² → RMSE → MAE → smaller k
                ↓
       Final stable descriptor subset
                ↓
Correlation and SHAP-based interpretation
                ↓
Physics-guided interpretation of the descriptors
