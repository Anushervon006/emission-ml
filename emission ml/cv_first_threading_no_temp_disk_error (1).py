"""
N-FEATURE SWEEP + FOLD-LEVEL FEATURE-SELECTION STABILITY
========================================================
Dataset   : IPOP phosphor database (single-dopant subset)
Target    : Emission wavelength (nm)
Methods   : 12 feature-selection methods
Regressors: Random Forest and Gradient Boosting
CV        : 10-fold; feature selection occurs inside each training fold

This version:
1. Saves the selected feature subset from every fold.
2. Calculates seven stability metrics for every Method × k.
3. Selects the optimal k separately for RF and GB using mean CV R².
4. Reports stability for the final optimal subset configuration.
5. Produces CSV files and heatmaps.

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
The feature-selection step depends on Method, k, and training fold, but not on
the downstream regressor. Therefore, for the same Method and k, the fold-level
feature subsets—and hence their stability—are identical for RF and GB.

RF and GB can nevertheless have different FINAL stability rows because each
regressor may choose a different optimal k based on predictive performance.
"""

import os
import math
import warnings
from itertools import combinations

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    ttest_ind,
    ranksums,
)
from joblib import Parallel, delayed

# Thread-based Joblib execution avoids Windows loky process pickling
# and large temporary memory-mapped files.

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    SequentialFeatureSelector,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ================================================================
# 0. CONFIGURATION — EDIT ONLY THIS SECTION
# ================================================================
DATA_PATH = r"c:\\Users\\anush\\Desktop\\ML projects\\separeted by number of dopands\\single_doped\\group_one_dopant_without_exact_duplicates.csv"
TARGET_COL = "Emission max. (nm)"

N_SPLITS = 10
FEATURE_STEP = 1
MAX_FEATURES = 30
RANDOM_SEED = 42
N_JOBS = min(4, os.cpu_count() or 1)

# Final k selection rule:
# For each feature-selection method and regressor, choose the result with:
#   1. Highest mean R²
#   2. Lowest mean RMSE
#   3. Lowest mean MAE
#   4. Fewer selected features as the final tie-breaker
#
# This is a lexicographic multi-criteria rule. R² is the primary criterion,
# while RMSE and MAE resolve cases with similar or equal R².
OPTIMAL_K_RULE = "HIGH_R2_LOW_RMSE_LOW_MAE"

OUTPUT_DIR = r"C:\\Users\\anush\Desktop\\ML projects\\separeted by number of dopands\\Test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DROP_COLS = [
    "Inorganic phosphor",
    "Host",
    "1st dopant",
    "Reference",
    "MP-ID",
    "ICSD-ID",
]


# ================================================================
# 1. REGRESSORS
# ================================================================
REGRESSORS = {
    "RF": lambda: RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "GB": lambda: GradientBoostingRegressor(
        n_estimators=100,
        random_state=RANDOM_SEED,
    ),
}


# ================================================================
# 2. LOAD DATA
# ================================================================
def load_data(path, target):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\n"
            "Change DATA_PATH in the configuration section."
        )

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=[target])
    df = df.select_dtypes(include=[np.number])

    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' was not found after numeric-column filtering."
        )

    X = df.drop(columns=[target], errors="ignore")
    y = df[target].astype(float)

    # Remove constant features.
    X = X.loc[:, X.std() > 0]

    # Median imputation.
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"Imputing NaNs in {len(nan_cols)} columns with column medians.")
        X = X.fillna(X.median())

    if X.empty:
        raise ValueError("No usable numeric features remained after preprocessing.")

    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} candidate features")
    return X, y


# ================================================================
# 3. FEATURE-SELECTION METHODS
# ================================================================
def ranked_top_k(scores, k):
    """Return indices of the top-k features; larger score is better."""
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    return np.argsort(scores, kind="stable")[-k:]


def select_features(X_train, y_train, method, k):
    n_features = X_train.shape[1]

    if not 1 <= k <= n_features:
        raise ValueError(f"k={k} is invalid for {n_features} features.")

    if method == "Pearson":
        scores = np.array(
            [
                abs(pearsonr(X_train[:, j], y_train)[0])
                if np.std(X_train[:, j]) > 0
                else 0.0
                for j in range(n_features)
            ]
        )
        return ranked_top_k(scores, k)

    if method == "MutualInfo":
        selector = SelectKBest(
            score_func=lambda X, y: mutual_info_regression(
                X,
                y,
                random_state=RANDOM_SEED,
            ),
            k=k,
        )
        selector.fit(X_train, y_train)
        return np.flatnonzero(selector.get_support())

    if method == "ANOVA_F":
        scores, _ = f_regression(X_train, y_train)
        return ranked_top_k(scores, k)

    if method == "T_Test":
        median_y = np.median(y_train)
        high_idx = np.flatnonzero(y_train >= median_y)
        low_idx = np.flatnonzero(y_train < median_y)

        scores = np.array(
            [
                abs(
                    ttest_ind(
                        X_train[high_idx, j],
                        X_train[low_idx, j],
                        equal_var=False,
                        nan_policy="omit",
                    )[0]
                )
                for j in range(n_features)
            ]
        )
        return ranked_top_k(scores, k)

    if method == "Wilcoxon":
        median_y = np.median(y_train)
        high_idx = np.flatnonzero(y_train >= median_y)
        low_idx = np.flatnonzero(y_train < median_y)

        scores = np.array(
            [
                abs(
                    ranksums(
                        X_train[high_idx, j],
                        X_train[low_idx, j],
                    )[0]
                )
                for j in range(n_features)
            ]
        )
        return ranked_top_k(scores, k)

    if method == "RFE":
        selector = RFE(
            estimator=Ridge(),
            n_features_to_select=k,
            step=1,
        )
        selector.fit(X_train, y_train)
        return np.flatnonzero(selector.support_)

    if method == "SFS":
        selector = SequentialFeatureSelector(
            estimator=Ridge(),
            n_features_to_select=k,
            direction="forward",
            cv=3,
            n_jobs=1,
        )
        selector.fit(X_train, y_train)
        return np.flatnonzero(selector.get_support())

    if method == "SBS":
        selector = SequentialFeatureSelector(
            estimator=Ridge(),
            n_features_to_select=k,
            direction="backward",
            cv=3,
            n_jobs=1,
        )
        selector.fit(X_train, y_train)
        return np.flatnonzero(selector.get_support())

    if method == "LASSO":
        model = Lasso(
            alpha=0.01,
            max_iter=5000,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        return ranked_top_k(np.abs(model.coef_), k)

    if method == "Ridge":
        model = Ridge()
        model.fit(X_train, y_train)
        return ranked_top_k(np.abs(model.coef_), k)

    if method == "RF_Importance":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        return ranked_top_k(model.feature_importances_, k)

    if method == "GB_Importance":
        model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        return ranked_top_k(model.feature_importances_, k)

    raise ValueError(f"Unknown feature-selection method: {method}")


METHODS = [
    "Pearson",
    "MutualInfo",
    "ANOVA_F",
    "T_Test",
    "Wilcoxon",
    "RFE",
    "SFS",
    "SBS",
    "LASSO",
    "Ridge",
    "RF_Importance",
    "GB_Importance",
]


# ================================================================
# 4. SINGLE-FOLD WORKER
# ================================================================
def process_fold(
    fold_number,
    train_idx,
    test_idx,
    X_arr,
    y_arr,
    method,
    k,
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_arr[train_idx])
    X_test = scaler.transform(X_arr[test_idx])

    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]

    # Feature selection is performed only on this fold's training data.
    feat_idx = np.asarray(
        select_features(X_train, y_train, method, k),
        dtype=int,
    )

    X_train_selected = X_train[:, feat_idx]
    X_test_selected = X_test[:, feat_idx]

    model_scores = {}

    for reg_name, reg_factory in REGRESSORS.items():
        model = reg_factory()
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        model_scores[reg_name] = {
            "r2": float(r2),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

    return {
        "fold": fold_number,
        "feat_idx": feat_idx,
        "scores": model_scores,
    }


# ================================================================
# 5. STABILITY METRICS
# ================================================================
STABILITY_METRIC_COLUMNS = [
    "Jaccard",
    "Dice",
    "Hamming",
    "Kuncheva",
    "Spearman",
    "Kendall",
    "Pearson_PCC",
]

# Nogueira is intentionally excluded from the full k-sweep.
# It is calculated only for the final selected subset.
FINAL_STABILITY_METRIC_COLUMNS = [
    *STABILITY_METRIC_COLUMNS,
    "Nogueira",
]


def subsets_to_binary_matrix(subsets, n_total_features):
    matrix = np.zeros(
        (len(subsets), n_total_features),
        dtype=float,
    )

    for row_idx, subset in enumerate(subsets):
        matrix[row_idx, np.asarray(subset, dtype=int)] = 1.0

    return matrix


def compute_nogueira_stability(subsets, n_total_features):
    """
    Compute Nogueira stability from the final fold-level subsets only.

    The finite-sample correction M/(M-1) is used, where M is the
    number of repeated feature-selection runs (10 CV folds here).
    """
    n_runs = len(subsets)

    if n_runs < 2:
        raise ValueError(
            "At least two feature subsets are required for Nogueira."
        )

    binary = subsets_to_binary_matrix(
        subsets,
        n_total_features,
    )

    selection_probability = binary.mean(axis=0)
    mean_selected = binary.sum(axis=1).mean()

    observed_variance = (
        n_runs / (n_runs - 1)
    ) * np.mean(
        selection_probability
        * (1.0 - selection_probability)
    )

    expected_variance = (
        (mean_selected / n_total_features)
        * (1.0 - mean_selected / n_total_features)
    )

    if abs(expected_variance) < 1e-12:
        return 1.0

    score = 1.0 - observed_variance / expected_variance
    return float(score)


def safe_binary_correlation(x, y, function):
    """
    Correlation between two binary selection vectors.

    Identical vectors are assigned 1.0. If one vector is constant and the
    vectors are not identical, correlation is undefined and is represented
    as 0.0.
    """
    if np.array_equal(x, y):
        return 1.0

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    value, _ = function(x, y)
    return 0.0 if not np.isfinite(value) else float(value)


def compute_stability_metrics(subsets, n_total_features):
    """
    Compute stability from repeated fold-level feature subsets.

    Jaccard, Dice, Hamming similarity, generalized Kuncheva, Pearson,
    Spearman, and Kendall are averaged over all fold pairs.
    """
    n_runs = len(subsets)

    if n_runs < 2:
        raise ValueError("At least two feature subsets are required.")

    if n_total_features < 1:
        raise ValueError("The total feature count must be positive.")

    binary = subsets_to_binary_matrix(subsets, n_total_features)
    pair_indices = list(combinations(range(n_runs), 2))

    values = {
        "Jaccard": [],
        "Dice": [],
        "Hamming": [],
        "Kuncheva": [],
        "Spearman": [],
        "Kendall": [],
        "Pearson_PCC": [],
    }

    for i, j in pair_indices:
        set_i = set(np.flatnonzero(binary[i]))
        set_j = set(np.flatnonzero(binary[j]))

        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        k_i = len(set_i)
        k_j = len(set_j)

        values["Jaccard"].append(
            intersection / union if union else 1.0
        )

        values["Dice"].append(
            2.0 * intersection / (k_i + k_j)
            if (k_i + k_j)
            else 1.0
        )

        # Report similarity: 1 means identical, 0 means all positions differ.
        values["Hamming"].append(
            1.0 - np.mean(np.abs(binary[i] - binary[j]))
        )

        # Generalized Kuncheva index. For this sweep, k_i = k_j,
        # but this form also remains valid if sizes differ.
        expected_overlap = (k_i * k_j) / n_total_features
        maximum_overlap = min(k_i, k_j)
        denominator = maximum_overlap - expected_overlap

        if abs(denominator) < 1e-12:
            kuncheva = 1.0 if set_i == set_j else 0.0
        else:
            kuncheva = (
                intersection - expected_overlap
            ) / denominator

        values["Kuncheva"].append(float(kuncheva))

        values["Pearson_PCC"].append(
            safe_binary_correlation(
                binary[i],
                binary[j],
                pearsonr,
            )
        )
        values["Spearman"].append(
            safe_binary_correlation(
                binary[i],
                binary[j],
                spearmanr,
            )
        )
        values["Kendall"].append(
            safe_binary_correlation(
                binary[i],
                binary[j],
                kendalltau,
            )
        )

    result = {
        metric: float(np.mean(metric_values))
        for metric, metric_values in values.items()
    }
    result["N_subsets"] = int(n_runs)
    result["N_pairwise_comparisons"] = int(len(pair_indices))
    result["Mean_k"] = float(
        np.mean([len(subset) for subset in subsets])
    )

    return result


# ================================================================
# 6. SWEEP LOOP
# ================================================================
def run_sweep(X, y):
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)

    n_total_features = X_arr.shape[1]
    feature_names = X.columns.tolist()

    max_k = min(MAX_FEATURES, n_total_features)
    k_values = list(range(1, max_k + 1, FEATURE_STEP))

    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    splits = list(splitter.split(X_arr))

    performance_records = []
    fold_subset_records = []
    frequency_records = []

    total_jobs = len(METHODS) * len(k_values)
    completed_jobs = 0

    for method in METHODS:
        print(f"\n[{method}] sweeping k = 1 to {max_k}")

        for k in k_values:
            fold_results = Parallel(n_jobs=N_JOBS, prefer="threads", require="sharedmem")(
                delayed(process_fold)(
                    fold_number,
                    train_idx,
                    test_idx,
                    X_arr,
                    y_arr,
                    method,
                    k,
                )
                for fold_number, (train_idx, test_idx)
                in enumerate(splits, start=1)
            )

            # ----------------------------------------------------
            # 6A. Save every fold-level feature subset
            # ----------------------------------------------------
            fold_subsets = []

            for fold_result in fold_results:
                feat_idx = fold_result["feat_idx"]
                fold_subsets.append(feat_idx)

                selected_names = [
                    feature_names[index]
                    for index in feat_idx
                ]

                fold_subset_records.append(
                    {
                        "FS_Method": method,
                        "N_features": k,
                        "Fold": fold_result["fold"],
                        "Feature_indices": "; ".join(
                            map(str, feat_idx.tolist())
                        ),
                        "Selected_features": "; ".join(selected_names),
                    }
                )

            # ----------------------------------------------------
            # 6B. Feature-selection frequency for this Method × k
            # ----------------------------------------------------
            frequency_counter = np.zeros(
                n_total_features,
                dtype=int,
            )

            for subset in fold_subsets:
                frequency_counter[subset] += 1

            ranked_indices = np.argsort(
                frequency_counter,
                kind="stable",
            )[::-1]

            selected_union = [
                index
                for index in ranked_indices
                if frequency_counter[index] > 0
            ]

            frequency_records.append(
                {
                    "FS_Method": method,
                    "N_features": k,
                    "All_features": "; ".join(
                        [
                            f"{feature_names[index]} "
                            f"({frequency_counter[index]}/{N_SPLITS})"
                            for index in selected_union
                        ]
                    ),
                    "N_all_features": len(selected_union),
                }
            )

            # ----------------------------------------------------
            # 6C. Predictive performance for RF and GB
            # ----------------------------------------------------
            for reg_name in REGRESSORS:
                r2_values = [
                    result["scores"][reg_name]["r2"]
                    for result in fold_results
                ]
                mae_values = [
                    result["scores"][reg_name]["mae"]
                    for result in fold_results
                ]
                rmse_values = [
                    result["scores"][reg_name]["rmse"]
                    for result in fold_results
                ]

                performance_records.append(
                    {
                        "FS_Method": method,
                        "Regressor": reg_name,
                        "N_features": k,
                        "R2_mean": np.mean(r2_values),
                        "R2_std": np.std(r2_values, ddof=1),
                        "MAE_mean": np.mean(mae_values),
                        "MAE_std": np.std(mae_values, ddof=1),
                        "RMSE_mean": np.mean(rmse_values),
                        "RMSE_std": np.std(rmse_values, ddof=1),
                    }
                )

            completed_jobs += 1
            rf_mean = np.mean(
                [
                    result["scores"]["RF"]["r2"]
                    for result in fold_results
                ]
            )
            gb_mean = np.mean(
                [
                    result["scores"]["GB"]["r2"]
                    for result in fold_results
                ]
            )

            print(
                f"  k={k:3d} | "
                f"RF R²={rf_mean:.4f} | "
                f"GB R²={gb_mean:.4f} | "
                f"[{completed_jobs}/{total_jobs}]"
            )

    performance_df = pd.DataFrame(performance_records)
    fold_subsets_df = pd.DataFrame(fold_subset_records)
    frequency_df = pd.DataFrame(frequency_records)
    combined_df = (
        performance_df
        .merge(
            frequency_df,
            on=["FS_Method", "N_features"],
            how="left",
        )
    )

    # Save complete results.
    performance_df.to_csv(
        os.path.join(OUTPUT_DIR, "PERFORMANCE_ALL_K.csv"),
        index=False,
    )
    fold_subsets_df.to_csv(
        os.path.join(OUTPUT_DIR, "FOLD_FEATURE_SUBSETS.csv"),
        index=False,
    )
    frequency_df.to_csv(
        os.path.join(OUTPUT_DIR, "FEATURE_FREQUENCY_ALL_K.csv"),
        index=False,
    )
    combined_df.to_csv(
        os.path.join(OUTPUT_DIR, "RESULTS_ALL_K.csv"),
        index=False,
    )

    # Also retain one CSV per feature-selection method.
    for method in METHODS:
        method_results = combined_df[
            combined_df["FS_Method"] == method
        ]
        method_results.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"RESULT_{method}.csv",
            ),
            index=False,
        )

    return (
        performance_df,
        fold_subsets_df,
        frequency_df,
        combined_df,
    )


# ================================================================
# 7. CHOOSE THE FINAL OPTIMAL k FOR EACH METHOD × REGRESSOR
# ================================================================
def identify_fully_stable_candidates(fold_subsets_df):
    """
    Identify Method × k candidates for which all ten CV folds selected
    exactly the same feature subset.

    This is equivalent to saying that every feature in the selected
    k-feature subset has selection frequency 10/10.
    """
    records = []

    for (method, k), group in fold_subsets_df.groupby(
        ["FS_Method", "N_features"],
        sort=False,
    ):
        group = group.sort_values("Fold").copy()

        if len(group) != N_SPLITS:
            records.append(
                {
                    "FS_Method": method,
                    "N_features": int(k),
                    "Fully_stable_10_of_10": False,
                    "Stable_feature_set": "",
                    "Number_of_distinct_fold_sets": group[
                        "Feature_indices"
                    ].nunique(),
                    "Reason": (
                        f"Expected {N_SPLITS} folds but found {len(group)}"
                    ),
                }
            )
            continue

        # Sort the indices within each fold so equality does not depend
        # on the order returned by the feature-selection method.
        canonical_sets = []

        for value in group["Feature_indices"]:
            indices = sorted(
                int(item.strip())
                for item in str(value).split(";")
                if item.strip()
            )
            canonical_sets.append(tuple(indices))

        number_of_distinct_sets = len(set(canonical_sets))
        fully_stable = number_of_distinct_sets == 1

        if fully_stable:
            stable_features = group.iloc[0]["Selected_features"]
            reason = "All 10 folds selected exactly the same subset"
        else:
            stable_features = ""
            reason = (
                f"The 10 folds produced {number_of_distinct_sets} "
                "different subsets"
            )

        records.append(
            {
                "FS_Method": method,
                "N_features": int(k),
                "Fully_stable_10_of_10": fully_stable,
                "Stable_feature_set": stable_features,
                "Number_of_distinct_fold_sets": number_of_distinct_sets,
                "Reason": reason,
            }
        )

    result = pd.DataFrame(records)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "FULLY_STABLE_10_OF_10_CANDIDATES.csv",
        ),
        index=False,
    )

    return result


def select_optimal_k(combined_df, fold_subsets_df):
    """
    Strict final-subset rule
    ------------------------
    1. Keep only Method × k candidates for which all 10 folds selected
       exactly the same feature subset.
    2. For each Method × Regressor, choose one candidate by:
         a. highest mean R²,
         b. lowest mean RMSE,
         c. lowest mean MAE,
         d. fewer features as the final tie-breaker.
    3. If no fully stable candidate exists, report that Method × Regressor
       as unavailable rather than silently using a 9/10 or lower subset.
    """
    eligibility_df = identify_fully_stable_candidates(
        fold_subsets_df
    )

    candidates = combined_df.merge(
        eligibility_df,
        on=["FS_Method", "N_features"],
        how="left",
        validate="many_to_one",
    )

    eligible = candidates[
        candidates["Fully_stable_10_of_10"] == True
    ].copy()

    final_rows = []
    missing_rows = []
    ranked_rows = []

    for method in METHODS:
        for regressor in REGRESSORS:
            group = eligible[
                (eligible["FS_Method"] == method)
                & (eligible["Regressor"] == regressor)
            ].copy()

            if group.empty:
                missing_rows.append(
                    {
                        "FS_Method": method,
                        "Regressor": regressor,
                        "Status": "No fully stable 10/10 subset found",
                    }
                )
                continue

            # All candidates are already fully stable. Performance decides.
            group = group.sort_values(
                [
                    "R2_mean",
                    "RMSE_mean",
                    "MAE_mean",
                    "N_features",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                ],
            ).reset_index(drop=True)

            group["Performance_rank_among_10_of_10_sets"] = np.arange(
                1,
                len(group) + 1,
            )

            selected = group.iloc[0].copy()
            selected["Selection_rule"] = (
                "Eligible only when all 10 folds selected the identical "
                "feature subset; then maximize R2, minimize RMSE, "
                "minimize MAE, and prefer fewer features on ties"
            )

            final_rows.append(selected.to_dict())
            ranked_rows.append(group)

    if ranked_rows:
        ranked_df = pd.concat(
            ranked_rows,
            ignore_index=True,
        )
    else:
        ranked_df = pd.DataFrame()

    ranked_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "RANKED_FULLY_STABLE_CANDIDATES.csv",
        ),
        index=False,
    )

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "METHOD_MODEL_WITHOUT_10_OF_10_SET.csv",
        ),
        index=False,
    )

    if not final_rows:
        raise RuntimeError(
            "No fully stable 10/10 subset was found for any "
            "feature-selection method."
        )

    optimal_df = pd.DataFrame(final_rows).sort_values(
        ["Regressor", "FS_Method"]
    ).reset_index(drop=True)

    optimal_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "FINAL_10_OF_10_SELECTED_SUBSETS.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------------
    # Calculate all eight stability metrics only after the final
    # fully stable subset has been chosen.
    # No stability metric is calculated during the initial k-sweep.
    # ------------------------------------------------------------
    n_total_features = int(
        fold_subsets_df["Feature_indices"]
        .str.split(";")
        .explode()
        .str.strip()
        .astype(int)
        .max()
        + 1
    )

    final_metric_records = []

    for _, selected_row in optimal_df.iterrows():
        method = selected_row["FS_Method"]
        k = int(selected_row["N_features"])

        selected_folds = fold_subsets_df[
            (fold_subsets_df["FS_Method"] == method)
            & (fold_subsets_df["N_features"] == k)
        ].sort_values("Fold")

        fold_subsets = []

        for value in selected_folds["Feature_indices"]:
            indices = np.array(
                [
                    int(item.strip())
                    for item in str(value).split(";")
                    if item.strip()
                ],
                dtype=int,
            )
            fold_subsets.append(indices)

        metrics = compute_stability_metrics(
            subsets=fold_subsets,
            n_total_features=n_total_features,
        )
        metrics["Nogueira"] = compute_nogueira_stability(
            subsets=fold_subsets,
            n_total_features=n_total_features,
        )

        final_metric_records.append(metrics)

    final_metrics_df = pd.DataFrame(final_metric_records)

    for metric in FINAL_STABILITY_METRIC_COLUMNS:
        optimal_df[metric] = final_metrics_df[metric].to_numpy()

    final_columns = [
        "FS_Method",
        "Regressor",
        "N_features",
        "Stable_feature_set",
        "R2_mean",
        "R2_std",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "Fully_stable_10_of_10",
        "Number_of_distinct_fold_sets",
        "Selection_rule",
        *FINAL_STABILITY_METRIC_COLUMNS,
    ]

    optimal_stability_df = optimal_df[
        final_columns
    ].copy()

    optimal_stability_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "FINAL_10_OF_10_STABILITY_METRICS.csv",
        ),
        index=False,
    )

    return optimal_df, optimal_stability_df


# ================================================================
# 8. PLOTS
# ================================================================
FS_COLORS = {
    "Pearson": "#2196F3",
    "MutualInfo": "#FF9800",
    "ANOVA_F": "#4CAF50",
    "T_Test": "#009688",
    "Wilcoxon": "#00ACC1",
    "RFE": "#E91E63",
    "SFS": "#9C27B0",
    "SBS": "#673AB7",
    "LASSO": "#795548",
    "Ridge": "#FF5722",
    "RF_Importance": "#00BCD4",
    "GB_Importance": "#F44336",
}

FS_LABELS = {
    "Pearson": "Filter: Pearson",
    "MutualInfo": "Filter: MI",
    "ANOVA_F": "Filter: ANOVA F",
    "T_Test": "Filter: T-Test",
    "Wilcoxon": "Filter: Wilcoxon",
    "RFE": "Wrapper: RFE",
    "SFS": "Wrapper: SFS",
    "SBS": "Wrapper: SBS",
    "LASSO": "Embedded: Lasso",
    "Ridge": "Embedded: Ridge",
    "RF_Importance": "Embedded: RF",
    "GB_Importance": "Embedded: GB",
}


def plot_performance_sweep(performance_df):
    metrics = [
        ("R2_mean", "R2_std", "R² score", True),
        ("MAE_mean", "MAE_std", "MAE (nm)", False),
        ("RMSE_mean", "RMSE_std", "RMSE (nm)", False),
    ]

    for regressor in REGRESSORS:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(13, 14),
            sharex=True,
        )

        reg_df = performance_df[
            performance_df["Regressor"] == regressor
        ]

        for ax, (
            mean_column,
            std_column,
            y_label,
            higher_is_better,
        ) in zip(axes, metrics):

            for method in METHODS:
                method_df = reg_df[
                    reg_df["FS_Method"] == method
                ].sort_values("N_features")

                if method_df.empty:
                    continue

                x = method_df["N_features"].to_numpy()
                y = method_df[mean_column].to_numpy()
                error = method_df[std_column].to_numpy()

                ax.plot(
                    x,
                    y,
                    color=FS_COLORS[method],
                    label=FS_LABELS[method],
                    linewidth=2,
                    marker="o",
                    markersize=3,
                )
                ax.fill_between(
                    x,
                    y - error,
                    y + error,
                    color=FS_COLORS[method],
                    alpha=0.10,
                )

            ax.set_ylabel(y_label)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.annotate(
                "↑ better" if higher_is_better else "↓ better",
                xy=(0.98, 0.05),
                xycoords="axes fraction",
                ha="right",
                fontsize=9,
                color="gray",
            )

        axes[-1].set_xlabel("Number of selected features")
        axes[0].set_title(
            f"Performance versus selected-feature count — {regressor}\n"
            f"{N_SPLITS}-fold cross-validation",
            fontweight="bold",
        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"PERFORMANCE_SWEEP_{regressor}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_stability_heatmap_for_each_method(stability_df):
    """
    One heatmap for each feature-selection method.
    Rows = k values; columns = seven stability metrics.
    """
    for method in METHODS:
        method_df = stability_df[
            stability_df["FS_Method"] == method
        ].sort_values("N_features")

        if method_df.empty:
            continue

        matrix = method_df.set_index(
            "N_features"
        )[STABILITY_METRIC_COLUMNS]

        plt.figure(
            figsize=(
                12,
                max(5, 0.38 * len(matrix)),
            )
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            center=0,
            linewidths=0.25,
            cbar_kws={"label": "Stability score"},
        )
        plt.title(
            f"{method}: stability metrics for every selected-feature count"
        )
        plt.xlabel("Stability metric")
        plt.ylabel("Number of selected features (k)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"STABILITY_HEATMAP_ALL_K_{method}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_optimal_stability_heatmaps(optimal_stability_df):
    """
    Separate final heatmap for RF and GB.
    Each method uses the optimal k for that regressor.
    All eight stability metrics are calculated only at this final stage.
    """
    for regressor in REGRESSORS:
        subset = optimal_stability_df[
            optimal_stability_df["Regressor"] == regressor
        ].copy()

        if subset.empty:
            continue

        row_labels = (
            subset["FS_Method"].astype(str)
            + " (k="
            + subset["N_features"].astype(int).astype(str)
            + ")"
        )

        matrix = subset[
            FINAL_STABILITY_METRIC_COLUMNS
        ].copy()
        matrix.index = row_labels

        plt.figure(
            figsize=(
                12,
                max(6, 0.62 * len(matrix)),
            )
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            center=0,
            linewidths=0.4,
            cbar_kws={"label": "Stability score"},
        )
        plt.title(
            f"{regressor}: stability of each method at its optimal k"
        )
        plt.xlabel("Stability metric")
        plt.ylabel("Feature-selection method and optimal k")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"OPTIMAL_K_STABILITY_HEATMAP_{regressor}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_stability_curves(stability_df):
    """
    One line plot per stability metric.
    """
    for metric in STABILITY_METRIC_COLUMNS:
        plt.figure(figsize=(13, 7))

        for method in METHODS:
            subset = stability_df[
                stability_df["FS_Method"] == method
            ].sort_values("N_features")

            if subset.empty:
                continue

            plt.plot(
                subset["N_features"],
                subset[metric],
                label=FS_LABELS[method],
                color=FS_COLORS[method],
                linewidth=2,
                marker="o",
                markersize=3,
            )

        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Number of selected features (k)")
        plt.ylabel(metric)
        plt.title(f"{metric} stability versus selected-feature count")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"STABILITY_CURVE_{metric}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# ================================================================
# 9. ENTRY POINT
# ================================================================
def cleanup_joblib_temp_files():
    """Remove abandoned Joblib temporary folders when possible."""
    import shutil
    import tempfile

    temp_root = Path(tempfile.gettempdir())

    for pattern in (
        "joblib_memmapping_folder_*",
        "joblib-*",
        "loky-*",
    ):
        for item in temp_root.glob(pattern):
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    cleanup_joblib_temp_files()
    X, y = load_data(DATA_PATH, TARGET_COL)

    (
        performance_df,
        fold_subsets_df,
        frequency_df,
        combined_df,
    ) = run_sweep(X, y)

    optimal_df, optimal_stability_df = select_optimal_k(
        combined_df=combined_df,
        fold_subsets_df=fold_subsets_df,
    )

    plot_performance_sweep(performance_df)
    # Final publication-style heatmaps only: 12 methods × 8 metrics.
    plot_optimal_stability_heatmaps(optimal_stability_df)

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nMain output files:")
    print("  1. PERFORMANCE_ALL_K.csv")
    print("  2. FOLD_FEATURE_SUBSETS.csv")
    print("  3. FULLY_STABLE_10_OF_10_CANDIDATES.csv")
    print("  4. RANKED_FULLY_STABLE_CANDIDATES.csv")
    print("  5. FINAL_10_OF_10_SELECTED_SUBSETS.csv")
    print("  6. FINAL_10_OF_10_STABILITY_METRICS.csv")
    print("  7. METHOD_MODEL_WITHOUT_10_OF_10_SET.csv")
    print("  8. OPTIMAL_K_STABILITY_HEATMAP_RF.png")
    print("  9. OPTIMAL_K_STABILITY_HEATMAP_GB.png")
    print("\nFinal fully stable 10/10 subset summary:")
    print(
        optimal_stability_df[
            [
                "FS_Method",
                "Regressor",
                "N_features",
                "R2_mean",
                "RMSE_mean",
                *FINAL_STABILITY_METRIC_COLUMNS,
            ]
        ].round(4).to_string(index=False)
    )

    cleanup_joblib_temp_files()
