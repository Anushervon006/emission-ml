import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind, ranksums
from joblib import Parallel, delayed

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



DATA_PATH = r"C:\\Users\\anush\\Desktop\\ML projects\\separeted by number of dopands\\single_doped\\group_one_dopant_without_exact_duplicates.csv"
TARGET_COL = "Emission max. (nm)"

OUTPUT_DIR = r"C:\\Users\\anush\Desktop\\ML projects\\separeted by number of dopands\\Final_Results_COMBINED"

N_SPLITS = 10
FEATURE_STEP = 1
MAX_FEATURES = 30
RANDOM_SEED = 42
N_JOBS = min(4, os.cpu_count() or 1)



DROP_COLS = [
    "Inorganic phosphor",
    "Host",
    "1st dopant",
    "Reference",
    "MP-ID",
    "ICSD-ID",
]

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

STABILITY_METRICS = [
    "Jaccard",
    "Dice",
    "Hamming",
    "Kuncheva",
    "Spearman",
    "Kendall",
    "Pearson_PCC",
    "Nogueira",
]

FS_LABELS = {
    "Pearson": "Pearson",
    "MutualInfo": "Mutual information",
    "ANOVA_F": "ANOVA-F",
    "T_Test": "t-test",
    "Wilcoxon": "Wilcoxon",
    "RFE": "RFE",
    "SFS": "SFS",
    "SBS": "SBS",
    "LASSO": "LASSO",
    "Ridge": "Ridge",
    "RF_Importance": "RF importance",
    "GB_Importance": "GB importance",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)



#  DATA
def load_data(path, target):
     """Load the dataset and prepare the features for machine learning."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\nChange DATA_PATH in the configuration section."
        )

    raw_df = pd.read_csv(path)
    original_columns = raw_df.columns.tolist()
    original_n_columns = raw_df.shape[1]

    # Remove rows lacking the target first.
    df = raw_df.dropna(subset=[target]).copy()

    # Explicit metadata/identifier exclusions.
    explicit_dropped = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=explicit_dropped)

    # Identify non-numeric columns before numeric filtering.
    non_numeric = [
        c for c in df.columns
        if c != target and not pd.api.types.is_numeric_dtype(df[c])
    ]

    df = df.select_dtypes(include=[np.number])

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' was not found after filtering.")

    X = df.drop(columns=[target], errors="ignore")
    y = df[target].astype(float)

    # Remove constant features.
    constant_cols = X.columns[X.std() <= 0].tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    # Median imputation.
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"Imputing NaNs in {len(nan_cols)} columns with column medians.")
        X = X.fillna(X.median())

    if X.empty:
        raise ValueError("No usable numeric features remained.")

    # Audit all exclusions relative to original CSV.
    used_features = X.columns.tolist()
    excluded_from_X = [c for c in original_columns if c not in used_features]

    print("\n" + "=" * 80)
    print("DATA PREPROCESSING AUDIT")
    print("=" * 80)
    print(f"Original CSV shape               : {raw_df.shape[0]} rows × {original_n_columns} columns")
    print(f"Rows after dropping missing target: {len(df)}")
    print(f"Target column                    : {target}")
    print(f"Final ML candidate features      : {X.shape[1]}")
    print(f"Columns not used as X            : {len(excluded_from_X)}")

    if explicit_dropped:
        print("\nExplicit metadata/ID columns removed:")
        for c in explicit_dropped:
            print(f"  - {c}")

    if non_numeric:
        print("\nAdditional non-numeric columns removed:")
        for c in non_numeric:
            print(f"  - {c}")

    print("\nTarget excluded from X:")
    print(f"  - {target}")

    if constant_cols:
        print("\nConstant numeric features removed:")
        for c in constant_cols:
            print(f"  - {c}")

    print("\nCandidate features used by feature selection:")
    for i, c in enumerate(used_features, start=1):
        print(f"  {i:2d}. {c}")

    audit_rows = []
    for c in original_columns:
        if c in used_features:
            status = "USED_AS_FEATURE"
        elif c == target:
            status = "TARGET"
        elif c in explicit_dropped:
            status = "DROPPED_METADATA_OR_ID"
        elif c in non_numeric:
            status = "DROPPED_NON_NUMERIC"
        elif c in constant_cols:
            status = "DROPPED_CONSTANT"
        else:
            status = "NOT_USED_OTHER"
        audit_rows.append({"Column": c, "Status": status})

    pd.DataFrame(audit_rows).to_csv(
        os.path.join(OUTPUT_DIR, "DATA_PREPROCESSING_AUDIT.csv"),
        index=False,
    )

    print(f"\nLoaded: {X.shape[0]} samples, {X.shape[1]} candidate features")
    return X, y



#  FEATURE SELECTION

def ranked_top_k(scores, k):
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    # Stable descending order; return feature indices in ascending index order
    # so serialization is deterministic. Membership is unaffected.
    order = np.argsort(-scores, kind="stable")[:k]
    return np.sort(order)


def select_features(X_train, y_train, method, k):
    n_features = X_train.shape[1]
    if not 1 <= k <= n_features:
        raise ValueError(f"k={k} is invalid for {n_features} features.")

    if method == "Pearson":
        scores = np.array([
            abs(pearsonr(X_train[:, j], y_train)[0])
            if np.std(X_train[:, j]) > 0 else 0.0
            for j in range(n_features)
        ])
        return ranked_top_k(scores, k)

    if method == "MutualInfo":
        selector = SelectKBest(
            score_func=lambda X, y: mutual_info_regression(
                X, y, random_state=RANDOM_SEED
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
        scores = np.array([
            abs(ttest_ind(
                X_train[high_idx, j], X_train[low_idx, j],
                equal_var=False, nan_policy="omit"
            )[0])
            for j in range(n_features)
        ])
        return ranked_top_k(scores, k)

    if method == "Wilcoxon":
        median_y = np.median(y_train)
        high_idx = np.flatnonzero(y_train >= median_y)
        low_idx = np.flatnonzero(y_train < median_y)
        scores = np.array([
            abs(ranksums(X_train[high_idx, j], X_train[low_idx, j])[0])
            for j in range(n_features)
        ])
        return ranked_top_k(scores, k)

    if method == "RFE":
        selector = RFE(Ridge(), n_features_to_select=k, step=1)
        selector.fit(X_train, y_train)
        return np.flatnonzero(selector.support_)

    if method == "SFS":
        selector = SequentialFeatureSelector(
            Ridge(), n_features_to_select=k, direction="forward", cv=3, n_jobs=1
        )
        selector.fit(X_train, y_train)
        return np.flatnonzero(selector.get_support())

    if method == "SBS":
        selector = SequentialFeatureSelector(
            Ridge(), n_features_to_select=k, direction="backward", cv=3, n_jobs=1
        )
        selector.fit(X_train, y_train)
        return np.flatnonzero(selector.get_support())

    if method == "LASSO":
        model = Lasso(alpha=0.01, max_iter=5000, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        return ranked_top_k(np.abs(model.coef_), k)

    if method == "Ridge":
        model = Ridge()
        model.fit(X_train, y_train)
        return ranked_top_k(np.abs(model.coef_), k)

    if method == "RF_Importance":
        model = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=1
        )
        model.fit(X_train, y_train)
        return ranked_top_k(model.feature_importances_, k)

    if method == "GB_Importance":
        model = GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_SEED
        )
        model.fit(X_train, y_train)
        return ranked_top_k(model.feature_importances_, k)

    raise ValueError(f"Unknown method: {method}")



#  FOLD WORKER

def process_fold(fold_number, train_idx, test_idx, X_arr, y_arr, method, k):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_arr[train_idx])
    X_test = scaler.transform(X_arr[test_idx])
    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]

    # Raw fold-level subset: preserve exactly what this fold selected.
    feat_idx = np.asarray(select_features(X_train, y_train, method, k), dtype=int)

    model_scores = {}
    for reg_name, factory in REGRESSORS.items():
        model = factory()
        model.fit(X_train[:, feat_idx], y_train)
        pred = model.predict(X_test[:, feat_idx])

        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        model_scores[reg_name] = {
            "R2": float(r2),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        }

    return {
        "Fold": fold_number,
        "Feature_indices": feat_idx,
        "Scores": model_scores,
    }



# STABILITY — ALWAYS FROM THE 10 RAW FOLD SUBSETS
def subsets_to_binary_matrix(subsets, n_total_features):
    matrix = np.zeros((len(subsets), n_total_features), dtype=float)
    for row, subset in enumerate(subsets):
        matrix[row, np.asarray(subset, dtype=int)] = 1.0
    return matrix


def safe_binary_correlation(x, y, function):
    if np.array_equal(x, y):
        return 1.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    value, _ = function(x, y)
    return 0.0 if not np.isfinite(value) else float(value)


def compute_nogueira_stability(subsets, n_total_features):
    n_runs = len(subsets)
    if n_runs < 2:
        raise ValueError("At least two subsets are required.")

    binary = subsets_to_binary_matrix(subsets, n_total_features)
    p = binary.mean(axis=0)
    mean_k = binary.sum(axis=1).mean()

    observed_variance = (n_runs / (n_runs - 1)) * np.mean(p * (1.0 - p))
    expected_variance = (mean_k / n_total_features) * (1.0 - mean_k / n_total_features)

    if abs(expected_variance) < 1e-12:
        return 1.0
    return float(1.0 - observed_variance / expected_variance)


def compute_stability_metrics(subsets, n_total_features):
    """Compute metrics directly from all pairings of the raw fold subsets."""
    if len(subsets) < 2:
        raise ValueError("At least two raw fold subsets are required.")

    binary = subsets_to_binary_matrix(subsets, n_total_features)
    pairs = list(combinations(range(len(subsets)), 2))

    values = {m: [] for m in STABILITY_METRICS if m != "Nogueira"}

    for i, j in pairs:
        set_i = set(np.flatnonzero(binary[i]))
        set_j = set(np.flatnonzero(binary[j]))
        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        k_i, k_j = len(set_i), len(set_j)

        values["Jaccard"].append(intersection / union if union else 1.0)
        values["Dice"].append(
            2.0 * intersection / (k_i + k_j) if (k_i + k_j) else 1.0
        )
        values["Hamming"].append(1.0 - np.mean(np.abs(binary[i] - binary[j])))

        expected = (k_i * k_j) / n_total_features
        maximum = min(k_i, k_j)
        denominator = maximum - expected
        if abs(denominator) < 1e-12:
            kuncheva = 1.0 if set_i == set_j else 0.0
        else:
            kuncheva = (intersection - expected) / denominator
        values["Kuncheva"].append(float(kuncheva))

        values["Spearman"].append(
            safe_binary_correlation(binary[i], binary[j], spearmanr)
        )
        values["Kendall"].append(
            safe_binary_correlation(binary[i], binary[j], kendalltau)
        )
        values["Pearson_PCC"].append(
            safe_binary_correlation(binary[i], binary[j], pearsonr)
        )

    result = {metric: float(np.mean(scores)) for metric, scores in values.items()}
    result["Nogueira"] = compute_nogueira_stability(subsets, n_total_features)
    result["Mean_stability"] = float(np.mean([result[m] for m in STABILITY_METRICS]))
    result["N_subsets"] = len(subsets)
    result["N_pairwise_comparisons"] = len(pairs)
    result["Mean_k"] = float(np.mean([len(s) for s in subsets]))
    result["Number_of_distinct_fold_sets"] = len({tuple(sorted(s)) for s in subsets})
    result["Fully_identical_10_of_10"] = result["Number_of_distinct_fold_sets"] == 1
    return result



# ALL-k SWEEP

def run_sweep(X, y):
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    feature_names = X.columns.tolist()
    n_total_features = X.shape[1]

    max_k = min(MAX_FEATURES, n_total_features)
    k_values = list(range(1, max_k + 1, FEATURE_STEP))

    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(X_arr))

    performance_records = []
    fold_subset_records = []
    stability_records = []
    frequency_records = []

    total = len(METHODS) * len(k_values)
    completed = 0

    for method in METHODS:
        print(f"\n[{method}] k = 1 ... {max_k}")

        for k in k_values:
            fold_results = Parallel(
                n_jobs=N_JOBS, prefer="threads", require="sharedmem"
            )(
                delayed(process_fold)(
                    fold_number, train_idx, test_idx,
                    X_arr, y_arr, method, k
                )
                for fold_number, (train_idx, test_idx)
                in enumerate(splits, start=1)
            )

            raw_subsets = [r["Feature_indices"] for r in fold_results]

            # Save the untouched fold-level subsets.
            for result in fold_results:
                idx = result["Feature_indices"]
                fold_subset_records.append({
                    "FS_Method": method,
                    "N_features": k,
                    "Fold": result["Fold"],
                    "Feature_indices": "; ".join(map(str, idx.tolist())),
                    "Selected_features": "; ".join(feature_names[i] for i in idx),
                })

            # Calculate stability NOW, using the 10 original raw subsets.
            stability = compute_stability_metrics(raw_subsets, n_total_features)
            stability_records.append({
                "FS_Method": method,
                "N_features": k,
                **stability,
            })

            # Frequencies are descriptive only; they never replace raw subsets.
            counts = np.zeros(n_total_features, dtype=int)
            for subset in raw_subsets:
                counts[subset] += 1
            present = np.flatnonzero(counts > 0)
            present = present[np.argsort(-counts[present], kind="stable")]
            frequency_records.append({
                "FS_Method": method,
                "N_features": k,
                "Feature_frequencies": "; ".join(
                    f"{feature_names[i]} ({counts[i]}/{N_SPLITS})" for i in present
                ),
                "N_features_in_union": len(present),
            })

            # RF and GB performance at this Method × k.
            for reg_name in REGRESSORS:
                r2 = [r["Scores"][reg_name]["R2"] for r in fold_results]
                mae = [r["Scores"][reg_name]["MAE"] for r in fold_results]
                rmse = [r["Scores"][reg_name]["RMSE"] for r in fold_results]
                performance_records.append({
                    "FS_Method": method,
                    "Regressor": reg_name,
                    "N_features": k,
                    "R2_mean": np.mean(r2),
                    "R2_std": np.std(r2, ddof=1),
                    "MAE_mean": np.mean(mae),
                    "MAE_std": np.std(mae, ddof=1),
                    "RMSE_mean": np.mean(rmse),
                    "RMSE_std": np.std(rmse, ddof=1),
                })

            completed += 1
            print(
                f"  k={k:2d} | RF R²={performance_records[-2]['R2_mean']:.4f} "
                f"| GB R²={performance_records[-1]['R2_mean']:.4f} "
                f"| stability={stability['Mean_stability']:.4f} "
                f"[{completed}/{total}]"
            )

    performance_df = pd.DataFrame(performance_records)
    fold_subsets_df = pd.DataFrame(fold_subset_records)
    stability_df = pd.DataFrame(stability_records)
    frequency_df = pd.DataFrame(frequency_records)

    results_all_k = (
        performance_df
        .merge(stability_df, on=["FS_Method", "N_features"], how="left")
        .merge(frequency_df, on=["FS_Method", "N_features"], how="left")
    )

    performance_df.to_csv(os.path.join(OUTPUT_DIR, "PERFORMANCE_ALL_K.csv"), index=False)
    fold_subsets_df.to_csv(os.path.join(OUTPUT_DIR, "RAW_FOLD_FEATURE_SUBSETS.csv"), index=False)
    stability_df.to_csv(os.path.join(OUTPUT_DIR, "STABILITY_ALL_K_RAW_FOLDS.csv"), index=False)
    frequency_df.to_csv(os.path.join(OUTPUT_DIR, "FEATURE_FREQUENCY_ALL_K.csv"), index=False)
    results_all_k.to_csv(os.path.join(OUTPUT_DIR, "RESULTS_ALL_K_WITH_STABILITY.csv"), index=False)

    return performance_df, fold_subsets_df, stability_df, results_all_k



# OPTIMAL k — PERFORMANCE ONLY, NO 10/10 FILTER
def select_optimal_k(performance_df, stability_df):
    rows = []

    for regressor in REGRESSORS:
        for method in METHODS:
            group = performance_df[
                (performance_df["Regressor"] == regressor)
                & (performance_df["FS_Method"] == method)
            ].copy()

            if group.empty:
                continue

            group = group.sort_values(
                ["R2_mean", "RMSE_mean", "MAE_mean", "N_features"],
                ascending=[False, True, True, True],
            )
            selected = group.iloc[0].to_dict()

            stability_row = stability_df[
                (stability_df["FS_Method"] == method)
                & (stability_df["N_features"] == int(selected["N_features"]))
            ].iloc[0]

            for column in [
                *STABILITY_METRICS,
                "Mean_stability",
                "Number_of_distinct_fold_sets",
                "Fully_identical_10_of_10",
            ]:
                selected[column] = stability_row[column]

            selected["Selection_rule"] = (
                "Highest mean R2, then lowest RMSE, lowest MAE, fewer k; "
                "no stability or 10/10 eligibility filter"
            )
            rows.append(selected)

    summary_df = pd.DataFrame(rows).sort_values(
        ["Regressor", "FS_Method"]
    ).reset_index(drop=True)

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "OPTIMAL_K_SUMMARY_RF_GB.csv"), index=False
    )

    for regressor in REGRESSORS:
        summary_df[summary_df["Regressor"] == regressor].to_csv(
            os.path.join(OUTPUT_DIR, f"OPTIMAL_K_SUMMARY_{regressor}.csv"),
            index=False,
        )

    return summary_df



#  STRICT FULLY-STABLE 10/10 FINAL SELECTION

def select_best_fully_stable_k(performance_df, stability_df, fold_subsets_df):
    """
    Code-1 branch:
    1) Keep only Method × k configurations for which all 10 folds selected
       exactly the same feature subset.
    2) For each Method × Regressor, select the best eligible row using:
       highest R² -> lowest RMSE -> lowest MAE -> fewer k.
    3) Report methods without any fully stable candidate instead of silently
       substituting a less-stable solution.
    """
    stable_flags = stability_df[
        [
            "FS_Method",
            "N_features",
            "Number_of_distinct_fold_sets",
            "Fully_identical_10_of_10",
            *STABILITY_METRICS,
            "Mean_stability",
        ]
    ].copy()

    candidates = performance_df.merge(
        stable_flags,
        on=["FS_Method", "N_features"],
        how="left",
        validate="many_to_one",
    )

    eligible = candidates[
        candidates["Fully_identical_10_of_10"] == True
    ].copy()

    final_rows = []
    missing_rows = []
    ranked_rows = []

    for regressor in REGRESSORS:
        for method in METHODS:
            group = eligible[
                (eligible["Regressor"] == regressor)
                & (eligible["FS_Method"] == method)
            ].copy()

            if group.empty:
                missing_rows.append({
                    "FS_Method": method,
                    "Regressor": regressor,
                    "Status": "No fully stable 10/10 subset found",
                })
                continue

            group = group.sort_values(
                ["R2_mean", "RMSE_mean", "MAE_mean", "N_features"],
                ascending=[False, True, True, True],
            ).reset_index(drop=True)

            group["Performance_rank_among_10_of_10_sets"] = np.arange(
                1, len(group) + 1
            )

            selected = group.iloc[0].to_dict()

            # Recover the exact stable descriptor set from one fold.
            k = int(selected["N_features"])
            fold_rows = fold_subsets_df[
                (fold_subsets_df["FS_Method"] == method)
                & (fold_subsets_df["N_features"] == k)
            ].sort_values("Fold")

            if len(fold_rows) == N_SPLITS:
                stable_feature_set = fold_rows.iloc[0]["Selected_features"]
            else:
                stable_feature_set = ""

            selected["Stable_feature_set"] = stable_feature_set
            selected["Selection_rule"] = (
                "Eligible only if all 10 folds selected the identical subset; "
                "then highest mean R2, lowest RMSE, lowest MAE, fewer k"
            )
            final_rows.append(selected)
            ranked_rows.append(group)

    ranked_df = (
        pd.concat(ranked_rows, ignore_index=True)
        if ranked_rows else pd.DataFrame()
    )
    ranked_df.to_csv(
        os.path.join(OUTPUT_DIR, "RANKED_FULLY_STABLE_10_OF_10_CANDIDATES.csv"),
        index=False,
    )

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(
        os.path.join(OUTPUT_DIR, "METHOD_MODEL_WITHOUT_10_OF_10_SET.csv"),
        index=False,
    )

    fully_stable_candidates = stable_flags[
        stable_flags["Fully_identical_10_of_10"] == True
    ].copy()
    fully_stable_candidates.to_csv(
        os.path.join(OUTPUT_DIR, "FULLY_STABLE_10_OF_10_CANDIDATES.csv"),
        index=False,
    )

    if final_rows:
        final_df = pd.DataFrame(final_rows).sort_values(
            ["Regressor", "FS_Method"]
        ).reset_index(drop=True)
    else:
        final_df = pd.DataFrame()

    final_df.to_csv(
        os.path.join(OUTPUT_DIR, "FINAL_10_OF_10_SELECTED_SUBSETS.csv"),
        index=False,
    )

    return final_df, missing_df



#  PEARSON / ANOVA-F k=7 EXACT-DESCRIPTOR VERIFICATION

def verify_pearson_anova_k7(fold_subsets_df):
    target_methods = ["Pearson", "ANOVA_F"]
    report_rows = []

    available_k = set(fold_subsets_df["N_features"].astype(int).unique())
    if 7 not in available_k:
        report = pd.DataFrame([{
            "Check": "Pearson/ANOVA-F at k=7",
            "Result": "Not available because k=7 was not included in the sweep.",
        }])
        report.to_csv(
            os.path.join(OUTPUT_DIR, "PEARSON_ANOVA_F_K7_VERIFICATION.csv"),
            index=False,
        )
        return report

    parsed = {}
    for method in target_methods:
        group = fold_subsets_df[
            (fold_subsets_df["FS_Method"] == method)
            & (fold_subsets_df["N_features"] == 7)
        ].sort_values("Fold")

        parsed[method] = {}
        for _, row in group.iterrows():
            names = tuple(sorted(
                x.strip() for x in str(row["Selected_features"]).split(";")
                if x.strip()
            ))
            parsed[method][int(row["Fold"])] = names
            report_rows.append({
                "FS_Method": method,
                "N_features": 7,
                "Fold": int(row["Fold"]),
                "Selected_features": "; ".join(names),
            })

    pearson_sets = list(parsed.get("Pearson", {}).values())
    anova_sets = list(parsed.get("ANOVA_F", {}).values())

    pearson_identical = len(pearson_sets) == N_SPLITS and len(set(pearson_sets)) == 1
    anova_identical = len(anova_sets) == N_SPLITS and len(set(anova_sets)) == 1
    methods_match_each_fold = (
        set(parsed.get("Pearson", {})) == set(parsed.get("ANOVA_F", {}))
        and all(
            parsed["Pearson"][fold] == parsed["ANOVA_F"][fold]
            for fold in parsed.get("Pearson", {})
        )
    )
    same_seven_all_ten = pearson_identical and anova_identical and methods_match_each_fold

    exact_descriptors = (
        "; ".join(pearson_sets[0]) if same_seven_all_ten and pearson_sets else ""
    )

    detail_df = pd.DataFrame(report_rows)
    detail_df.to_csv(
        os.path.join(OUTPUT_DIR, "PEARSON_ANOVA_F_K7_FOLD_DETAILS.csv"),
        index=False,
    )

    confirmation_df = pd.DataFrame([{
        "Pearson_identical_across_10_folds": pearson_identical,
        "ANOVA_F_identical_across_10_folds": anova_identical,
        "Pearson_and_ANOVA_F_match_in_each_fold": methods_match_each_fold,
        "Same_exact_seven_descriptors_in_all_10_folds": same_seven_all_ten,
        "Exact_seven_descriptors": exact_descriptors,
    }])
    confirmation_df.to_csv(
        os.path.join(OUTPUT_DIR, "PEARSON_ANOVA_F_K7_VERIFICATION.csv"),
        index=False,
    )

    with open(
        os.path.join(OUTPUT_DIR, "PEARSON_ANOVA_F_K7_REPORT.txt"),
        "w", encoding="utf-8"
    ) as f:
        f.write("PEARSON / ANOVA-F + RF, k=7 VERIFICATION\n")
        f.write("=" * 55 + "\n")
        f.write(f"Pearson identical across 10 folds: {pearson_identical}\n")
        f.write(f"ANOVA-F identical across 10 folds: {anova_identical}\n")
        f.write(f"Methods match fold-by-fold: {methods_match_each_fold}\n")
        f.write(f"Same exact seven in all 10 folds: {same_seven_all_ten}\n\n")
        if same_seven_all_ten:
            f.write("Exact seven descriptors:\n")
            for number, descriptor in enumerate(pearson_sets[0], start=1):
                f.write(f"{number}. {descriptor}\n")
        else:
            f.write(
                "No single seven-descriptor list can be truthfully reported "
                "as identical across both methods and all ten folds. See the "
                "fold-detail CSV.\n"
            )

    return confirmation_df



# FIGURES
def plot_performance_sweep(performance_df, optimal_df):
    metric_specs = [
        ("R2_mean", "R2_std", "R²", True),
        ("MAE_mean", "MAE_std", "MAE (nm)", False),
        ("RMSE_mean", "RMSE_std", "RMSE (nm)", False),
    ]

    for regressor in REGRESSORS:
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        reg_df = performance_df[performance_df["Regressor"] == regressor]
        reg_opt = optimal_df[optimal_df["Regressor"] == regressor]

        for ax, (mean_col, std_col, ylabel, higher_better) in zip(axes, metric_specs):
            for method in METHODS:
                method_df = reg_df[reg_df["FS_Method"] == method].sort_values("N_features")
                if method_df.empty:
                    continue

                x = method_df["N_features"].to_numpy()
                y = method_df[mean_col].to_numpy()
                err = method_df[std_col].to_numpy()

                line, = ax.plot(
                    x, y, marker="o", markersize=3, linewidth=1.7,
                    label=FS_LABELS[method]
                )
                ax.fill_between(x, y - err, y + err, alpha=0.08)

                opt_row = reg_opt[reg_opt["FS_Method"] == method]
                if not opt_row.empty:
                    k_opt = int(opt_row.iloc[0]["N_features"])
                    y_opt = float(opt_row.iloc[0][mean_col])
                    ax.scatter(
                        [k_opt], [y_opt], s=90, marker="*",
                        edgecolors="black", linewidths=0.7,
                        zorder=6, color=line.get_color()
                    )

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                0.99, 0.04, "higher is better" if higher_better else "lower is better",
                transform=ax.transAxes, ha="right", fontsize=9
            )

        axes[-1].set_xlabel("Number of selected features (k)")
        axes[0].set_title(
            f"{regressor}: predictive performance versus k\n"
            "Stars mark the performance-selected optimal k for each method",
            fontweight="bold",
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"PERFORMANCE_VS_K_{regressor}_OPTIMAL_MARKED.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()


def plot_stability_heatmaps(stability_df):
    for method in METHODS:
        subset = stability_df[stability_df["FS_Method"] == method].sort_values("N_features")
        if subset.empty:
            continue

        matrix = subset.set_index("N_features")[STABILITY_METRICS]
        plt.figure(figsize=(12, max(6, 0.38 * len(matrix))))
        sns.heatmap(
            matrix, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.25, cbar_kws={"label": "Stability score"}
        )
        plt.title(
            f"{FS_LABELS[method]}: raw fold-level stability versus k\n"
            "Calculated directly from the 10 original CV subsets"
        )
        plt.xlabel("Stability metric")
        plt.ylabel("Number of selected features (k)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"SUPPLEMENTARY_STABILITY_ALL_K_{method}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()


def plot_summary_table(optimal_df):
    for regressor in REGRESSORS:
        subset = optimal_df[optimal_df["Regressor"] == regressor].copy()
        subset["Method"] = subset["FS_Method"].map(FS_LABELS)
        subset["k"] = subset["N_features"].astype(int)
        subset["R²"] = subset["R2_mean"].map(lambda x: f"{x:.3f}")
        subset["MAE"] = subset["MAE_mean"].map(lambda x: f"{x:.2f}")
        subset["RMSE"] = subset["RMSE_mean"].map(lambda x: f"{x:.2f}")
        subset["Stability"] = subset["Mean_stability"].map(lambda x: f"{x:.3f}")
        subset["10/10"] = subset["Fully_identical_10_of_10"].map({True: "Yes", False: "No"})

        display = subset[["Method", "k", "R²", "MAE", "RMSE", "Stability", "10/10"]]

        fig, ax = plt.subplots(figsize=(13, 0.55 * len(display) + 2.0))
        ax.axis("off")
        table = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)
        ax.set_title(
            f"{regressor}: optimal-k performance and raw-fold stability summary",
            fontweight="bold", pad=16
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"OPTIMAL_K_COMPACT_SUMMARY_{regressor}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()



def plot_fully_stable_summary(fully_stable_df):
    """Create RF/GB heatmaps for the final strict 10/10 solutions."""
    if fully_stable_df.empty:
        return

    for regressor in REGRESSORS:
        subset = fully_stable_df[
            fully_stable_df["Regressor"] == regressor
        ].copy()

        if subset.empty:
            continue

        row_labels = (
            subset["FS_Method"].astype(str)
            + " (k="
            + subset["N_features"].astype(int).astype(str)
            + ")"
        )

        matrix = subset[STABILITY_METRICS].copy()
        matrix.index = row_labels

        plt.figure(figsize=(12, max(6, 0.62 * len(matrix))))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.4,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Stability score"},
        )
        plt.title(
            f"{regressor}: final fully stable 10/10 subsets\n"
            "Selected among only identical 10-fold feature sets"
        )
        plt.xlabel("Stability metric")
        plt.ylabel("Feature-selection method and selected k")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"FINAL_10_OF_10_STABILITY_HEATMAP_{regressor}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()



# OPTIONAL CLEANUP
def cleanup_joblib_temp_files():
    import shutil
    import tempfile

    temp_root = Path(tempfile.gettempdir())
    for pattern in ("joblib_memmapping_folder_*", "joblib-*", "loky-*"):
        for item in temp_root.glob(pattern):
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except OSError:
                pass


#  ENTRY POINT
if __name__ == "__main__":
    cleanup_joblib_temp_files()
    X, y = load_data(DATA_PATH, TARGET_COL)

    
    # ONE CV SWEEP: performance + raw-fold stability for every k
    
    performance_df, fold_subsets_df, stability_df, results_all_k = run_sweep(X, y)

    
    # BRANCH A: performance-optimal k, no stability eligibility rule
    
    performance_optimal_df = select_optimal_k(performance_df, stability_df)

    
    # BRANCH B: strict 10/10-stable candidates only
    fully_stable_optimal_df, missing_stable_df = select_best_fully_stable_k(
        performance_df=performance_df,
        stability_df=stability_df,
        fold_subsets_df=fold_subsets_df,
    )

    
    # Pearson / ANOVA-F k=7 verification
    k7_confirmation = verify_pearson_anova_k7(fold_subsets_df)

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    plot_performance_sweep(performance_df, performance_optimal_df)
    plot_stability_heatmaps(stability_df)
    plot_summary_table(performance_optimal_df)
    plot_fully_stable_summary(fully_stable_optimal_df)

    print("\n" + "=" * 88)
    print("COMBINED ANALYSIS COMPLETE")
    print("=" * 88)
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nMethodological guarantee:")
    print(
        "All eight stability metrics were calculated directly from the 10 original "
        "fold-level feature subsets for every Method × k. Frequencies never replace "
        "the raw subsets."
    )

    print("\nBRANCH A — PERFORMANCE-OPTIMAL k (NO 10/10 FILTER)")
    print(
        performance_optimal_df[[
            "FS_Method", "Regressor", "N_features",
            "R2_mean", "MAE_mean", "RMSE_mean", "Mean_stability",
            "Fully_identical_10_of_10"
        ]].round(4).to_string(index=False)
    )

    print("\nBRANCH B — BEST AMONG FULLY STABLE 10/10 SUBSETS")
    if fully_stable_optimal_df.empty:
        print("No fully stable 10/10 candidate was found.")
    else:
        print(
            fully_stable_optimal_df[[
                "FS_Method", "Regressor", "N_features",
                "R2_mean", "MAE_mean", "RMSE_mean",
                "Stable_feature_set"
            ]].round(4).to_string(index=False)
        )

    if not missing_stable_df.empty:
        print("\nMethod × model combinations without a 10/10 candidate:")
        print(missing_stable_df.to_string(index=False))

    print("\nPearson / ANOVA-F k=7 verification:")
    print(k7_confirmation.to_string(index=False))

    print("\nMain output files:")
    outputs = [
        "DATA_PREPROCESSING_AUDIT.csv",
        "PERFORMANCE_ALL_K.csv",
        "RAW_FOLD_FEATURE_SUBSETS.csv",
        "STABILITY_ALL_K_RAW_FOLDS.csv",
        "RESULTS_ALL_K_WITH_STABILITY.csv",
        "FEATURE_FREQUENCY_ALL_K.csv",
        "OPTIMAL_K_SUMMARY_RF_GB.csv",
        "OPTIMAL_K_SUMMARY_RF.csv",
        "OPTIMAL_K_SUMMARY_GB.csv",
        "FULLY_STABLE_10_OF_10_CANDIDATES.csv",
        "RANKED_FULLY_STABLE_10_OF_10_CANDIDATES.csv",
        "FINAL_10_OF_10_SELECTED_SUBSETS.csv",
        "METHOD_MODEL_WITHOUT_10_OF_10_SET.csv",
        "PEARSON_ANOVA_F_K7_VERIFICATION.csv",
        "PEARSON_ANOVA_F_K7_FOLD_DETAILS.csv",
        "PEARSON_ANOVA_F_K7_REPORT.txt",
        "PERFORMANCE_VS_K_RF_OPTIMAL_MARKED.png",
        "PERFORMANCE_VS_K_GB_OPTIMAL_MARKED.png",
        "SUPPLEMENTARY_STABILITY_ALL_K_<METHOD>.png (12 files)",
        "OPTIMAL_K_COMPACT_SUMMARY_RF.png",
        "OPTIMAL_K_COMPACT_SUMMARY_GB.png",
        "FINAL_10_OF_10_STABILITY_HEATMAP_RF.png",
        "FINAL_10_OF_10_STABILITY_HEATMAP_GB.png",
    ]
    for i, name in enumerate(outputs, start=1):
        print(f"  {i:2d}. {name}")

    cleanup_joblib_temp_files()
