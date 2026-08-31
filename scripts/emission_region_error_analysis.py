from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


# Configuration
DATA_PATH = Path(
    r"C:\Users\anush\Desktop\ML projects\separeted by number of dopands"
    r"\single_doped\group_one_dopant_without_exact_duplicates.csv"
)
OUTPUT_DIR = Path("results_emission_region")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Emission max. (nm)"

FEATURES = [
    "ionic_radius_emission_center",
    "Excitation source (nm)",
    "1st dopant valency",
    "avg_d_electrons",
    "Ionization Energy_sum",
    "ionic_radius_substituted",
    "EN_ligand_avg",
]

RF_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}

GB_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
}

N_FOLDS = 10
RANDOM_SEED = 42

REGION_ORDER = [
    "UV",
    "Violet/Blue",
    "Green",
    "Yellow",
    "Red",
    "NIR",
    "MIR",
]


def emission_region(wavelength):
    if wavelength < 400:
        return "UV"
    if wavelength < 500:
        return "Violet/Blue"
    if wavelength < 580:
        return "Green"
    if wavelength < 620:
        return "Yellow"
    if wavelength < 750:
        return "Red"
    if wavelength < 1100:
        return "NIR"
    return "MIR"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Update DATA_PATH in the configuration section."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n- "
            + "\n- ".join(missing_columns)
        )

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df[TARGET].notna()].reset_index(drop=True)

    if len(df) < N_FOLDS:
        raise ValueError(
            f"Only {len(df)} samples are available, but N_FOLDS={N_FOLDS}."
        )

    return df


def make_models():
    return {
        "RF": RandomForestRegressor(**RF_PARAMS),
        "GB": GradientBoostingRegressor(**GB_PARAMS),
    }


def generate_oof_predictions(df):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(float)

    cv = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    splits = list(cv.split(X))

    predictions = {
        model_name: np.full(len(df), np.nan)
        for model_name in ("RF", "GB")
    }

    for model_name in ("RF", "GB"):
        print(f"\n{model_name}: {N_FOLDS}-fold OOF prediction")

        for fold_number, (train_idx, val_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

            model = make_models()[model_name]
            model.fit(X_train, y_train)

            predictions[model_name][val_idx] = model.predict(X_val)
            print(f"  Fold {fold_number}/{N_FOLDS} completed")

        if np.isnan(predictions[model_name]).any():
            raise RuntimeError(
                f"Some {model_name} out-of-fold predictions are missing."
            )

    return y, predictions


def build_error_tables(y, predictions):
    error_df = pd.DataFrame(
        {
            "y_true": y.to_numpy(),
            "y_pred_RF": predictions["RF"],
            "y_pred_GB": predictions["GB"],
        }
    )

    error_df["abs_error_RF"] = np.abs(
        error_df["y_pred_RF"] - error_df["y_true"]
    )
    error_df["abs_error_GB"] = np.abs(
        error_df["y_pred_GB"] - error_df["y_true"]
    )
    error_df["region"] = error_df["y_true"].apply(emission_region)

    rows = []

    for region in REGION_ORDER:
        subset = error_df[error_df["region"] == region]

        if subset.empty:
            continue

        rows.append(
            {
                "Region": region,
                "n": len(subset),
                "MAE_RF": mean_absolute_error(
                    subset["y_true"], subset["y_pred_RF"]
                ),
                "MAE_GB": mean_absolute_error(
                    subset["y_true"], subset["y_pred_GB"]
                ),
            }
        )

    return error_df, pd.DataFrame(rows)


def plot_mae_by_region(region_df):
    x = np.arange(len(region_df))
    bar_width = 0.34

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.bar(
        x - bar_width / 2,
        region_df["MAE_RF"],
        bar_width,
        label="RF",
        alpha=0.90,
    )
    ax.bar(
        x + bar_width / 2,
        region_df["MAE_GB"],
        bar_width,
        label="GB",
        alpha=0.90,
    )

    title_font_size = 26
    axis_font_size = 24
    tick_font_size = 20
    legend_font_size = 20
    annotation_font_size = 17

    maximum_mae = max(
        region_df["MAE_RF"].max(),
        region_df["MAE_GB"].max(),
    )

    for index, sample_count in enumerate(region_df["n"]):
        highest_bar = max(
            region_df.loc[index, "MAE_RF"],
            region_df.loc[index, "MAE_GB"],
        )
        ax.text(
            index,
            highest_bar + maximum_mae * 0.025,
            f"n={sample_count}",
            ha="center",
            va="bottom",
            fontsize=annotation_font_size,
            fontweight="bold",
        )

    display_labels = [
        "Violet/\nBlue" if label == "Violet/Blue" else label
        for label in region_df["Region"]
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(
        display_labels,
        fontsize=tick_font_size,
        fontweight="bold",
    )
    ax.tick_params(
        axis="x",
        width=1.5,
        length=6,
        pad=10,
    )

    ax.set_ylabel(
        "MAE (nm)",
        fontsize=axis_font_size,
        fontweight="bold",
        labelpad=18,
    )
    ax.tick_params(
        axis="y",
        labelsize=tick_font_size,
        width=1.5,
        length=6,
    )

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.set_title(
        "MAE by Emission Region",
        fontsize=title_font_size,
        fontweight="bold",
        pad=20,
    )

    legend = ax.legend(
        fontsize=legend_font_size,
        frameon=False,
        loc="upper right",
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.set_ylim(0, maximum_mae * 1.18)

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.20,
        top=0.90,
    )

    fig.savefig(
        OUTPUT_DIR / "MAE_by_emission_region_1200dpi.png",
        dpi=1200,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "MAE_by_emission_region.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "MAE_by_emission_region.svg",
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
    )

    plt.show()


def main():
    df = load_data()

    print(f"Dataset: {DATA_PATH}")
    print(f"Samples used: {len(df)}")

    y, predictions = generate_oof_predictions(df)
    error_df, region_df = build_error_tables(y, predictions)

    print("\nMAE by emission region")
    print(
        region_df.to_string(
            index=False,
            formatters={
                "MAE_RF": "{:.2f}".format,
                "MAE_GB": "{:.2f}".format,
            },
        )
    )

    region_df.to_csv(
        OUTPUT_DIR / "MAE_by_emission_region.csv",
        index=False,
    )
    error_df.to_csv(
        OUTPUT_DIR / "RF_GB_10fold_OOF_errors.csv",
        index=False,
    )

    plot_mae_by_region(region_df)

    print(f"\nAnalysis completed. Results saved to:\n{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
