from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(
    r"C:\Users\anush\Desktop\ML projects\separeted by number of dopands"
    r"\single_doped\group_one_dopant_without_exact_duplicates.csv"
)

OUTPUT_DIR = Path("correlation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "1st dopant valency",
    "EN_ligand_avg",
    "Excitation source",
    "Ionization Energy_sum",
    "avg_d_electrons",
    "ionic_radius_emission_center",
    "ionic_radius_substituted",
]

DISPLAY_LABELS = {
    "1st dopant valency": "1st dopant\nvalency",
    "EN_ligand_avg": "EN_ligand_avg",
    "Excitation source": "Excitation\nsource",
    "Ionization Energy_sum": "Ionization\nEnergy_sum",
    "avg_d_electrons": "avg_d_electrons",
    "ionic_radius_emission_center": "ionic_radius_\nemission_center",
    "ionic_radius_substituted": "ionic_radius_\nsubstituted",
}

DPI = 1200
ANNOTATE_VALUES = True

TITLE_FONT_SIZE = 26
X_TICK_FONT_SIZE = 17
Y_TICK_FONT_SIZE = 17
CELL_FONT_SIZE = 15
COLORBAR_LABEL_SIZE = 20
COLORBAR_TICK_SIZE = 17


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\nUpdate DATA_PATH before running."
        )

    if DATA_PATH.suffix.lower() == ".csv":
        df = pd.read_csv(DATA_PATH)
    elif DATA_PATH.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(DATA_PATH)
    else:
        raise ValueError("Dataset must be CSV, XLSX, or XLS.")

    df.columns = df.columns.str.strip()

    print(f"Dataset: {DATA_PATH.name}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def prepare_descriptors(df):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing:
        raise KeyError(
            "Missing required descriptors:\n- " + "\n- ".join(missing)
        )

    data = df[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)

    print("\nMissing values:")
    print(data.isna().sum().to_string())

    constant = [
        c for c in FEATURE_COLUMNS
        if data[c].nunique(dropna=True) <= 1
    ]

    if constant:
        raise ValueError(
            "Correlation cannot be calculated for constant descriptors:\n- "
            + "\n- ".join(constant)
        )

    return data


def draw_correlation_heatmap(correlation_matrix, title, output_stem):
    labels = [DISPLAY_LABELS[c] for c in correlation_matrix.columns]
    matrix = correlation_matrix.to_numpy()

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.045,
    )

    colorbar.set_label(
        "Correlation coefficient",
        fontsize=COLORBAR_LABEL_SIZE,
        fontweight="bold",
        labelpad=16,
    )

    colorbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        width=1.4,
        length=6,
    )

    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=90,
        ha="center",
        va="top",
        fontsize=X_TICK_FONT_SIZE,
        fontweight="bold",
    )

    ax.set_yticklabels(
        labels,
        fontsize=Y_TICK_FONT_SIZE,
        fontweight="bold",
    )

    ax.tick_params(axis="x", length=0, pad=14)
    ax.tick_params(axis="y", length=0, pad=14)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)

    ax.grid(which="minor", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if ANNOTATE_VALUES:
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix[row, column]

                if pd.isna(value):
                    text = "NaN"
                    reference_value = 0.0
                else:
                    text = f"{value:.2f}"
                    reference_value = value

                rgba = image.cmap(image.norm(reference_value))
                luminance = (
                    0.299 * rgba[0]
                    + 0.587 * rgba[1]
                    + 0.114 * rgba[2]
                )

                text_color = "black" if luminance > 0.58 else "white"

                ax.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=CELL_FONT_SIZE,
                    fontweight="bold",
                    color=text_color,
                )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=25,
    )

    fig.subplots_adjust(
        left=0.27,
        right=0.88,
        bottom=0.30,
        top=0.90,
    )

    fig.savefig(
        OUTPUT_DIR / f"{output_stem}_1200dpi.png",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.20,
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_DIR / f"{output_stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.20,
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_DIR / f"{output_stem}.svg",
        bbox_inches="tight",
        pad_inches=0.20,
        facecolor="white",
    )

    plt.show()
    plt.close(fig)


def create_pairwise_table(pearson_matrix, spearman_matrix):
    records = []

    for i in range(len(FEATURE_COLUMNS)):
        for j in range(i + 1, len(FEATURE_COLUMNS)):
            descriptor_1 = FEATURE_COLUMNS[i]
            descriptor_2 = FEATURE_COLUMNS[j]

            pearson = pearson_matrix.loc[
                descriptor_1,
                descriptor_2,
            ]

            spearman = spearman_matrix.loc[
                descriptor_1,
                descriptor_2,
            ]

            records.append(
                {
                    "Descriptor 1": descriptor_1,
                    "Descriptor 2": descriptor_2,
                    "Pearson correlation": pearson,
                    "Absolute Pearson": abs(pearson),
                    "Spearman correlation": spearman,
                    "Absolute Spearman": abs(spearman),
                }
            )

    result = pd.DataFrame(records)

    result["Maximum absolute correlation"] = result[
        ["Absolute Pearson", "Absolute Spearman"]
    ].max(axis=1)

    return result.sort_values(
        "Maximum absolute correlation",
        ascending=False,
    ).reset_index(drop=True)


def main():
    df = load_data()
    descriptor_data = prepare_descriptors(df)

    pearson_matrix = descriptor_data.corr(
        method="pearson",
        min_periods=3,
    )

    spearman_matrix = descriptor_data.corr(
        method="spearman",
        min_periods=3,
    )

    pairwise_table = create_pairwise_table(
        pearson_matrix,
        spearman_matrix,
    )

    pearson_matrix.to_csv(
        OUTPUT_DIR / "Pearson_Correlation_Seven_Descriptors.csv"
    )

    spearman_matrix.to_csv(
        OUTPUT_DIR / "Spearman_Correlation_Seven_Descriptors.csv"
    )

    pairwise_table.to_csv(
        OUTPUT_DIR / "Pearson_Spearman_Pairwise_Comparison.csv",
        index=False,
    )

    draw_correlation_heatmap(
        pearson_matrix,
        "Pearson Correlation of Selected Descriptors",
        "Pearson_Heatmap_Seven_Descriptors",
    )

    draw_correlation_heatmap(
        spearman_matrix,
        "Spearman Correlation of Selected Descriptors",
        "Spearman_Heatmap_Seven_Descriptors",
    )

    print("\nPearson correlation matrix:")
    print(pearson_matrix.round(3).to_string())

    print("\nSpearman correlation matrix:")
    print(spearman_matrix.round(3).to_string())

    print("\nStrongest descriptor relationships:")
    print(
        pairwise_table.head(10)
        .round(3)
        .to_string(index=False)
    )

    print(
        f"\nAnalysis completed. Results saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
