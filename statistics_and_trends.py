"""
Statistics and trends assignment.

This script loads a CSV file named ``data.csv`` and performs:

* basic preprocessing with simple exploratory tools,
* a relational plot,
* a categorical plot,
* a statistical plot, and
* a moment-based analysis with a short textual summary.

You should NOT change any function, file or variable names that are
given in the template.
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


# Column used for all numerical analyses
ANALYSIS_COL = "numeric_col"


def plot_relational_plot(df):
    """
    Create a relational plot for the dataset.

    A simple scatter plot is produced using the first numeric column
    as the x-axis and ``ANALYSIS_COL`` as the y-axis. The figure is
    saved to ``relational_plot.png``.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Identify numeric columns
    numeric = df.select_dtypes(include="number").columns.tolist()

    if ANALYSIS_COL in numeric and len(numeric) > 1:
        # Use the first numeric column that is not the analysis column
        x_col = next(col for col in numeric if col != ANALYSIS_COL)

        sns.scatterplot(
            data=df,
            x=x_col,
            y=ANALYSIS_COL,
            ax=ax,
        )
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(ANALYSIS_COL.replace("_", " ").title())
        ax.set_title("Relational plot")
    else:
        ax.text(
            0.5,
            0.5,
            "Not enough numeric columns for relational plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    plt.savefig("relational_plot.png", dpi=300)
    plt.close(fig)


def plot_categorical_plot(df):
    """
    Create a categorical plot for the dataset.

    If a non-numeric (categorical) column exists, a boxplot of
    ``ANALYSIS_COL`` by that category is drawn. Otherwise, the
    analysis column is binned into quartiles and used as the
    categorical axis. The figure is saved to
    ``categorical_plot.png``.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_cols and ANALYSIS_COL in df.columns:
        cat_col = categorical_cols[0]
        sns.boxplot(
            data=df,
            x=cat_col,
            y=ANALYSIS_COL,
            ax=ax,
        )
        ax.set_xlabel(cat_col.replace("_", " ").title())
        ax.set_ylabel(ANALYSIS_COL.replace("_", " ").title())
        ax.set_title("Categorical plot")
        ax.tick_params(axis="x", rotation=45)
    elif ANALYSIS_COL in df.columns:
        # Fallback: bin numeric column into quartiles
        bins = pd.qcut(
            df[ANALYSIS_COL],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        )
        sns.boxplot(x=bins, y=df[ANALYSIS_COL], ax=ax)
        ax.set_xlabel("Quartile of " + ANALYSIS_COL)
        ax.set_ylabel(ANALYSIS_COL.replace("_", " ").title())
        ax.set_title("Categorical plot (binned numeric column)")
    else:
        ax.text(
            0.5,
            0.5,
            "Data not suitable for categorical plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    plt.savefig("categorical_plot.png", dpi=300)
    plt.close(fig)


def plot_statistical_plot(df):
    """
    Create a statistical plot for the dataset.

    A corner plot is generated for up to three numeric columns,
    including ``ANALYSIS_COL`` wherever possible. The figure is
    saved to ``statistical_plot.png``.
    """
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No numeric columns available for statistical plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        plt.savefig("statistical_plot.png", dpi=300)
        plt.close(fig)
        return

    cols = numeric.columns.tolist()
    # Ensure analysis column is included and limit to at most 3 columns
    if ANALYSIS_COL in cols:
        cols.remove(ANALYSIS_COL)
        cols = [ANALYSIS_COL] + cols
    cols = cols[:3]

    data = numeric[cols].dropna()

    fig = corner(
        data,
        labels=[c.replace("_", " ").title() for c in cols],
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    fig.suptitle("Statistical plot")
    fig.tight_layout()
    fig.savefig("statistical_plot.png", dpi=300)
    plt.close(fig)


def statistical_analysis(df, col: str):
    """
    Compute basic statistical moments for a numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed input data.
    col : str
        Name of the numeric column being analysed.

    Returns
    -------
    mean : float
        Sample mean of the column.
    stddev : float
        Sample standard deviation (ddof = 1).
    skew : float
        Sample skewness.
    excess_kurtosis : float
        Sample excess kurtosis (0.0 for a normal distribution).
    """
    series = pd.to_numeric(df[col], errors="coerce").dropna()

    mean = series.mean()
    stddev = series.std(ddof=1)
    skew = ss.skew(series, bias=False)
    excess_kurtosis = ss.kurtosis(series, fisher=True, bias=False)

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the data and perform quick exploratory checks.

    The function:
    * standardises column names to lower snake case,
    * calls ``describe``, ``corr`` and ``head`` for quick inspection,
    * drops rows with missing values in ``ANALYSIS_COL`` if present.
    """
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Quick exploratory tools (results are not printed, only computed)
    _desc = df.describe(include="all")
    _corr = df.corr(numeric_only=True)
    _head = df.head()

    # Use the variables so linters do not complain about them
    _ = (_desc, _corr, _head)

    if ANALYSIS_COL in df.columns:
        df = df[df[ANALYSIS_COL].notna()]

    return df


def writing(moments, col):
    """
    Print a textual summary of the distribution of ``col``.

    The four moments are printed and the skewness and kurtosis are
    interpreted qualitatively.
    """
    print(f"For the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, and "
        f"Excess Kurtosis = {moments[3]:.2f}."
    )

    skew = moments[2]
    kurt = moments[3]

    if skew > 2:
        skew_desc = "right-skewed"
    elif skew < -2:
        skew_desc = "left-skewed"
    else:
        skew_desc = "approximately symmetric"

    if kurt < -0.5:
        kurt_desc = "platykurtic"
    elif kurt > 0.5:
        kurt_desc = "leptokurtic"
    else:
        kurt_desc = "mesokurtic"

    print(f"The data were {skew_desc} and {kurt_desc}.")
    return


def main():
    """
    Main entry point for the statistics and trends assignment.

    The function reads ``data.csv``, preprocesses it, generates all
    plots, computes the four moments of ``ANALYSIS_COL`` and prints
    a short written summary.
    """
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    col = ANALYSIS_COL

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == "__main__":
    main()
