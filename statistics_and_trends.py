"""
Statistics and trends assignment.

This script loads a CSV file called 'data.csv' and performs:
- basic preprocessing,
- a relational plot,
- a categorical plot,
- a statistical plot, and
- a simple moment-based analysis with textual interpretation.

You should NOT change any function, file or variable names,
if they are given to you here.
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

# Column chosen for detailed analysis
ANALYSIS_COL = "co2_per_capita"


def plot_relational_plot(df):
    """
    Relational plot between GDP per capita and CO2 per capita.

    Produces a scatter plot coloured by GDP quartile and
    saves it as 'relational_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if {"gdp_per_capita", ANALYSIS_COL}.issubset(df.columns):
        sns.scatterplot(
            data=df,
            x="gdp_per_capita",
            y=ANALYSIS_COL,
            hue="gdp_quartile",
            palette="viridis",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xlabel("GDP per capita (current US$)")
        ax.set_ylabel("CO2 per capita (metric tons)")
        ax.set_title("Relational: GDP per capita vs CO2 per capita (2020)")
        ax.legend(title="GDP quartile", loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "Required columns not found for relational plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    plt.savefig("relational_plot.png", dpi=300)
    plt.close(fig)


def plot_categorical_plot(df):
    """
    Categorical plot: distribution of CO2 per capita by GDP quartile.

    Uses a boxplot and saves it as 'categorical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if {"gdp_quartile", ANALYSIS_COL}.issubset(df.columns):
        sns.boxplot(
            data=df,
            x="gdp_quartile",
            y=ANALYSIS_COL,
            ax=ax,
        )
        ax.set_xlabel("GDP quartile")
        ax.set_ylabel("CO2 per capita (metric tons)")
        ax.set_title("Categorical: CO2 per capita by GDP quartile (2020)")
    else:
        ax.text(
            0.5,
            0.5,
            "Required columns not found for categorical plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    plt.savefig("categorical_plot.png", dpi=300)
    plt.close(fig)


def plot_statistical_plot(df):
    """
    Statistical plot of the joint numeric structure.

    Uses a corner plot (pairwise scatter + 1D histograms)
    for GDP per capita and CO2 per capita, saved as
    'statistical_plot.png'.
    """
    numeric_cols = []
    for col in ["gdp_per_capita", ANALYSIS_COL]:
        if col in df.columns:
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        # Fall back to a simple histogram of the analysis column.
        fig, ax = plt.subplots(figsize=(8, 6))
        if ANALYSIS_COL in df.columns:
            sns.histplot(df[ANALYSIS_COL], bins=30, kde=True, ax=ax)
            ax.set_xlabel("CO2 per capita (metric tons)")
            ax.set_ylabel("Count")
            ax.set_title("Statistical: distribution of CO2 per capita")
        else:
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

    data = df[numeric_cols].dropna()

    fig = corner(
        data,
        labels=[col.replace("_", " ").title() for col in numeric_cols],
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    fig.suptitle("Statistical: joint distribution of key numeric attributes")
    fig.tight_layout()
    fig.savefig("statistical_plot.png", dpi=300)
    plt.close(fig)


def statistical_analysis(df, col: str):
    """
    Compute basic moments for a numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed data.
    col : str
        Name of the numeric column to analyse.

    Returns
    -------
    mean : float
    stddev : float
    skew : float
    excess_kurtosis : float
    """
    # Ensure we are working with numeric values only
    series = pd.to_numeric(df[col], errors="coerce").dropna()

    mean = series.mean()
    stddev = series.std(ddof=1)
    skew = ss.skew(series, bias=False)
    excess_kurtosis = ss.kurtosis(series, fisher=True, bias=False)

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the raw data.

    Steps
    -----
    - Standardise column names to lower snake_case.
    - Filter to the latest year (2020) if a 'year' column exists.
    - Drop aggregate rows (regions, income groups) using simple
      keyword matching on 'country_name'.
    - Remove rows with missing values in the analysis column.
    - Create GDP quartiles for categorical plotting.
    """
    df = df.copy()

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Filter to 2020 if year column present
    if "year" in df.columns:
        df = df[df["year"] == 2020]

    # Drop obvious aggregates based on country_name keywords
    if "country_name" in df.columns:
        keywords = [
            "World",
            "income",
            "states",
            "area",
            "Union",
            "Africa",
            "Europe",
            "America",
            "Asia",
            "Pacific",
            "Caribbean",
            "OECD",
            "IBRD",
            "IDA",
            "dividend",
            "small states",
        ]
        mask_agg = df["country_name"].astype(str).apply(
            lambda s: any(k in s for k in keywords)
        )
        df = df[~mask_agg]

    # Drop rows with missing analysis values
    if ANALYSIS_COL in df.columns:
        df = df[df[ANALYSIS_COL].notna()]

    # Create GDP quartiles for categorical plot
    if "gdp_per_capita" in df.columns:
        df["gdp_quartile"] = pd.qcut(
            df["gdp_per_capita"],
            4,
            labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        )

    # Quick checks (commented out to keep output clean)
    # print(df.head())
    # print(df.describe(include="all"))
    # print(df.corr(numeric_only=True))

    return df


def writing(moments, col):
    """
    Print a short written summary of the distribution of `col`.
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

    # Very simple interpretation rules
    if skew > 2:
        skew_desc = "strongly right-skewed"
    elif skew < -2:
        skew_desc = "strongly left-skewed"
    else:
        skew_desc = "approximately symmetric"

    if kurt < -0.5:
        kurt_desc = "platykurtic (light tails)"
    elif kurt > 0.5:
        kurt_desc = "leptokurtic (heavy tails)"
    else:
        kurt_desc = "mesokurtic (similar to a normal distribution)"

    print(f"The data were {skew_desc} and {kurt_desc}.")
    return


def main():
    """
    Main entry point for the statistics and trends assignment.
    """
    # The grader runs this from the directory containing data.csv
    df = pd.read_csv("data.csv")

    df = preprocessing(df)

    # Choose the main column for analysis
    col = ANALYSIS_COL

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == "__main__":
    main()
