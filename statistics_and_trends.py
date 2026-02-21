"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

from pathlib import Path

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


# Column we will analyse throughout the script
ANALYSIS_COL = "co2_per_capita"


def plot_relational_plot(df: pd.DataFrame) -> None:
    """
    Make a relational plot for the GDP–CO2 relationship.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed data containing at least the columns
        'gdp_per_capita' and ANALYSIS_COL.
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
        ax.set_title("GDP per capita vs CO2 per capita (2020)")
        ax.legend(title="GDP quartile", loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "Required columns not found.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    plt.savefig("relational_plot.png", dpi=300)
    plt.close(fig)


def plot_categorical_plot(df: pd.DataFrame) -> None:
    """
    Make a categorical plot: CO2 per capita by GDP quartile.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed data containing 'gdp_quartile' and ANALYSIS_COL.
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
        ax.set_title("CO2 per capita by GDP quartile (2020)")
    else:
        ax.text(
            0.5,
            0.5,
            "Required columns not found.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    plt.savefig("categorical_plot.png", dpi=300)
    plt.close(fig)


def plot_statistical_plot(df: pd.DataFrame) -> None:
    """
    Make a univariate statistical plot for the analysis column.

    Here we use a histogram with a kernel density estimate (KDE)
    to visualise the distribution of ANALYSIS_COL.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if ANALYSIS_COL in df.columns:
        sns.histplot(
            df[ANALYSIS_COL],
            bins=30,
            kde=True,
            ax=ax,
        )
        ax.set_xlabel("CO2 per capita (metric tons)")
        ax.set_ylabel("Count of countries")
        ax.set_title("Distribution of CO2 per capita across countries (2020)")
    else:
        ax.text(
            0.5,
            0.5,
            f"Column '{ANALYSIS_COL}' not found.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    plt.savefig("statistical_plot.png", dpi=300)
    plt.close(fig)


def statistical_analysis(df: pd.DataFrame, col: str):
    """
    Compute the first four central moments for a numeric column.

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
    series = pd.to_numeric(df[col], errors="coerce").dropna()

    mean = series.mean()
    stddev = series.std(ddof=1)
    skew = ss.skew(series, bias=False)
    excess_kurtosis = ss.kurtosis(series, fisher=True, bias=False)

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the World Bank GDP/CO2 dataset.

    Steps
    -----
    - Standardise column names.
    - Filter to year 2020.
    - Drop obvious regional/income aggregates.
    - Create GDP quartiles.
    - Drop rows with missing analysis values.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame ready for analysis and plotting.
    """
    df = df.copy()

    # Standardise column names (spaces -> underscores)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Focus on latest year
    if "year" in df.columns:
        df = df[df["year"] == 2020]

    # Remove aggregate regions/income groups based on name keywords
    agg_keywords = [
        "World",
        "income",
        "dividend",
        "states",
        "Union",
        "area",
        "Europe",
        "OECD",
        "IBRD",
        "IDA",
        "Heavily indebted",
        "Fragile",
        "Caribbean",
        "North America",
        "South Asia",
        "Sub-Saharan",
        "Middle East",
        "Latin America",
        "Africa",
        "East Asia",
        "Pacific",
        "Arab World",
        "small states",
        "Central Europe",
        "Post-demographic",
        "Pre-demographic",
        "Early-demographic",
        "Late-demographic",
    ]

    if "Country_Name" in df.columns:
        mask_agg = df["Country_Name"].astype(str).apply(
            lambda s: any(key in s for key in agg_keywords)
        )
        df = df[~mask_agg]

    # Drop missing values in the analysis column
    if ANALYSIS_COL in df.columns:
        df = df[df[ANALYSIS_COL].notna()]

    # Create GDP quartiles for categorical plotting
    if "gdp_per_capita" in df.columns:
        df["gdp_quartile"] = pd.qcut(
            df["gdp_per_capita"],
            4,
            labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        )

    # Quick diagnostics (optional; can be commented out)
    # print(df.describe(include="all"))
    # print(df.corr(numeric_only=True))

    return df


def writing(moments, col: str) -> None:
    """
    Print a short textual summary of the distribution moments.
    """
    print(f"For the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, and "
        f"Excess Kurtosis = {moments[3]:.2f}."
    )

    # Interpret skewness and kurtosis in plain language.
    skew, kurt = moments[2], moments[3]

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
        kurt_desc = "mesokurtic (similar to normal)"

    print(f"The data are {skew_desc} and {kurt_desc}.")
    return


def main() -> None:
    """
    Main entry point for the statistics and trends assignment.
    """
    # Resolve data.csv relative to this script so it works on any OS
    here = Path(__file__).parent
    csv_path = here / "data.csv"

    df = pd.read_csv(csv_path)
    df = preprocessing(df)

    col = ANALYSIS_COL

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == "__main__":
    main()
