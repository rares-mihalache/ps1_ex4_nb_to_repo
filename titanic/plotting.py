"""Plotting functions for the Titanic data.

The notebook draws the same two kinds of plots for many columns, so each kind
is written once here as a function with the column and the hue as parameters.
Figures for the report are saved to the "figures" folder with ``save_fig``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# Defined once here, so it exists before any plot uses it (the notebook used it
# one cell before defining it).
COLOR_LIST = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]

# The "figures" folder at the root of the repository.
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def save_fig(fig: Figure, name: str) -> Path:
    """Save a figure as a PNG file in the "figures" folder.

    Args:
        fig: The figure to save.
        name: The file name without the extension, e.g. "survival_by_sex".

    Returns:
        The path of the saved file.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_count_pairs(
    data_df: pd.DataFrame,
    feature: str,
    title: str | None = None,
    hue: str = "set",
    save_as: str | None = None,
) -> None:
    """Plot the number of passengers for each value of a column, split by ``hue``.

    Args:
        data_df: The data to plot, e.g. ``all_df`` or ``train_df``.
        feature: The column whose values go on the x-axis, e.g. "Sex".
        title: The text after "Number of passengers / " in the plot title.
            Defaults to the name of ``feature``.
        hue: The column used to split and colour the bars. The default "set"
            compares the train and test data; use "Survived" to compare
            survivors and non-survivors.
        save_as: If given, the plot is also saved to the "figures" folder
            under this name (see ``save_fig``).
    """
    # Drop rows without a hue value (test passengers have no "Survived") and
    # renumber the rows: all_df repeats index labels, which makes seaborn fail
    # on columns with missing values such as "Embarked".
    data_df = data_df.dropna(subset=[hue]).reset_index(drop=True)
    # One colour per hue value, otherwise seaborn warns about unused colours.
    n_colors = data_df[hue].nunique()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=feature, data=data_df, hue=hue, palette=COLOR_LIST[:n_colors])
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {title or feature}")
    if save_as is not None:
        save_fig(fig, save_as)
    plt.show()


def plot_distribution_pairs(
    data_df: pd.DataFrame,
    feature: str,
    title: str | None = None,
    hue: str = "set",
    save_as: str | None = None,
) -> None:
    """Plot overlapping histograms of a column, one for each value of ``hue``.

    Args:
        data_df: The data to plot, e.g. ``all_df`` or ``train_df``.
        feature: The column to draw the histograms for, e.g. "Age".
        title: The text after "Number of passengers / " in the plot title.
            Defaults to the name of ``feature``.
        hue: The column that splits the data into one histogram per value.
            The default "set" compares the train and test data; use
            "Survived" to compare survivors and non-survivors.
        save_as: If given, the plot is also saved to the "figures" folder
            under this name (see ``save_fig``).
    """
    # Same data preparation as in plot_count_pairs.
    data_df = data_df.dropna(subset=[hue]).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, hue_value in enumerate(data_df[hue].unique()):
        sns.histplot(
            data_df.loc[data_df[hue] == hue_value],
            x=feature,
            color=COLOR_LIST[i],
            ax=ax,
            label=hue_value,
        )
    ax.set_title(f"Number of passengers / {title or feature}")
    ax.legend()
    if save_as is not None:
        save_fig(fig, save_as)
    plt.show()
