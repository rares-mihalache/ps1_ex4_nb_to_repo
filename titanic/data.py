from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def load_data(
    train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Titanic train and test data.

    Args:
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.

    Returns:
        A tuple (train_df, test_df).
    """
    return pd.read_csv(train_path), pd.read_csv(test_path)


def quick_glimpse(df: pd.DataFrame) -> None:
    """Display the first rows, column information and summary statistics.

    Args:
        df: The DataFrame to inspect.
    """
    display(df.head())
    df.info()
    display(df.describe())


def missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values for each column.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A table with the number and percentage of missing values,
        and the data type, for each column.
    """
    total = df.isnull().sum()
    percent = df.isnull().sum() / df.isnull().count() * 100
    tt = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    tt["Types"] = [str(df[col].dtype) for col in df.columns]
    return np.transpose(tt)


def most_frequent_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Find the most frequent value in each column.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A table with the most frequent value, how often it appears,
        and its percentage of the non-missing values, for each column.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ["Total"]
    items = []
    vals = []
    for col in df.columns:
        counts = df[col].value_counts()
        if counts.empty:
            items.append(0)
            vals.append(0)
        else:
            items.append(counts.index[0])
            vals.append(counts.values[0])
    tt["Most frequent item"] = items
    tt["Frequence"] = vals
    tt["Percent from total"] = np.round(vals / total * 100, 3)
    return np.transpose(tt)


def unique_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of unique values in each column.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A table with the number of non-missing and unique values
        for each column.
    """
    tt = pd.DataFrame(df.count())
    tt.columns = ["Total"]
    tt["Uniques"] = [df[col].nunique() for col in df.columns]
    return np.transpose(tt)


def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Stack the train and test data into one DataFrame.

    Args:
        train_df: The training data (with a "Survived" column).
        test_df: The test data (without "Survived").

    Returns:
        A DataFrame with all passengers and a "set" column that says whether
        each row comes from "train" or "test".
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df
