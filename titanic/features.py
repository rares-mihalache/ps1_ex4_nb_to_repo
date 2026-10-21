"""Feature engineering for the Titanic data.

Each function takes a DataFrame, adds one group of engineered features and
returns a new DataFrame. This lets the notebook apply the same code to both
``all_df`` and ``train_df`` instead of writing every block twice.
"""

import pandas as pd


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Add the family size: siblings/spouses + parents/children + the passenger.

    Args:
        df: DataFrame with "SibSp" and "Parch" columns.

    Returns:
        A copy of ``df`` with a new "Family Size" column.
    """
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df


def add_age_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Group "Age" into five intervals.

    The intervals are 0: age <= 16, 1: 16-32, 2: 32-48, 3: 48-64 and 4: > 64.
    Missing ages stay in interval 0, as in the original notebook.

    Args:
        df: DataFrame with an "Age" column.

    Returns:
        A copy of ``df`` with a new "Age Interval" column.
    """
    df = df.copy()
    df["Age Interval"] = 0.0
    df.loc[df["Age"] <= 16, "Age Interval"] = 0
    df.loc[(df["Age"] > 16) & (df["Age"] <= 32), "Age Interval"] = 1
    df.loc[(df["Age"] > 32) & (df["Age"] <= 48), "Age Interval"] = 2
    df.loc[(df["Age"] > 48) & (df["Age"] <= 64), "Age Interval"] = 3
    df.loc[df["Age"] > 64, "Age Interval"] = 4
    return df


def add_fare_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Group "Fare" into four intervals.

    The intervals are 0: fare <= 7.91, 1: 7.91-14.454, 2: 14.454-31 and
    3: > 31. Missing fares stay in interval 0, as in the original notebook.

    Args:
        df: DataFrame with a "Fare" column.

    Returns:
        A copy of ``df`` with a new "Fare Interval" column.
    """
    df = df.copy()
    df["Fare Interval"] = 0.0
    df.loc[df["Fare"] <= 7.91, "Fare Interval"] = 0
    df.loc[(df["Fare"] > 7.91) & (df["Fare"] <= 14.454), "Fare Interval"] = 1
    df.loc[(df["Fare"] > 14.454) & (df["Fare"] <= 31), "Fare Interval"] = 2
    df.loc[df["Fare"] > 31, "Fare Interval"] = 3
    return df


def add_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """Combine sex and passenger class into one feature, e.g. "F_C1" or "M_C3".

    Args:
        df: DataFrame with "Sex" and "Pclass" columns.

    Returns:
        A copy of ``df`` with a new "Sex_Pclass" column.
    """
    df = df.copy()
    df["Sex_Pclass"] = df.apply(
        lambda row: row["Sex"][0].upper() + "_C" + str(row["Pclass"]), axis=1
    )
    return df


def parse_names(row: pd.Series) -> pd.Series:
    """Split a passenger's "Name" into family name, title, given name and maiden name.

    Names look like "Braund, Mr. Owen Harris" or
    "Cumings, Mrs. John Bradley (Florence Briggs Thayer)".

    Args:
        row: A row of the Titanic data with a "Name" field.

    Returns:
        A Series with the family name, title, given name and maiden name
        (None if there is no maiden name).
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")


def add_name_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Add the parts of each passenger's name as separate columns.

    Args:
        df: DataFrame with a "Name" column.

    Returns:
        A copy of ``df`` with new "Family Name", "Title", "Given Name" and
        "Maiden Name" columns.
    """
    df = df.copy()
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(
        parse_names, axis=1
    )
    return df


def add_family_type(df: pd.DataFrame) -> pd.DataFrame:
    """Group "Family Size" into "Single" (1), "Small" (2-4) and "Large" (5+).

    Args:
        df: DataFrame with a "Family Size" column (see ``add_family_size``).

    Returns:
        A copy of ``df`` with a new "Family Type" column.
    """
    df = df.copy()
    # Start from an object column: pandas 3 no longer lets us write strings
    # into the integer "Family Size" copy, which the original notebook did.
    df["Family Type"] = df["Family Size"].astype(object)
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[df["Family Size"] >= 5, "Family Type"] = "Large"
    return df


def add_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Add a "Titles" column that groups similar and rare titles together.

    "Mlle." and "Ms." become "Miss.", "Mme." becomes "Mrs.", and rare titles
    such as "Dr." or "Rev." become "Rare".

    Args:
        df: DataFrame with a "Title" column (see ``add_name_parts``).

    Returns:
        A copy of ``df`` with a new "Titles" column.
    """
    df = df.copy()
    df["Titles"] = df["Title"]
    df["Titles"] = df["Titles"].replace("Mlle.", "Miss.")
    df["Titles"] = df["Titles"].replace("Ms.", "Miss.")
    df["Titles"] = df["Titles"].replace("Mme.", "Mrs.")
    rare_titles = [
        "Lady.", "the Countess.", "Capt.", "Col.", "Don.", "Dr.",
        "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona.",
    ]
    df["Titles"] = df["Titles"].replace(rare_titles, "Rare")
    return df


def survival_by_title_and_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the survival rate for each combination of title and sex.

    Args:
        df: Training DataFrame with "Titles", "Sex" and "Survived" columns.

    Returns:
        A table with the mean of "Survived" for each ("Titles", "Sex") group.
    """
    return df[["Titles", "Sex", "Survived"]].groupby(["Titles", "Sex"], as_index=False).mean()


def map_sex_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Encode "Sex" as a number for the model: female = 1, male = 0.

    Args:
        df: DataFrame with a "Sex" column containing "female" and "male".

    Returns:
        A copy of ``df`` where "Sex" is an integer column.
    """
    df = df.copy()
    df["Sex"] = df["Sex"].map({"female": 1, "male": 0}).astype(int)
    return df
