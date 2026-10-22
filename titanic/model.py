"""Training and evaluating the baseline model for the Titanic data.

The model parameters are defined once at the top of this module instead of
being hard-coded in the notebook.
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

VALID_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
TARGET_NAMES = ["Not Survived", "Survived"]


def split_train_valid(
    df: pd.DataFrame, valid_size: float = VALID_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training data into a training and a validation set.

    Args:
        df: The training data.
        valid_size: The share of rows that go into the validation set.

    Returns:
        A tuple (train, valid).
    """
    train, valid = train_test_split(
        df, test_size=valid_size, random_state=RANDOM_STATE, shuffle=True
    )
    return train, valid


def get_features_and_target(
    df: pd.DataFrame, predictors: list[str], target: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate the predictor features from the target labels.

    Args:
        df: The data to separate, e.g. ``train`` or ``valid``.
        predictors: The columns used to make the prediction.
        target: The column to predict.

    Returns:
        A tuple (X, Y) with the predictor columns and the target values.
    """
    return df[predictors], df[target].to_numpy()


def train_random_forest(
    train_X: pd.DataFrame, train_Y: np.ndarray
) -> RandomForestClassifier:
    """Create a random forest classifier and fit it to the training data.

    The fitted model is returned rather than its predictions, so the same model
    can predict any data set, e.g. the validation data or ``test_df``.

    Args:
        train_X: The predictor features of the training data.
        train_Y: The labels of the training data.

    Returns:
        The fitted classifier.
    """
    clf = RandomForestClassifier(
        n_jobs=-1,
        random_state=RANDOM_STATE,
        criterion="gini",
        n_estimators=N_ESTIMATORS,
        verbose=False,
    )
    clf.fit(train_X, train_Y)
    return clf


def print_classification_report(true_Y: np.ndarray, preds: np.ndarray) -> None:
    """Print the precision, recall and F1-score for both classes.

    Args:
        true_Y: The true labels.
        preds: The predicted labels.
    """
    print(metrics.classification_report(true_Y, preds, target_names=TARGET_NAMES))
