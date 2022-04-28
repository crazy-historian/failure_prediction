import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def show_slice(data: pd.DataFrame, labels: pd.DataFrame, pipeline: Pipeline, regressor) -> pd.DataFrame:
    data_prepared = pipeline.fit_transform(data)
    return pd.DataFrame({'prediction': regressor.predict(data_prepared), 'expected': list(labels)})


def get_rmse(data: pd.DataFrame, labels: pd.DataFrame, regressor) -> float:
    predictions = regressor.predict(data)
    mse = mean_squared_error(labels, predictions)
    return np.sqrt(mse)


def get_r2(data: pd.DataFrame, labels: pd.DataFrame, regressor) -> float:
    return r2_score(regressor.predict(data), labels)


def cross_val_evaluation(data: pd.DataFrame, labels: pd.DataFrame, regressor, cv: int = 10) -> tuple:
    scores = cross_val_score(regressor, data, labels,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean(), rmse_scores.std()
