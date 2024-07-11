
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

def train_linear_regression(X_train, y_train, X_test, y_test):
    reg = LinearRegression().fit(X_train, y_train["target_t1"])
    p_train = reg.predict(X_train)
    p_test = reg.predict(X_test)

    RMSE_train = np.sqrt(mean_squared_error(y_train["target_t1"], p_train))
    RMSE_test = np.sqrt(mean_squared_error(y_test["target_t1"], p_test))

    print("Train RMSE: {}\nTest RMSE: {}".format(RMSE_train, RMSE_test))
    return reg

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray]:
    splits = TimeSeriesSplit(n_splits=3, max_train_size=365 * 2)
    rfr = RandomForestRegressor()
    rfr_grid = {
        "n_estimators": [500],
        "max_depth": [3, 5, 10, 20, 30],
        "max_features": [4, 8, 16, 32, 59],
        "random_state": [123],
    }
    rfr_paramGrid = list(ParameterGrid(rfr_grid))

    def time_split_mod_build(
        model: RandomForestRegressor,
        paramGrid: list[dict[str, int]],
        splits: TimeSeriesSplit,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        train_scores = np.empty(len(paramGrid))
        val_scores = np.empty(len(paramGrid))

        for idx, g in enumerate(paramGrid):
            model.set_params(**g)
            for train_idx, val_idx in splits.split(X):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                p_train = model.predict(X.iloc[train_idx])
                p_val = model.predict(X.iloc[val_idx])
                train_scores[idx] = np.mean(
                    mean_squared_error(y.iloc[train_idx], p_train)
                )
                val_scores[idx] = np.mean(mean_squared_error(y.iloc[val_idx], p_val))

        best_idx = np.argmin(val_scores)
        return train_scores, val_scores, best_idx

    CV_rfr_tup = time_split_mod_build(
        rfr, rfr_paramGrid, splits, X_train, y_train["target_t1"]
    )
    best_rfr_idx = CV_rfr_tup[2]
    best_rfr_grid = rfr_paramGrid[best_rfr_idx]
    best_rfr = RandomForestRegressor(**best_rfr_grid).fit(
        X_train.loc["2016":"2017"], y_train.loc["2016":"2017", "target_t1"]
    )
    importances = best_rfr.feature_importances_
    sorted_index = np.argsort(importances)[::-1]
    sorted_index_top = sorted_index[:10]
    labels = np.array(X_train.columns)[sorted_index_top]
    plt.bar(range(len(sorted_index_top)), importances[sorted_index_top], tick_label=labels)
    plt.title("Feature importance analysis")
    plt.xticks(rotation=45)
    plt.show()

    p_train = best_rfr.predict(X_train)
    p_test = best_rfr.predict(X_test)

    return best_rfr, p_train, p_test
