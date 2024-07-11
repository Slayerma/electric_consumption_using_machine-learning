# model_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

def evaluate_model(y_train, p_train, y_test, p_test, std, mean):
    # Calculate residuals
    train_resid_1step = y_train["target_t1"] - p_train
    test_resid_1step = y_test["target_t1"] - p_test

    # Calculate Mean Absolute Percentage Error (MAPE) and Mean Absolute Error (MAE)
    test_df = y_test[["target_t1"]] * std + mean
    test_df["pred_t1"] = p_test * std + mean
    test_df["resid_t1"] = test_df["target_t1"] - test_df["pred_t1"]
    test_df["abs_resid_t1"] = abs(test_df["resid_t1"])
    test_df["ape_t1"] = test_df["abs_resid_t1"] / test_df["target_t1"]

    test_MAPE = test_df["ape_t1"].mean() * 100
    test_MAE = test_df["abs_resid_t1"].mean()

    # Print evaluation metrics
    print("Train RMSE: {}\nTest RMSE: {}".format(np.sqrt(mean_squared_error(y_train["target_t1"], p_train)), np.sqrt(mean_squared_error(y_test["target_t1"], p_test))))
    print("Test MAPE: {} \nTest MAE: {}".format(test_MAPE, test_MAE))
    
    # Plot predictions
    test_df[["target_t1", "pred_t1"]].plot()
    plt.ylabel("MWh")
    plt.title("1-Step Ahead Predictions")
    plt.show()

def plot_residual_analysis(test_resid_1step):
    # Plot histogram of residuals
    test_resid_1step.plot.hist(bins=10, title="Test 1-step ahead residuals distribution")
    plt.xlabel("Residuals")
    plt.show()

    # Plot residuals time series
    test_resid_1step.plot(title="Test 1-step ahead residuals time series")
    plt.ylabel("Residuals")
    plt.show()

def plot_forecasting_results(y_train, p_train, y_test, p_test):
    # Scatter plot of actual vs predicted values
    plt.scatter(y=y_train["target_t1"], x=p_train, label="train")
    plt.scatter(y=y_test["target_t1"], x=p_test, label="test")
    plt.title("1-period ahead Actual vs Forecasting")
    plt.ylabel("Actual")
    plt.xlabel("Forecast")
    plt.legend()
    plt.show()
