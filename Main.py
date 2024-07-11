
from Data_processing import load_and_preprocess_data
from EDA import plot_energy_demand, calculate_statistics, plot_distribution, shapiro_test
from Feature_Eng import feature_engineering
from Model_Train import train_linear_regression, train_random_forest
from Model_eval import evaluate_model, plot_residual_analysis, plot_forecasting_results

# Load and preprocess the data
data_path = "/Users/syedmohathashimali/DATA_PROJECTS/Data/spain_energy_market.csv"
data = load_and_preprocess_data(data_path)

# Exploratory Data Analysis
plot_energy_demand(data)
mean, std = calculate_statistics(data)
plot_distribution(data, mean, std)
shapiro_test(data["energy"])

# Feature Engineering
data, features, targets = feature_engineering(data)

# Split Data
X_train = data.loc["2016":"2017", features]
X_test = data.loc["2018", features]
y_train = data.loc["2016":"2017", targets]
y_test = data.loc["2018", targets]

# Train and Evaluate Linear Regression Model
lr_model = train_linear_regression(X_train, y_train, X_test, y_test)
evaluate_model(y_train, lr_model.predict(X_train), y_test, lr_model.predict(X_test), std, mean)

# Train and Evaluate Random Forest Model
rf_model, p_train, p_test = train_random_forest(X_train, y_train, X_test, y_test)
evaluate_model(y_train, p_train, y_test, p_test, std, mean)

# Additional Residual Analysis and Forecasting Plots
plot_residual_analysis(y_test["target_t1"] - p_test)
plot_forecasting_results(y_train, lr_model.predict(X_train), y_test, p_test)
