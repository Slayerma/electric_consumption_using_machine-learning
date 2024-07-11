import pandas as pd

def feature_engineering(data):
    data["target"] = data.energy.add(-data.energy.mean()).div(data.energy.std())
    features = []
    targets = []
    tau = 30  # forecasting periods

    for t in range(1, tau + 1):
        data[f"target_t{t}"] = data.target.shift(-t)
        targets.append(f"target_t{t}")

    for t in range(1, 31):
        data[f"feat_ar{t}"] = data.target.shift(t)
        features.append(f"feat_ar{t}")

    for t in [7, 14, 30]:
        data[[f"feat_movave{t}", f"feat_movstd{t}", f"feat_movmin{t}", f"feat_movmax{t}"]] = data.energy.rolling(t).agg([pd.Series.mean, pd.Series.std, pd.Series.max, pd.Series.min])
        features.append(f"feat_movave{t}")
        features.append(f"feat_movstd{t}")
        features.append(f"feat_movmin{t}")
        features.append(f"feat_movmax{t}")

    months = pd.get_dummies(data.mon, prefix="mon", drop_first=True)
    months.index = data.index
    data = pd.concat([data, months], axis=1)

    days = pd.get_dummies(data.day, prefix="day", drop_first=True)
    days.index = data.index
    data = pd.concat([data, days], axis=1)

    features += months.columns.values.tolist() + days.columns.values.tolist()
    
    return data, features, targets
