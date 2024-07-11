# data_processing.py
import pandas as pd

def load_and_preprocess_data(path):
    data = pd.read_csv(path, sep=",", parse_dates=["datetime"])
    data = data[data["name"] == "Demanda programada PBF total"]
    data["date"] = data["datetime"].dt.date
    data.set_index("date", inplace=True)
    data = data[["value"]]
    data = data.asfreq("D")
    data = data.rename(columns={"value": "energy"})
    data["year"] = data.index.year
    data["qtr"] = data.index.quarter
    data["mon"] = data.index.month
    data["week"] = data.index.isocalendar().week
    data["day"] = data.index.weekday
    data["ix"] = range(0, len(data))
    data[["movave_7", "movstd_7"]] = data.energy.rolling(7).agg([pd.Series.mean, pd.Series.std])
    data[["movave_30", "movstd_30"]] = data.energy.rolling(30).agg([pd.Series.mean, pd.Series.std])
    data[["movave_90", "movstd_90"]] = data.energy.rolling(90).agg([pd.Series.mean, pd.Series.std])
    data[["movave_365", "movstd_365"]] = data.energy.rolling(365).agg([pd.Series.mean, pd.Series.std])
    return data

