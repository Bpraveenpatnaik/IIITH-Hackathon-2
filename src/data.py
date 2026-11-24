"""Load the Ames housing data used in Kaggle's "House Prices" competition.

Uses data/train.csv if present (download it from
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data),
otherwise fetches the same 1,460-row dataset from OpenML (id 42165).
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=42165, as_frame=True).frame
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df.drop(columns="Id")
