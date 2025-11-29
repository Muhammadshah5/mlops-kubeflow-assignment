from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename("target")], axis=1)
df.to_csv("data/raw/raw_data.csv", index=False)
print("Saved data/raw/raw_data.csv")
