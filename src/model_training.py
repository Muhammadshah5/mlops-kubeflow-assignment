import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump

df = pd.read_csv("data/raw/raw_data.csv")
if 'target' not in df.columns:
    # try last column as fallback
    df.columns = list(df.columns[:-1]) + ['target']

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
dump(model, "model.joblib")
print("Model trained and saved to model.joblib")
