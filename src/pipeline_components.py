import os
from kfp.dsl import component, Output, Dataset, Model

# -----------------------------
#  Data Extraction Component
# -----------------------------
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "dvc"]
)
def data_extraction(output_data: Output[Dataset]):
    """Fetch versioned dataset from DVC remote"""
    import os
    
    os.system("dvc pull")
    os.system(f"cp data/raw/raw_data.csv {output_data.path}")


# -----------------------------
#  Data Preprocessing Component
# -----------------------------
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def data_preprocessing(
    input_data: Dataset,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(input_data.path)
    
    if "target" not in df.columns:
        df.columns = list(df.columns[:-1]) + ["target"]
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    pd.concat([X_train, y_train], axis=1).to_csv(train_data.path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_data.path, index=False)


# -----------------------------
#  Model Training Component
# -----------------------------
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def model_training(train_data: Dataset, model: Output[Model]):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    df = pd.read_csv(train_data.path)
    X_train = df.drop(columns=["target"])
    y_train = df["target"]
    
    model_rf = RandomForestRegressor(n_estimators=50, random_state=42)
    model_rf.fit(X_train, y_train)
    
    joblib.dump(model_rf, model.path)


# -----------------------------
#  Model Evaluation Component
# -----------------------------
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def model_evaluation(
    model: Model,
    test_data: Dataset,
    metrics: Output[Dataset]
):
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    
    df = pd.read_csv(test_data.path)
    X_test = df.drop(columns=["target"])
    y_test = df["target"]
    
    loaded_model = joblib.load(model.path)
    y_pred = loaded_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    with open(metrics.path, "w") as f:
        f.write(f"MSE: {mse}\nR2: {r2}\n")