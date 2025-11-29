import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime


class CaliforniaHousingPipeline:
    """Complete ML Pipeline with MLflow integration"""
    
    def __init__(self, experiment_name="California Housing Pipeline"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run(self, n_estimators=100, max_depth=10, test_size=0.2, random_state=42):
        """Run complete pipeline"""
        
        with mlflow.start_run(run_name=f"pipeline_run_{self.timestamp}") as parent_run:
            print("=" * 70)
            print("🚀 CALIFORNIA HOUSING ML PIPELINE")
            print("=" * 70)
            
            # Log pipeline config
            mlflow.log_params({
                "pipeline_version": "1.0.0",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "test_size": test_size,
                "random_state": random_state
            })
            
            try:
                # Step 1: Data Extraction
                print("\n📦 STEP 1: Data Extraction")
                print("-" * 70)
                raw_data = self.extract_data()
                
                # Step 2: Data Preprocessing
                print("\n🔧 STEP 2: Data Preprocessing")
                print("-" * 70)
                X_train, X_test, y_train, y_test = self.preprocess_data(
                    raw_data, test_size, random_state
                )
                
                # Step 3: Model Training
                print("\n🤖 STEP 3: Model Training")
                print("-" * 70)
                model = self.train_model(X_train, y_train, n_estimators, max_depth, random_state)
                
                # Step 4: Model Evaluation
                print("\n📊 STEP 4: Model Evaluation")
                print("-" * 70)
                metrics = self.evaluate_model(model, X_test, y_test)
                
                # Log final metrics
                mlflow.log_metrics(metrics)
                mlflow.set_tag("status", "SUCCESS")
                
                print("\n" + "=" * 70)
                print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
                print("=" * 70)
                print(f"🔗 Run ID: {parent_run.info.run_id}")
                print(f"📊 View in MLflow UI: http://localhost:5000")
                print("=" * 70)
                
                return metrics
                
            except Exception as e:
                mlflow.set_tag("status", "FAILED")
                mlflow.log_param("error", str(e))
                print(f"\n❌ Pipeline failed: {e}")
                raise
    
    def extract_data(self):
        """Extract California Housing dataset"""
        with mlflow.start_run(run_name="data_extraction", nested=True):
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            
            mlflow.log_params({
                "dataset_rows": df.shape[0],
                "dataset_columns": df.shape[1],
                "dataset_name": "California Housing"
            })
            
            os.makedirs("data/raw", exist_ok=True)
            raw_path = "data/raw/raw_data.csv"
            df.to_csv(raw_path, index=False)
            mlflow.log_artifact(raw_path, "raw_data")
            
            print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            return df
    
    def preprocess_data(self, df, test_size, random_state):
        """Preprocess and split data"""
        with mlflow.start_run(run_name="data_preprocessing", nested=True):
            if "MedHouseVal" in df.columns:
                df = df.rename(columns={"MedHouseVal": "target"})
            
            missing = df.isnull().sum().sum()
            mlflow.log_metric("missing_values", missing)
            
            if missing > 0:
                print(f"⚠️  Dropping {missing} missing values")
                df = df.dropna()
            
            X = df.drop(columns=["target"])
            y = df["target"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            mlflow.log_params({
                "test_size": test_size,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            os.makedirs("data/processed", exist_ok=True)
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            train_df.to_csv("data/processed/train.csv", index=False)
            test_df.to_csv("data/processed/test.csv", index=False)
            
            mlflow.log_artifact("data/processed/train.csv", "processed_data")
            mlflow.log_artifact("data/processed/test.csv", "processed_data")
            
            print(f"✅ Training set: {len(X_train)} samples")
            print(f"✅ Test set: {len(X_test)} samples")
            
            return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, n_estimators, max_depth, random_state):
        """Train Random Forest model"""
        with mlflow.start_run(run_name="model_training", nested=True):
            mlflow.log_params({
                "algorithm": "RandomForestRegressor",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state
            })
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            
            print(f"🌲 Training Random Forest (trees={n_estimators}, depth={max_depth})...")
            model.fit(X_train, y_train)
            
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="CaliforniaHousingModel"
            )
            
            os.makedirs("models", exist_ok=True)
            model_path = f"models/rf_model_{self.timestamp}.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, "saved_models")
            
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 5 Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
                mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
            
            print(f"✅ Model trained successfully")
            return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        with mlflow.start_run(run_name="model_evaluation", nested=True):
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2
            }
            
            mlflow.log_metrics(metrics)
            
            print(f"\n📈 Evaluation Metrics:")
            print(f"   MSE:  {mse:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE:  {mae:.4f}")
            print(f"   R²:   {r2:.4f}")
            
            os.makedirs("outputs", exist_ok=True)
            
            # Predictions vs Actual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
            plt.plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
            plt.xlabel("Actual Values", fontsize=12)
            plt.ylabel("Predicted Values", fontsize=12)
            plt.title("Predictions vs Actual Values", fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            pred_plot = f"outputs/predictions_{self.timestamp}.png"
            plt.savefig(pred_plot, dpi=150)
            mlflow.log_artifact(pred_plot, "plots")
            plt.close()
            
            # Residuals plot
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel("Predicted Values", fontsize=12)
            plt.ylabel("Residuals", fontsize=12)
            plt.title("Residual Plot", fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            resid_plot = f"outputs/residuals_{self.timestamp}.png"
            plt.savefig(resid_plot, dpi=150)
            mlflow.log_artifact(resid_plot, "plots")
            plt.close()
            
            metrics_file = f"outputs/metrics_{self.timestamp}.txt"
            with open(metrics_file, "w") as f:
                f.write("=" * 50 + "\n")
                f.write("MODEL EVALUATION METRICS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Mean Squared Error (MSE):  {mse:.4f}\n")
                f.write(f"Root Mean Squared Error:   {rmse:.4f}\n")
                f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
                f.write(f"R² Score:                  {r2:.4f}\n")
                f.write("\n" + "=" * 50 + "\n")
            
            mlflow.log_artifact(metrics_file, "metrics")
            
            print(f"✅ Evaluation complete")
            return metrics


if __name__ == "__main__":
    pipeline = CaliforniaHousingPipeline("California Housing Pipeline")
    
    metrics = pipeline.run(
        n_estimators=100,
        max_depth=10,
        test_size=0.2,
        random_state=42
    )
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():20s}: {value:.4f}")
    print("=" * 70)
    print("\n💡 View detailed results at: http://localhost:5000")
