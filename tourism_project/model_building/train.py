
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# --------------------------------------------------
# 1. Load train-test data
# --------------------------------------------------

X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv").squeeze()
y_test = pd.read_csv("ytest.csv").squeeze()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# --------------------------------------------------
# 2. Identify numerical and categorical features
# --------------------------------------------------

numerical_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# --------------------------------------------------
# 4. XGBoost model
# --------------------------------------------------

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# --------------------------------------------------
# 5. Complete machine learning pipeline
# --------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)

# --------------------------------------------------
# 6. Hyperparameter tuning
# --------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1]
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("\nStarting GridSearchCV...")
grid_search.fit(X_train, y_train)

# --------------------------------------------------
# 7. Best model and parameters
# --------------------------------------------------

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_f1 = grid_search.best_score_

print("\nBest Parameters:")
print(best_params)

print("\nBest Cross-Validation F1 Score:")
print(best_cv_f1)

# --------------------------------------------------
# 8. Test-set evaluation
# --------------------------------------------------

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nTest Set Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 9. MLflow Experiment Tracking
# --------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Visit_With_Us_Tourism_Prediction")

with mlflow.start_run(run_name="Best_XGBoost_Model"):

    mlflow.log_params({
        "n_estimators": best_params["model__n_estimators"],
        "max_depth": best_params["model__max_depth"],
        "learning_rate": best_params["model__learning_rate"]
    })

    mlflow.log_metrics({
        "cv_f1": best_cv_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model"
    )

print("\nMLflow tracking completed successfully.")

# --------------------------------------------------
# 10. Save complete trained pipeline
# --------------------------------------------------

model_path = Path("tourism_project/deployment/best_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_path)

print(f"\nBest model saved to: {model_path}")
print(f"Model file exists: {model_path.exists()}")
