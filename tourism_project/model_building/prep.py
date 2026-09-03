
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Dataset path
DATA_PATH = Path("tourism_project/data/tourism.csv")

# Load dataset directly from repository data folder
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Remove unnecessary columns
columns_to_drop = ["Unnamed: 0", "CustomerID"]
df_clean = df.drop(columns=columns_to_drop)

print("Shape after removing unnecessary columns:", df_clean.shape)

# Separate features and target
X = df_clean.drop(columns=["ProdTaken"])
y = df_clean["ProdTaken"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Save train-test splits
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("\nTrain-test split files saved successfully.")

# Verify files
for file_name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    print(f"{file_name}: {Path(file_name).exists()}")
