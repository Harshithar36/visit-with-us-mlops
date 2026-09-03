
import pandas as pd
from pathlib import Path

# Path to the registered dataset
DATA_PATH = Path("tourism_project/data/tourism.csv")

# Expected columns in the tourism dataset
EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# Check whether dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Validate columns
missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

if extra_columns:
    print(f"Additional columns found: {extra_columns}")

# Dataset validation summary
print("Dataset registration successful.")
print(f"Dataset path: {DATA_PATH}")
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset summary:")
print(df.describe(include="all").transpose())
