import pandas as pd

df = pd.read_csv("C:\\Users\\ayank\\OneDrive\Desktop\\Privacy_Preserving_Federated_Learning\\Dataset\\healthcare_dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=[
    "Name",
    "Doctor",
    "Hospital",
    "Date of Admission",
    "Discharge Date",
    "Room Number"
])

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Separate features and target
X = df.drop("Test Results", axis=1)
y = df["Test Results"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns

print("\nCategorical columns:")
print(categorical_cols)

# Encode categorical columns
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])

# Encode target column
y = encoder.fit_transform(y)

print("\nData after encoding:")
print(X.head())
print("\nEncoded target:")
print(y[:5])


from sklearn.preprocessing import StandardScaler

# Identify numerical columns
numerical_cols = ["Age", "Billing Amount"]

# Apply normalization
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\nData after normalization:")
print(X.head())

# Combine features and target
processed_df = X.copy()
processed_df["Test Results"] = y

# Save preprocessed data
processed_df.to_csv(
    "C:\\Users\\ayank\\OneDrive\\Desktop\\Privacy_Preserving_Federated_Learning\\Dataset\\processed_healthcare_data.csv",
    index=False
)

print("\nPreprocessed data saved successfully!")
