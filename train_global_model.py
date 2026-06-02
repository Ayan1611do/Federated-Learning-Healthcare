import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

path = r"C:\Users\ayank\OneDrive\Desktop\Privacy_Preserving_Federated_Learning\Dataset\processed_healthcare_data.csv"

df = pd.read_csv(path)

X = df.drop("Test Results", axis=1)
y = df["Test Results"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=500)

model.fit(X_train, y_train)

joblib.dump(model, "global_model.pkl")

print("Model Saved Successfully")