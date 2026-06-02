import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load one client data
path = "C:\\Users\\ayank\\OneDrive\\Desktop\\Privacy_Preserving_Federated_Learning\\Dataset\\clients\\client_1.csv"
df = pd.read_csv(path)

# Separate features and target
X = df.drop("Test Results", axis=1)
y = df["Test Results"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=500)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Local Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
