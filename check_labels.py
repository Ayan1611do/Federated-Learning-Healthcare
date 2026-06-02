import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    r"C:\Users\ayank\OneDrive\Desktop\Privacy_Preserving_Federated_Learning\Dataset\healthcare_dataset.csv"
)

encoder = LabelEncoder()

encoder.fit(df["Test Results"])

print("Label Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{i} -> {label}")