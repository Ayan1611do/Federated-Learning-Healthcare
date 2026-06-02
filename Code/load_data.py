import pandas as pd 
df = pd.read_csv(r"C:\\Users\\ayank\\OneDrive\Desktop\\Privacy_Preserving_Federated_Learning\\Dataset\\healthcare_dataset.csv")
print(df.isnull().sum())
