import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load preprocessed data
data_path = "C:\\Users\\ayank\\OneDrive\\Desktop\\Privacy_Preserving_Federated_Learning\\Dataset\\processed_healthcare_data.csv"
df = pd.read_csv(data_path)

# Create folder for clients if not exists
clients_dir = "C:\\Users\\ayank\\OneDrive\\Desktop\\Privacy_Preserving_Federated_Learning\\Dataset\\clients"
os.makedirs(clients_dir, exist_ok=True)

# Number of clients
NUM_CLIENTS = 5

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data equally
client_size = len(df) // NUM_CLIENTS

for i in range(NUM_CLIENTS):
    start = i * client_size
    end = (i + 1) * client_size
    client_data = df.iloc[start:end]

    client_data.to_csv(
        f"{clients_dir}\\client_{i+1}.csv",
        index=False
    )

    print(f"Client {i+1} data saved with {len(client_data)} rows")

print("\nClient data splitting completed successfully!")
