import pandas as pd
import os

clients_dir = "C:\\Users\\ayank\\OneDrive\\Desktop\\Privacy_Preserving_Federated_Learning\\Dataset\\clients"

client_files = os.listdir(clients_dir)

for file in client_files:
    path = os.path.join(clients_dir, file)
    df = pd.read_csv(path)

    print(f"\n{file}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
