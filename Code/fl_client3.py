import warnings
warnings.filterwarnings("ignore")

import flwr as fl
import pandas as pd
from sklearn.linear_model import LogisticRegression

CLIENT_ID = 3   # change this for each client file

data = pd.read_csv(f"../Dataset/clients/client_{CLIENT_ID}.csv")
X = data.drop("Test Results", axis=1)
y = data["Test Results"]

model = LogisticRegression(max_iter=100)

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        if not hasattr(model, "coef_"):
            model.fit(X, y)
        return [model.coef_, model.intercept_]

    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(X, y)
        return self.get_parameters(config), len(X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = model.score(X, y)
        return float(acc), len(X), {}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)
