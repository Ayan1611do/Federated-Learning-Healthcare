import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=5,
    min_available_clients=5,
    min_evaluate_clients=5,
)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

import joblib

joblib.dump(global_model, "global_model.pkl")