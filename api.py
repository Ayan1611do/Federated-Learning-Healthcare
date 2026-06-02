from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("global_model.pkl")

@app.route("/")
def home():
    return "Healthcare Federated Learning API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    label_map = {
    0: "Abnormal",
    1: "Inconclusive",
    2: "Normal"
}

    return jsonify({
    "prediction": label_map[int(prediction[0])]
})
if __name__ == "__main__":
    app.run(debug=True)