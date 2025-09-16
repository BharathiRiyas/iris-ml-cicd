from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model/iris_classifier_model.pkl')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = data.get("features")
    if not features or len(features) != 4:
        return jsonify({"error": "Please provide 4 numeric features"}), 400
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    classes = ['setosa', 'versicolor', 'virginica']
    return jsonify({"class": classes[prediction]})

if __name__ == "__main__":
    app.run(debug=True)
