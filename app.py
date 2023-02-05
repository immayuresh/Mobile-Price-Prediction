import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
model = pickle.load(open("knnmodel.pkl", "rb"))
flask_app = Flask(__name__,template_folder="template",static_folder='staticFiles')

@flask_app.route("/")
def home():
    return render_template("index.html")
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Predicted Mobile Price Range {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0",port=8080)