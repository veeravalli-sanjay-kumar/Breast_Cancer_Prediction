from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Get data from HTML form
    age = request.form["age"]
    menopause = request.form["menopause"]
    tumor_size = request.form["tumor-size"]
    inv_nodes = request.form["inv-nodes"]
    node_caps = request.form["node-caps"]
    deg_malig = request.form["deg-malig"]
    breast = request.form["breast"]
    breast_quad = request.form["breast-quad"]
    irradiat = request.form["irradiat"]

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "menopause": [menopause],
        "tumor-size": [tumor_size],
        "inv-nodes": [inv_nodes],
        "node-caps": [node_caps],
        "deg-malig": [deg_malig],
        "breast": [breast],
        "breast-quad": [breast_quad],
        "irradiat": [irradiat]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    confidence = round(
        max(probabilities) * 100,
        2
    )

    # Go to separate result page
    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)