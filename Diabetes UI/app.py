from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained pipeline
with open("diabetes_pipeline.pkl", "rb") as f:
    model = pickle.load(f)
    
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        bp = float(request.form["BloodPressure"])
        skin = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        dpf = float(request.form["DiabetesPedigreeFunction"])
        age = float(request.form["Age"])

        input_data = np.array([[
            pregnancies, glucose, bp, skin,
            insulin, bmi, dpf, age
        ]])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100

        prediction = "Diabetic" if pred == 1 else "Not Diabetic"
        probability = f"{prob:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
