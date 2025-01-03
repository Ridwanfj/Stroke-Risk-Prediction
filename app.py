from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/learn")
def learn():
    return render_template("learn.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result", methods=['POST', 'GET'])
def result():
    try:
       
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

    
        x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                      avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

        
        base_dir = os.path.dirname(__file__) 
        scaler_path = os.path.join(base_dir, 'models', 'scaler2.pkl')
        model_path = os.path.join(base_dir, 'models', 'rf2.sav')

       
        print("Scaler Path:", scaler_path)
        print("Model Path:", model_path)
        print("Apakah scaler file ada?:", os.path.exists(scaler_path))
        print("Apakah model file ada?:", os.path.exists(model_path))

        # Periksa keberadaan file
        if not os.path.exists(scaler_path):
            return "Scaler file not found. Please check the scaler path.", 500
        if not os.path.exists(model_path):
            return "Model file not found. Please check the model path.", 500

        
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        rf = joblib.load(model_path)

       
        x = scaler.transform(x)
        Y_pred = rf.predict(x)

        
        if Y_pred == 0:
            return render_template('nostroke.html')
        else:
            return render_template('stroke.html')

    except Exception as e:
        # Log error jika terjadi
        print("Error:", e)
        return "An error occurred: {}".format(e), 500

if __name__ == "__main__":
    app.run(debug=True, port=7384)