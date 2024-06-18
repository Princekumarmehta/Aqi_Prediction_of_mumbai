import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def classify_risk(pm25):
    if pm25 <= 12:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_advice_for_heart_patients(risk_category):
    advice = {
        "Good": "Air quality is good. No specific precautions needed.",
        "Moderate": "Air quality is acceptable. Heart patients should consider limiting prolonged exertion.",
        "Unhealthy for Sensitive Groups": "Members of sensitive groups, including heart patients, may experience health effects. It's advisable to limit outdoor exertion.",
        "Unhealthy": "Everyone may begin to experience health effects. Heart patients should avoid outdoor exertion.",
        "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Heart patients should stay indoors and limit physical activity.",
        "Hazardous": "Health warning of emergency conditions. The entire population is likely to be affected. Heart patients should stay indoors and avoid physical activities."
    }
    return advice[risk_category]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    pm25 = round(prediction[0], 2)
    risk_category = classify_risk(pm25)
    advice = get_advice_for_heart_patients(risk_category)

    return render_template('index.html', pm25=pm25, risk_category=risk_category, advice=advice)

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    pm25 = prediction[0]
    risk_category = classify_risk(pm25)
    advice = get_advice_for_heart_patients(risk_category)

    return jsonify({'pm25': pm25, 'risk_category': risk_category, 'advice': advice})

if __name__ == "__main__":
    app.run(debug=True)
