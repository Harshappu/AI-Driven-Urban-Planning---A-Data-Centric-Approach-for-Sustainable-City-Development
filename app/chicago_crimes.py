from flask import Blueprint, request, jsonify
import joblib
import numpy as np

chicago_crimes_bp = Blueprint('chicago_crimes', __name__)

rf_classifier = joblib.load('models/crimes/random_forest_classifier.pkl')

@chicago_crimes_bp.route('/predict-crime', methods=['POST'])
def predict_crime():
    data = request.json
    features = np.array([data['hour'], data['day_of_week'],
                         data['ARREST'], data['DOMESTIC'],
                         data['X_COORDINATE'], data['Y_COORDINATE']])
    features = features.reshape(1, -1)
    prediction = rf_classifier.predict(features)
    return jsonify({'crime_occurrence': bool(prediction[0])})
