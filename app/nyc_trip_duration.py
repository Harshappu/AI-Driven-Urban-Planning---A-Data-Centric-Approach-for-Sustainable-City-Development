from flask import Blueprint, request, jsonify
import joblib
import numpy as np

nyc_trip_duration_bp = Blueprint('nyc_trip_duration', __name__)

rf_model = joblib.load('models/taxi_trip/random_forest_model.pkl')

@nyc_trip_duration_bp.route('/predict-trip-duration', methods=['POST'])
def predict_trip_duration():
    data = request.json
    features = np.array([data['pickup_longitude'], data['pickup_latitude'],
                         data['dropoff_longitude'], data['dropoff_latitude'],
                         data['passenger_count'],data['haversine_distance'],data['store_and_fwd_flag'], data['day_of_week'], data['hour']])
    features = features.reshape(1, -1)
    prediction = rf_model.predict(features)
    return jsonify({'trip_duration': prediction[0]})
