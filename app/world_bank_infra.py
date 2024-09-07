from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import pandas as pd

world_bank_infra_bp = Blueprint('world_bank_infra', __name__)

gb_model = joblib.load('models/infrastructure/gradient_boosting_regression_model1.pkl')
preprocessor = joblib.load('models/infrastructure/preprocessor.pkl')
scaler = joblib.load('models/infrastructure/scaler.pkl')

@world_bank_infra_bp.route('/predict-infrastructure', methods=['POST'])
def predict_infrastructure():
    data = request.json
    
    features = np.array([data['lendprojectcost'], data['ibrdcommamt'], 
                         data['idacommamt'], data['grantamt']])
    
    input_df = pd.DataFrame([{
        'lendprojectcost': data['lendprojectcost'],
        'ibrdcommamt': data['ibrdcommamt'],
        'idacommamt': data['idacommamt'],
        'grantamt': data['grantamt'],
        # Add categorical features as well (you should pass these in the JSON body)
        'regionname': data['regionname'],
        'lendinginstr': data.get('lendinginstr', 'Unknown'),
        'lendinginstrtype': data.get('lendinginstrtype', 'Unknown'),
        'envassesmentcategorycode': data.get('envassesmentcategorycode', 'Unknown'),
        'supplementprojectflg': data.get('supplementprojectflg', 'Unknown'),
        'productlinetype': data.get('productlinetype', 'Unknown'),
        'projectstatusdisplay': data.get('projectstatusdisplay', 'Unknown'),
        'status': data.get('status', 'Unknown'),
        'borrower': data.get('borrower', 'Unknown'),
        'impagency': data.get('impagency', 'Unknown'),
        'sector': data.get('sector', 'Unknown'),
        'mjsector': data.get('mjsector', 'Unknown'),
        'goal': data.get('goal', 'Unknown'),
        'financier': data.get('financier', 'Unknown'),
        'location': data.get('location', 'Unknown')
    }])

    processed_features = preprocessor.transform(input_df)

    processed_features = scaler.transform(processed_features)

    prediction = gb_model.predict(processed_features)
    
    return jsonify({'predicted_infrastructure_growth': prediction[0]})
