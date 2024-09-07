from flask import Blueprint, request, jsonify
import joblib
import torch
import numpy as np
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1])
        return predictions

air_quality_bp = Blueprint('air_quality', __name__)

scaler = joblib.load('models/air_quality/scaler.pkl')  # This scaler was used to normalize the input features
lstm_model = LSTMModel(input_size=3, hidden_layer_size=50, output_size=1, num_layers=1)  # Input size is 3
lstm_model.load_state_dict(torch.load('models/air_quality/lstm_model.pth'))
lstm_model.eval()

@air_quality_bp.route('/predict-air-quality', methods=['POST'])
def predict_air_quality():
    data = request.json
    
    features = np.array(data['features']).reshape(1, -1)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # Add batch dimension
    
    with torch.no_grad():
        prediction = lstm_model(features).item()
    
    return jsonify({'predicted_air_quality': prediction})
