from flask import Flask, request, render_template
import joblib
import numpy as np
from collections import Counter 
from sklearn.preprocessing import StandardScaler
import pandas as pd


app = Flask(__name__)

  # Load the trained model and scaler
model_names = joblib.load('ensemble_models.pkl')
models = {name: joblib.load(f'{name}.pkl') for name in model_names}
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
      return render_template('start.html')

feature_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 
                 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']

@app.route('/predict', methods=['POST'])
def predict():
      features = [float(x) for x in request.form.values()]
      final_df = pd.DataFrame([features], columns=feature_names)
      
      scaled_features = scaler.transform(final_df)

    # Collect predictions from all models
      votes = []
      for model in models.values():
          pred = model.predict(scaled_features)[0]
          votes.append(pred)

    # Majority vote
      result = Counter(votes).most_common(1)[0][0]

      if result == 0:
          output = "No risk of cardiac arrest ❤️"
      else:
          output = "High risk of cardiac arrest ⚠️"
      
      
      
      return render_template('result.html', prediction_text=output)

if __name__ == '__main__':
      app.run(debug=True)