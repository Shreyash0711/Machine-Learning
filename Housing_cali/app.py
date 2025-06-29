# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained RandomForestRegressor model using pickle
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the affordability function based on median house value
def determine_affordability(price):
    if price <= 1.75:
        return "Affordable"
    elif 1.75 < price <= 3.00:
        return "Mid-range"
    else:
        return "Expensive"

# Home route that renders the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions and showing the result
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        features = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup'])
        ]
        
        # Predict using the model
        prediction = model.predict([features])
        price = round(prediction[0], 2)


        # Determine affordability
        affordability = determine_affordability(price)

        return render_template('index.html', prediction=price, affordability=affordability)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
