import pickle
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
with open('logistic_model1.pkl', 'rb') as f:
    model = pickle.load(f)

# Encodings for categorical variables (use these for dropdown lists)
encodings = {
    'person_gender': {'male': 1, 'female': 0},
    'person_education': {'Bachelor': 0, 'Associate': 1, 'High School': 2, 'Master': 3, 'Doctorate': 4},
    'person_home_ownership': {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3},
    'loan_intent': {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2, 'PERSONAL': 3, 'DEBTCONSOLIDATION': 4, 'HOMEIMPROVEMENT': 5},
    'previous_loan_defaults_on_file': {'No': 0, 'Yes': 1}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Collect form data from the web form and encode categorical variables
        data = {
            'person_age': float(request.form['person_age']),
            'person_gender': encodings['person_gender'][request.form['person_gender']],
            'person_education': encodings['person_education'][request.form['person_education']],
            'person_income': float(request.form['person_income']),
            'person_emp_exp': float(request.form['person_emp_exp']),
            'person_home_ownership': encodings['person_home_ownership'][request.form['person_home_ownership']],
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_intent': encodings['loan_intent'][request.form['loan_intent']],
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_cred_hist_length': float(request.form['cb_person_cred_hist_length']),
            'credit_score': float(request.form['credit_score']),
            'previous_loan_defaults_on_file': encodings['previous_loan_defaults_on_file'][request.form['previous_loan_defaults_on_file']]
        }

        input_df = pd.DataFrame([data])

        # Debugging: Print the type of the loaded model
        print(f"Model Type: {type(model)}")

        # Check if the model has the 'predict' method
        if hasattr(model, 'predict'):
            prediction = model.predict(input_df)[0]  # Get the first prediction from the array
        else:
            print("Error: Model does not have a predict method!")

    return render_template('index.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)
