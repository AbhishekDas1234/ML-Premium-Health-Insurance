# ML-Health-Insurance-Premium-Predictor

This project is a health insurance premium prediction system where the user needs to enter some informations like age, gender, income in lakhs etc as input, and the model provides an insurance premium estimate.
It consists of a frontend Streamlit server and backend python code containing the ML models (XGBoost (for age>25 years) and LinearRegression (for age<=25 years)).

## Project Structure
- **main.py/**: Contains Streamlit application code
- **prediction_helper.py/**: Contains backend  ML python code
- **requirements.txt**: Contains the required Python packages
- **README.md**: Contains an overview of the project

## Entrypoint:
- **main.py/**: A Streamlit UI that collects user inputs (age, dependants, income, genetical risk, insurance plan, employment, gender, marital status, BMI category, smoking status, region, medical history). When the user clicks "Predict" it calls predict method and displays:
Prediction Premium (integer)

## Prediction logic:
- **prediction_helper.py/**: Loads two joblib models and scalers from artifacts/, encodes inputs (one-hot features), converts medical-history strings into a normalized risk score, applies age-based scaling (young vs rest), and uses the appropriate model to produce a premium prediction.

## Inputs/outputs and artifact:
- **Inputs**: numeric features (age, income, loan_amount, loan_to_income_ratio, loan_tenure_months, avg_dpd_per_delinquency, delinquency_ratio, credit_utilization_ratio, num_open_accounts) and categoricals (residence_type, loan_purpose, loan_type).
- **Ouput**: Prediction Premium (integer).
- **Model artifact**: artifacts/model_rest.joblib, artifacts/model_young.joblib, artifacts/scaler_rest.joblib, artifacts/scaler_young.joblib (required to run).
- **Dependencies**: streamlit, pandas, joblib, scikit-learn.

## Setup Instructions
1. **Clone the repository**:
    ```bash
   git clone 'https://github.com/AbhishekDas1234/ML-Premium-Health-Insurance.git'
   cd ML-Premium-Health-Insurance
    ```
2. **Install dependencies:**:
    ```commandline
    pip install -r .\requirements.txt
    ```
3. **Run the Streamlit server:**:
    ```commandline
   streamlit run .\main.py
    ```
