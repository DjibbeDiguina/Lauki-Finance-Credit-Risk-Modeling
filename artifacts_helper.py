import joblib
import numpy as np
import pandas as pd

model_data = joblib.load("artifacts/model_data.joblib")

model = model_data['model']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']
scaler = model_data['scaler']


def prepare_input_data(age, income, loan_amount, number_of_open_accounts, credit_utilization_ratio,
                       loan_tenure_months, delinquency_ratio, avg_dpd_per_delinquency, residence_type,
                       loan_purpose, loan_type):
    #Prepare input data in the format expected by the ML model

    # Initialize a complete dictionary with dummy values for all features the model expects
    input_data = {
        'age': age,
        'number_of_open_accounts': number_of_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio / 100,  # convert % to ratio
        'loan_tenure_months': loan_tenure_months,
        'loan_to_income': loan_amount / income if income > 0 else 0.0,
        'delinquency_ratio': delinquency_ratio / 100,  # convert % to ratio
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'disbursal_date': 0, 'installment_start_dt': 0, 'total_loan_months': 0,
        'delinquent_months': 0, 'total_dpd': 0, 'sanction_amount': 0,
        'principal_outstanding': 0, 'processing_fee': 0, 'gst': 0,
        'net_disbursement': 0, 'zipcode': 0, 'number_of_closed_accounts': 0,
        'marital_status': 0, "gender": 0, "city": 0, "state": 0,
        'number_of_dependants': 0, 'years_at_current_address': 0, 'enquiry_count': 0,
        'bank_balance_at_application': 0,
    }

    df = pd.DataFrame([input_data])

    ## scale columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    ## selected_features
    df = df[features]

    return df

def predict(age, income, loan_amount, number_of_open_accounts, credit_utilization_ratio,
                       loan_tenure_months, delinquency_ratio,
                       avg_dpd_per_delinquency, residence_type, loan_purpose, loan_type):

    input_df = prepare_input_data(age, income, loan_amount, number_of_open_accounts, credit_utilization_ratio,
                                  loan_tenure_months, delinquency_ratio, avg_dpd_per_delinquency, residence_type,
                                  loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating

def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    credit_score = base_score + non_default_probability.flatten() * scale_length

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating

