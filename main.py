import streamlit as st
import pandas as pd
import numpy as np
import joblib
from artifacts_helper import predict

# Set page config
st.set_page_config(
    page_title="Credit Risk Loan Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c5aa0;
        border-bottom: 2px solid #e6f3ff;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .risk-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .medium-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    /* New styling for metric boxes */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">🏦 Lauki Finance: Credit Risk Modeling</div>', unsafe_allow_html=True)

    # Sidebar for model information
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.info(
            "This ML model evaluates credit risk based on a comprehensive set of personal, financial, and loan characteristics.")

        st.markdown("### 🎯 Credit Rating")
        st.markdown("""
        - **Excellent (750-900)**: ✅ Very low risk
        - **Good (650-749)**: 👍 Low risk
        - **Average (500-649)**: ⚠️ Moderate risk
        - **Poor (300-499)**: ❌ High risk
        """)

        # Model performance metrics
        st.markdown("### 📈 Model Performance")
        st.markdown('<br>', unsafe_allow_html=True)  # Adds a little space

        # Use columns to lay out the metrics in a grid
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col4, metric_col5, _ = st.columns(3)  # Use an empty column to balance the layout

        with metric_col1:
            st.metric(label="Accuracy", value="93%", delta="High", delta_color="normal")

        with metric_col2:
            st.metric(label="Precision", value="56%", delta="Moderate", delta_color="off")

        with metric_col3:
            st.metric(label="Recall", value="94%", delta="High", delta_color="normal")

        with metric_col4:
            st.metric(label="AUC", value="98%", delta="Excellent", delta_color="normal")

        with metric_col5:
            st.metric(label="Gini Coef", value="96%", delta="Excellent", delta_color="normal")

        # Optional: Add a brief explanation of the metrics
        st.markdown("""
        <br>
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h5 style="color: #2c5aa0;">Understanding the Metrics:</h5>
            <ul style="list-style-type: none; padding-left: 0;">
                <li><b>Accuracy:</b> The percentage of correct predictions. A high value indicates the model is generally right.</li>
                <li><b>Precision:</b> Of all positive predictions, how many were correct? Important for minimizing false positives.</li>
                <li><b>Recall:</b> Of all actual positives, how many did the model predict correctly? Crucial for finding all high-risk cases.</li>
                <li><b>AUC:</b> Area Under the Curve. Measures the model's ability to distinguish between positive and negative classes. Higher is better.</li>
                <li><b>Gini Coefficient:</b> A measure of the inequality or discriminatory power of the model. A Gini of 100% is a perfect model.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    # Create two columns for input layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Personal Information Section
        st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)

        personal_col1, personal_col2 = st.columns(2)

        with personal_col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35,
                help="Applicant's age in years"
            )

        with personal_col2:
            residence_type = st.selectbox(
                "Residence Type",
                ["Owned", "Rented", "Other"],
                help="Type of residence"
            )

        # Financial Information Section
        st.markdown('<div class="section-header">💰 Financial Information</div>', unsafe_allow_html=True)

        fin_col1, fin_col2 = st.columns(2)

        with fin_col1:
            income = st.number_input(
                "Monthly Income ($)",
                min_value=0,
                value=5000,
                step=100,
                help="Applicant's total monthly income"
            )

            number_of_open_accounts = st.number_input(
                "Number of Open Accounts",
                min_value=0,
                max_value=50,
                value=5,
                help="Total number of active credit accounts"
            )

            credit_utilization_ratio = st.slider(
                "Credit Utilization Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                format="%.1f",
                help="Ratio of credit used to total credit available (in %)"
            )

        with fin_col2:
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=0,
                value=20000,
                step=1000,
                help="The requested loan amount"
            )

            delinquency_ratio = st.slider(
                "Delinquency Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                format="%.1f",
                help="Ratio of delinquent payments to total payments (in %)"
            )

            avg_dpd_per_delinquency = st.number_input(
                "Average Days Past Due per Delinquency",
                min_value=0.0,
                max_value=365.0,
                value=30.0,
                step=1.0,
                help="Average number of days past due for each delinquent payment"
            )

        # Loan Information Section
        st.markdown('<div class="section-header">📋 Loan Information</div>', unsafe_allow_html=True)

        loan_col1, loan_col2 = st.columns(2)

        with loan_col1:
            loan_tenure_months = st.number_input(
                "Loan Tenure (Months)",
                min_value=1,
                max_value=360,
                value=36,
                help="Duration of the loan in months"
            )

            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["Personal", "Home", "Education", "Other"],
                help="Primary purpose of the loan"
            )

        with loan_col2:
            loan_type = st.selectbox(
                "Loan Type",
                ["Secured", "Unsecured"],
                help="Whether the loan is secured by collateral"
            )

        # Predict button
        if st.button("🔍 Calculate Credit Score", type="primary", use_container_width=True):
            # Prepare input data and get prediction
            try:
                probability, credit_score, rating = predict(
                        age, income, loan_amount, number_of_open_accounts, credit_utilization_ratio,
                        loan_tenure_months, delinquency_ratio, avg_dpd_per_delinquency,
                        residence_type, loan_purpose, loan_type
                )

                # Store results in session state
                st.session_state.probability = probability
                st.session_state.credit_score = credit_score
                st.session_state.rating = rating

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    with col2:
        st.markdown('<div class="section-header">🎯 Risk Assessment</div>', unsafe_allow_html=True)

        if hasattr(st.session_state, 'credit_score'):
            credit_score = st.session_state.credit_score
            rating = st.session_state.rating

            # Determine risk category based on rating
            if rating == "Excellent" or rating == "Good":
                risk_class = "low-risk"
                risk_emoji = "✅"
            elif rating == "Average":
                risk_class = "medium-risk"
                risk_emoji = "⚠️"
            else:  # Poor or Undefined
                risk_class = "high-risk"
                risk_emoji = "❌"

            # Display risk score
            st.markdown(f'''
            <div class="risk-score {risk_class}">
                {risk_emoji}<br>
                {credit_score}/900<br>
                <small>Rating: {rating}</small>
            </div>
            ''', unsafe_allow_html=True)

            # Risk gauge visualization
            st.markdown("### 📊 Credit Score Gauge")
            st.metric("Probability of Default", f"{st.session_state.probability * 100:.2f}%")

            score_ranges = {
                'Poor': [300, 500],
                'Average': [500, 650],
                'Good': [650, 750],
                'Excellent': [750, 900]
            }

            # Progress bar for visual representation of score
            score_normalized = (credit_score - 300) / 600
            st.progress(score_normalized)

            # Additional insights based on rating
            st.markdown("### 💡 Key Insights")
            if rating == "Excellent":
                st.success("✅ Excellent credit profile - High probability of loan approval.")
            elif rating == "Good":
                st.info("ℹ️ Good credit profile - Loan is likely to be approved with favorable terms.")
            elif rating == "Average":
                st.warning("⚠️ Average credit profile - May require additional documentation or higher interest rates.")
            else:  # Poor
                st.error("❌ Poor credit profile - High default risk. Loan application may be denied.")

        else:
            st.info("👆 Enter applicant details and click 'Calculate Credit Score' to see the assessment")

            # Placeholder visualization
            st.markdown("""
            <div class="info-box">
                <h4>🔍 What we analyze:</h4>
                <ul>
                    <li>Personal demographics</li>
                    <li>Credit history & utilization</li>
                    <li>Financial ratios</li>
                    <li>Loan characteristics</li>
                    <li>Payment behavior</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()