# 🏦 Lauki Finance: Credit Risk Modeling


**An AI-driven solution for the FinTech industry, providing automated credit scoring, risk analysis, and actionable insights using a machine learning model built with Streamlit and Python.**

---

## ✨ Features

### 🎯 Core Functionality
- **Interactive Dashboard** with a modern UI powered by **Streamlit**.
- **Dual Scoring System**:
    - **Credit Score** (300–900 scale) with a clear rating classification.
    - **Risk Score** (0–100 scale) for traditional risk assessment.
- **Real-time Metrics**: Dynamically calculates and displays key financial ratios like **Loan-to-Income** and **Delinquency ratios**.
- **ML-Powered Predictions**: Utilizes a robust **Logistic Regression** model for accurate credit scoring.

### 📊 Advanced Analytics
- **Credit Rating**: Classifies applicants into **Poor, Average, Good, or Excellent** categories.
- **Default Probability**: Provides the likelihood of default based on a logistic probability calculation.
- **Key Insights Engine**: Generates automated risk analysis and data-driven recommendations.
- **Model Transparency**: Showcases crucial performance metrics including **Accuracy, Precision, Recall, AUC, and Gini Coefficient**.

### 🎨 Modern UI/UX
- **Glassmorphism-inspired Design** for a clean, modern aesthetic.
- **Responsive Layout** ensuring seamless access on all screen sizes.
- **Color-coded Results** for intuitive and quick interpretation of risk levels.

---

## 📂 Project Structure

```plaintext
credit-risk-scoring/ 
├── main.py                # Main Streamlit application
├── artifacts_helper.py    # ML model integration and prediction logic
├── artifacts/
│   └── model_data.joblib  # Trained ML model and preprocessing objects
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
## 🚀 Quick Start
**Prerequisites**
- Python 3.8+

- pip package manager

- 4GB+ RAM recommended

## Installation

**Clone the repository:**
```plaintext
Bash
git clone [https://github.com/yourusername/credit-risk-scoring.git](https://github.com/yourusername/credit-risk-scoring.git)
cd credit-risk-scoring
Create and activate a virtual environment:
```
```plaintext
Bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

**Install the project dependencies:**
```plaintext
Bash

pip install -r requirements.txt
```
**The application will open in your default web browser at http://localhost:8501**

## 📊 Model Performance

**Metric Value :**
```plaintext
Accuracy	93%
Precision	56%
Recall	        94%
AUC Score	98%
Gini Coef	96%
```
## 📦 Dependencies
```plaintext
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
plotly>=5.0.0
```

## 👤 Author:
**Djibe Christian Diguina**

*📍 Kigali, Rwanda*

*I’m a Software Engineer & Data Scientist passionate about building AI-driven solutions for FinTech & AgriTech. Open to remote opportunities, collaborations, and innovative projects.*

- LinkedIn: https://www.linkedin.com/in/djibbediguina/

- Email: diguinafils1@gmail.com

## 🤝 Contribution
**Contributions are welcome! Please fork the repository, create a new branch for your changes, and submit a pull request. We especially appreciate suggestions for UI/UX improvements, model optimization, or new features.**

## 📄 License

This project is licensed under the MIT License – feel free to use, modify, and adapt it.
