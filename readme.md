# 🚨 AI-Powered Fraud Risk Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](#)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-green)](#)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi\&logoColor=black)](#)

## 📌 Project Overview

This project demonstrates an end-to-end fraud analytics workflow that combines machine learning, explainable AI (SHAP), and business intelligence to identify, explain, and monitor fraudulent financial transactions.

The solution transforms raw transaction data into actionable business insights through an interactive Power BI dashboard designed for fraud analysts, risk teams, and business stakeholders.

---

## 🎯 Business Objectives

* Detect potentially fraudulent transactions
* Estimate fraud probability for every transaction
* Explain model predictions using SHAP values
* Quantify revenue exposure from fraud
* Monitor fraud trends and emerging patterns
* Deliver executive-level business intelligence
  
---

## 🏗️ End-to-End Pipeline

```text
Raw Transaction Data
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Random Forest Classifier
        ↓
Fraud Probability Scoring
        ↓
SHAP Explainability
        ↓
Power BI Data Transformation
        ↓
Interactive Fraud Risk Dashboard
```

---

## 🔍 Exploratory Data Analysis

Performed comprehensive exploratory analysis including:

* Fraud class distribution analysis
* Fraud rate by transaction category
* Transaction amount distributions
* Feature importance visualization
* Confusion matrix evaluation
* Fraud behavior pattern analysis

---

## 🤖 Machine Learning Pipeline

### Feature Engineering

Additional features were engineered including:

* Transaction Hour
* Day of Week
* Customer Age
* Time-based transaction behavior features

### Model

**Random Forest Classifier**

Pipeline Components:

* Missing Value Imputation
* Categorical Encoding
* Feature Preprocessing
* Probability Estimation

### Performance

| Metric            | Score |
| ----------------- | ----- |
| ROC-AUC           | 0.984 |
| Average Precision | 0.855 |

The model achieved strong predictive performance despite the highly imbalanced nature of fraud detection data.

---

## 🧠 Explainable AI (SHAP)

To improve model transparency and trustworthiness:

* Generated SHAP values for every prediction
* Extracted the Top 3 contributing features per transaction
* Integrated explainability outputs into Power BI visualizations
* Enabled business users to understand why transactions are flagged

### Common Fraud Drivers

* Transaction Amount
* Transaction Hour
* Merchant Category
* Customer Age
* Population Characteristics

---

## 📈 Power BI Dashboard

The Power BI dashboard transforms machine learning outputs into executive-facing insights.

### Executive KPIs

* Fraud Rate
* Average Fraud Probability
* Total Revenue at Risk

### Risk Monitoring

* Revenue at Risk Over Time
* Geographic Distribution of Fraud Risk
* Revenue at Risk by Category

### Explainability

* SHAP Feature Importance Analysis
* Top Fraud Prediction Drivers

### Customer Risk Profiling

* Highest Risk Occupations
* Category-Level Fraud Exposure

---

## ⚙️ Data Engineering

To optimize dashboard performance:

* Fraud scoring performed in Python
* SHAP computations generated upstream
* Revenue-at-risk calculations precomputed
* Explainability data reshaped for Power BI
* Heavy processing removed from Power BI layer

This architecture follows production-oriented analytics engineering practices by separating computation from visualization.

---

## 📂 Repository Structure

```text
fraud-risk-analytics-dashboard/
│
├── notebooks/
│   └── code.ipynb
│
├── src/
│   ├── generate_explainability_dataset.py
│   └── create_powerbi_ready_dataset.py
│
├── dashboard/
│   └── Fraud_Risk_Dashboard.pbix
│
├── images/
│   └── dashboard.png
│
├── outputs/
│   ├── fraudTest_explainability_10000.csv
│   └── PowerBI_Ready_FraudData.csv
│
└── README.md
```

---

## 🛠️ Technologies Used

### Data Science & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* SHAP

### Visualization

* Matplotlib
* Seaborn
* Power BI

### Analytics Engineering

* Power Query
* Data Modeling
* Explainable AI Pipelines

---

## 🚀 Key Outcomes

* Built an end-to-end fraud detection pipeline
* Achieved ROC-AUC of 0.984
* Integrated Explainable AI into business reporting
* Developed an interactive Power BI fraud analytics dashboard
* Connected machine learning outputs directly to business decision-making
* Implemented production-style analytics engineering practices

---

## 👨‍💻 Author

**Manish Patil**


