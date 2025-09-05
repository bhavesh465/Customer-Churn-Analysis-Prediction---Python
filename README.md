
Customer Churn Analysis & Prediction

A complete Python-based machine learning project that explores and predicts customer churn for a telco service provider. Using the popular Telco Customer Churn dataset, this project illustrates a full data science pipeline—from data ingestion and preprocessing to model training and evaluation using a Random Forest classifier.

Table of Contents

Project Overview

Features

Dataset

Prerequisites

Project Structure

Getting Started

Methodology

Results

Next Steps

References

Project Overview

This project aims to predict customer churn—whether a customer will discontinue using the service—based on their behavior and demographics. Churn analysis helps businesses understand customer attrition and develop strategies to improve retention. 
GeeksforGeeks

Features

Data loading and initial exploration

Handling missing or invalid values (TotalCharges conversion & median imputation) 

Encoding categorical features via LabelEncoder 

Splitting features (X) and target (y), excluding irrelevant fields like customerID 

Feature scaling with StandardScaler for model optimization 

Model training using RandomForestClassifier 


Evaluation with accuracy score and confusion matrix visualization 

Dataset

Name: Telco Customer Churn

Contents: Customer demographics, service plans, usage history, payment methods, and churn outcome (Yes/No) 

Key Columns:

tenure, InternetService, PaymentMethod, TotalCharges, Churn, and other categorical customer attributes. 
GeeksforGeeks

Prerequisites

Make sure you have the following Python packages installed:

pandas
numpy
scikit-learn
seaborn
matplotlib


You can install them via:

pip install pandas numpy scikit-learn seaborn matplotlib

Project Structure
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── README.md
└── requirements.txt


data/: Raw dataset

notebooks/: Jupyter Notebook with analysis and modeling

requirements.txt: List of dependencies

README.md: This project description

Getting Started

Clone or download this repository.

Place the Telco-Customer-Churn.csv file in the data/ directory.

Open the notebook:

cd notebooks
jupyter notebook churn_analysis.ipynb


Run each cell sequentially for full walkthrough—from EDA to model evaluation.

Methodology

Load & Explore: Inspect dataset structure, missing values, and distribution of churn vs. non-churn. 


Preprocess:

Convert TotalCharges to numeric and impute missing values with median. 


Encode categorical variables using LabelEncoder. 


Split data into X (features) and y (target), and train/test sets (80/20 split). 

Scale Features: Standardize feature values (mean = 0, std = 1) using StandardScaler. 

Train Model: Use RandomForestClassifier to train on preprocessed data. 

Evaluate:

Calculate accuracy score (approximately 0.78) 

Display confusion matrix and interpret model’s performance. 

Results

Model Accuracy: ~0.78 


Confusion Matrix Insights:

Correctly predicted 924 non-churn customers and 181 churners.

117 non-churners were misclassified as churners, and 187 churners were missed.

Suggests need for improved recall on churn cases.
