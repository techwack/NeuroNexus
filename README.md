Credit Card Fraud Detection | NeuroNexus Internship
📌 Overview
This project tackles the imbalanced classification problem of detecting fraudulent credit card transactions. The focus was on maximizing recall to catch fraud while maintaining strong precision to reduce false positives.

🛠️ Technical Summary
Dataset: Kaggle – Credit Card Fraud Detection (284,807 transactions, 0.172% fraud)
Preprocessing:

Handled class imbalance using SMOTE

Scaled Amount using StandardScaler

Dropped Time due to low relevance

Models Evaluated:

Logistic Regression (baseline)

Random Forest (with GridSearchCV)

LightGBM (GPU-accelerated)

Performance Comparison:

Model	AUC-ROC	F2-Score	Recall
Logistic Regression	0.98	0.85	0.88
Random Forest	0.99	0.89	0.90
LightGBM (GPU)	0.99	0.92	0.92

📂 Project Structure
CreditCard_Fraud_Detection.ipynb – Main notebook

creditcard.csv – Dataset (excluded via .gitignore)

images/ – Contains ROC curve, feature importance plots

README.md – Project documentation

🔍 Key Findings
LightGBM was the top performer and 4x faster on GPU

Most predictive features: V14, V4, V12 (PCA components)

Precision-Recall curves illustrated trade-offs in fraud detection

🚀 Quickstart
Install dependencies:
pip install pandas numpy scikit-learn imbalanced-learn lightgbm matplotlib seaborn

Download the dataset from Kaggle and place it in the repo

Run the Jupyter notebook: CreditCard_Fraud_Detection.ipynb

🙌 Acknowledgments
Task by NeuroNexus Innovations

SMOTE via imbalanced-learn
