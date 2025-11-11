# PhishSAFE-AI

## 🚀 Project Overview  
**PhishSAFE-AI** is a machine learning based system to detect phishing websites/URLs. It’s built to help organizations and individuals alike recognise phishing threats early and take preventive action.

## 🎯 Objective  
- Build a binary classification model (Phishing vs Legitimate) using sensor data & URL features.  
- Preprocess data (cleaning, resampling), train multiple classifiers (SVM, Random Forest, XGBoost) and evaluate using metrics such as ROC-AUC, precision, recall.  
- Wrap up with a prototype that can be easily used or deployed (e.g., as a script or notebook).

## 🧪 Key Features  
- URL and webpage feature extraction: domain info, length, special characters, redirects, etc.  
- Handling of class imbalance (e.g., using SMOTE).  
- Multiple algorithms compared: SVM, Random Forest, XGBoost.  
- Clear pipeline: Data → Preprocessing → Feature Engineering → Model Training → Evaluation.

## 📂 Folder Structure  
PhishSAFE-AI/
│
├── data/
│ └── raw/
│ └── processed/
│
├── notebooks/
│ └── exploration.ipynb
│ └── model_comparison.ipynb
│
├── src/
│ └── feature_extraction.py
│ └── train_model.py
│ └── evaluate.py
│
├── models/
│ └── best_model.pkl
│
├── README.md
└── .gitignore


## 🛠 Tech Stack  
- Programming Language: Python  
- Libraries: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib/seaborn  
- Environment: Jupyter Notebook for explorations, Python scripts for end-to-end pipeline  

