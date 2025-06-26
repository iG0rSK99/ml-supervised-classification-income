# ml-supervised-classification-income

🧠 Income Classification Using Machine Learning.

This project aims to predict whether an individual earns more than $50K/year using demographic and employment-related data. Multiple classification algorithms were trained and evaluated, and the best model was selected based on cross-validation and performance metrics.

📌 Problem Statement:
Develop a machine learning model that classifies individuals as earning either <=50K or >50K annually based on features like education, occupation, age, hours worked per week, capital gain, etc.
This is a binary classification task based on the UCI Adult Income Dataset, often referred to as the "Census Income" dataset.

⚙️ Project Workflow:

Data Collection & Loading
Handling Missing Values (? replaced with NaN, then imputed or dropped)
Encoding Categorical Variables
Ordinal Encoding for education
Label Encoding for other features
Exploratory Data Analysis (EDA)
Histograms, Countplots, Heatmaps
Train-Test Split (80-20)
Model Training & Comparison
Cross-Validation (Accuracy)
Hyperparameter Tuning (GridSearchCV)
Model Evaluation
Confusion Matrix
Classification Report
F1 Score
Feature Importance

Models Compared:
✅ Random Forest	85.6% (Selected)
✅ Final Model Selection

After evaluating all models, Random Forest was selected based on its:

High F1-score
Strong recall for the minority class (>50K)
Balanced performance
Resistance to overfitting
Good interpretability via feature importance

🎯 Best Hyperparameters:
RandomForest{'n_estimators': 200, 'min_samples_split': 5, 'class_weight': 'balanced'}


🔍 Key Insights:
The dataset was slightly imbalanced. Using class_weight='balanced' helped improve recall for the minority class.
The most important features affecting income were:
education
capital-gain
hours-per-week
occupation
marital-status

Dataset used: adultKNN.csv

🛠️ Technologies Used:
Python
Jupyter Notebook
Scikit-learn
XGBoost
Pandas, NumPy
Matplotlib, Seaborn

# ---------------------------------------------- FINALIZING MODEL ----------------------------------------------

# After training and evaluating six different supervised classification algorithms — Logistic Regression, K-Nearest Neighbors, Decision Tree, Support Vector Machine, XGBoost, and Random Forest
# we used cross-validation to assess their performance based on accuracy and selected the top two: Random Forest and XGBoost.

# We then fine-tuned both models using GridSearchCV, optimizing key hyperparameters such as n_estimators, max_depth, and min_samples_split. 
# The goal was to improve the models ability to generalize while reducing potential overfitting.

# Final Model Comparison:
# Metric	         Random Forest	  XGBoost
# Accuracy	            85%	         84%
# Precision (>50K)    	0.69	       0.71
# Recall (>50K)	        0.70	       0.62
# F1-score (>50K)	      0.69	       0.66

# While XGBoost had slightly better precision, Random Forest outperformed it on recall and F1-score, making it more suitable for identifying higher-income individuals accurately 
# especially important in use cases where missing high-income earners is costlier than occasional false positives.

# Additionally, Random Forest showed more balanced performance across both classes, and learning curves confirmed no major signs of overfitting.

# Selected Final Model: Random Forest (with GridSearch tuning)
# Best Params: n_estimators=200, min_samples_split=5, class_weight='balanced'

# Test Accuracy: 85%

# Final Decision: Chosen for higher F1-score, better recall, and robust generalization.

