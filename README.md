🧠 Income Classification Using Machine Learning

This project predicts whether an individual earns more than $50K/year using demographic and employment-related features. Multiple classification algorithms were trained and evaluated, and the best model was selected using cross-validation and performance metrics.

📌 Problem Statement : 
Develop a machine learning model that classifies individuals as earning either <=50K or >50K annually.


⚙️ Project Workflow : 
Data Collection & Loading
Handling Missing Values (? → NaN, then imputed/dropped)
Categorical Encoding
Ordinal Encoding (education)
Label Encoding (other categorical features)
Exploratory Data Analysis (EDA)
Histograms, Countplots, Heatmaps
Train-Test Split (80/20)
Model Training & Comparison (6 classifiers)
Cross-Validation (Accuracy)
Hyperparameter Tuning (GridSearchCV)
Model Evaluation
Confusion Matrix
Classification Report
F1 Score
Feature Importance

🧪 Models Compared :

✅ Random Forest	85.6% (Selected)
✅ Final Model Selection

After evaluating all six models, we selected Random Forest based on:

High F1-score
Strong recall for the minority class (>50K)
Balanced performance across both classes
Resistance to overfitting
High interpretability via feature importance

➡️ Conclusion: 
While XGBoost had slightly better precision, Random Forest outperformed it on recall and F1-score, making it more suitable for identifying higher-income individuals accurately 
especially important in use cases where missing high-income earners is costlier than occasional false positives.

Additionally, Random Forest showed more balanced performance across both classes, and learning curves confirmed no major signs of overfitting.

Selected Final Model: Random Forest (with GridSearch tuning)
Best Params: n_estimators=200, min_samples_split=5, class_weight='balanced'

Test Accuracy: 85%

Final Decision: Chosen for higher F1-score, better recall, and robust generalization.


📊 Key Insights : 
The dataset was slightly imbalanced. Using class_weight='balanced' helped improve minority class recall.
Top 5 Important Features:
education
capital-gain
hours-per-week
occupation
marital-status


🛠️ Technologies Used:
Python, Jupyter Notebook
Scikit-learn, XGBoost
Pandas, NumPy
Matplotlib, Seaborn
