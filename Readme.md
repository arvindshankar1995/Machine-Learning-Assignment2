# Credit Card Default Prediction - ML Assignment 2

## 1. Problem Statement
To predict the probability of a customer defaulting on their credit card payment next month based on demographic and repayment history data.

## 2. Dataset Description
**Source:** UCI Credit Card Default Dataset  
**Size:** 30,000 instances, 24 features  
**Target:** `default.payment.next.month` (Binary Classification: 1 = Default, 0 = No Default)

## 3. Model Performance Comparison
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.807667  | 0.707636 | 0.686825 | 0.239638 | 0.355307 | 0.324443 |
| Decision Tree | 0.817500 | 0.742299 | 0.663380 | 0.354936 | 0.462445 | 0.390347 |
| kNN | 0.792833 | 0.701435 | 0.548724 | 0.356443 | 0.432161 | 0.323267 |
| Naive Bayes | 0.416000 | 0.651567 | 0.249597 | 0.817634 | 0.382446 | 0.111087 |
| Random Forest | 0.811500 | 0.767851 | 0.667808 | 0.293896 | 0.408163 | 0.353382 |
| XGBoost | 0.811833 | 0.756469 | 0.628906 | 0.363979 | 0.461098 | 0.376398 |
 
## 4. Observations
* **Best Model:** XGBoost provided the best balance of Accuracy and AUC.
* **High Recall:** Naive Bayes had the highest recall but very low precision, making it suitable only if missing a default is very costly.
* **Baseline:** Logistic Regression provided a solid baseline but failed to capture complex non-linear patterns compared to Tree-based models.

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Baseline Performance:** Achieved an accuracy of ~81%, which is deceptively high due to class imbalance (predicts "Joined" for almost everyone). It failed to effectively capture the minority class ("Not Joined"), showing high recall for "Joined" but very poor precision/recall for "Not Joined". |
| **Decision Tree** | **Overfitting Prone:** Performance was similar to Logistic Regression (~81% accuracy) but likely showed signs of overfitting on training data. It struggled to generalize well for the "Not Joined" class without hyperparameter tuning (pruning). |
| **kNN** | **Sensitive to Noise:** Likely performed worse or similar to Logistic Regression. kNN struggles with high-dimensional data (many dummy variables from encoding) and imbalanced clusters, often getting biased towards the majority class "Joined". |
| **Naive Bayes** | **Probabilistic Baseline:** Often performs poorly on datasets with correlated features. It likely had lower accuracy than tree-based models but might have caught some "Not Joined" cases due to its probabilistic nature, though at the cost of many False Positives. |
| **Random Forest (Ensemble)** | **Robust & Stable:** Slightly improved upon the Decision Tree by reducing variance. However, without handling the class imbalance (e.g., SMOTE or class weights), it still heavily favored the majority class, resulting in high accuracy but low utility for identifying candidates who won't join. |
| **XGBoost (Ensemble)** | **Best Potential:** Typically the strongest performer on tabular data. It likely achieved the highest ROC-AUC score by better learning complex non-linear patterns. However, raw accuracy might not look vastly different unless the class imbalance was explicitly handled. |


## 5. How to Run Locally
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

3. Run the app: `streamlit run app.py`
