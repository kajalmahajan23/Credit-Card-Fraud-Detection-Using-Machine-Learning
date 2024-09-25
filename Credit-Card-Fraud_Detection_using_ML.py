# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Data Cleaning
df.dropna(inplace=True)  # Drop missing values if any

# Features and Target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE for imbalanced dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Function to save confusion matrix and classification report
def save_metrics(model_name, y_true, y_pred):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['True', 'Fraudulent'], columns=['Predicted True', 'Predicted Fraudulent'])
    cm_df.to_csv(f"{model_name}_confusion_matrix.csv")
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{model_name}_classification_report.csv")

# 1. Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_smote, y_train_smote)
y_pred_log_reg = log_reg.predict(X_test)

# Save metrics for Logistic Regression
save_metrics("Logistic_Regression", y_test, y_pred_log_reg)

# Confusion Matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['True', 'Fraudulent'], 
            yticklabels=['True', 'Fraudulent'])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report for Logistic Regression:\n")
print(classification_report(y_test, y_pred_log_reg))
print("\n" + "="*50 + "\n")





# 2. KNN
knn = KNeighborsClassifier()
knn.fit(X_train_smote, y_train_smote)
y_pred_knn = knn.predict(X_test)

# Save metrics for KNN
save_metrics("KNN", y_test, y_pred_knn)

# Confusion Matrix for KNN
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['True', 'Fraudulent'], 
            yticklabels=['True', 'Fraudulent'])
plt.title("KNN Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report for KNN:\n")
print(classification_report(y_test, y_pred_knn))
print("\n" + "="*50 + "\n")




# 3. XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb.predict(X_test)

# Save metrics for XGBoost
save_metrics("XGBoost", y_test, y_pred_xgb)

# Confusion Matrix for XGBoost
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Genuine', 'Fraudulent'], 
            yticklabels=['Genuine', 'Fraudulent'])
plt.title("XGBoost Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report for XGBoost:\n")
print(classification_report(y_test, y_pred_xgb))
print("\n" + "="*50 + "\n")