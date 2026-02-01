#I want to evaluate the trained model using various metrics.

#Import necessary libraries
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from train import X_test, y_test
#Load model from joblib file
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

y_pred = model.predict(X_test)
#Print accuracy
acc = accuracy_score(y_test,y_pred)
print(f"Accuracy: {acc}")

#Print precision
precision = precision_score(y_test,y_pred)
print(f"Precision: {precision}")

#Print recall
recall = recall_score(y_test,y_pred)
print(f"Recall: {recall}")
#Print F1-score
f1 = f1_score(y_test,y_pred)
print(f"F1-score: {f1}")

#Print AUC
auc = roc_auc_score(y_test,y_pred)
print(f"AUC: {auc}")
