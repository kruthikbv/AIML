#I want to evaluate the trained model using various metrics.

#Import necessary libraries
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

#Load model from joblib file
model = joblib.load('breast_cancer_model.pkl')



#Print accuracy

#Print precision

#Print recall

#Print F1-score

#Print AUC

