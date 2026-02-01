#I want to train a model that learns patterns from data.

#Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

#Load Data
data = pd.read_csv('C:\\Users\\VASA6709\\Downloads\\AIML\\breast-cacer-ml\\data\\Breast_cancer_data.csv')

#Preprocess data
X = data.drop(columns='diagnosis')
y = data['diagnosis']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

#Standardize features (if necessary)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Save the model
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler,'scaler.pkl')
print("Model trained and saved successfully.")

