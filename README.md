# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Placement_Data.csv")   
print("Dataset Preview:")
print(data.head())
data = data.drop(["sl_no", "salary"], axis=1)
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})
X = data.drop("status", axis=1)
y = data["status"]
X = pd.get_dummies(X, drop_first=True)
print("\nAfter Encoding:")
print(X.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
<img width="741" height="723" alt="image" src="https://github.com/user-attachments/assets/131e7520-0363-41e9-b688-04f0e7d0f131" />


## Output:
<img width="711" height="745" alt="image" src="https://github.com/user-attachments/assets/72e13d3e-5268-4b8f-b7b6-059ced0c9bc1" />

<img width="662" height="732" alt="image" src="https://github.com/user-attachments/assets/681b76a4-95e2-4afa-b35a-402111142f43" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
