import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("train.csv")

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Shape:", data.shape)

# Convert categorical columns
data["Gender"] = data["Gender"].map({"Male":1, "Female":0})
data["Married"] = data["Married"].map({"Yes":1, "No":0})
data["Education"] = data["Education"].map({"Graduate":1, "Not Graduate":0})
data["Self_Employed"] = data["Self_Employed"].map({"Yes":1, "No":0})
data["Property_Area"] = data["Property_Area"].map({"Urban":2, "Semiurban":1, "Rural":0})

# Convert Dependents
data["Dependents"] = data["Dependents"].replace("3+",4)

# Convert target column
data["Loan_Status"] = data["Loan_Status"].map({"N":0, "Y":1})

# Remove missing values
data = data.dropna()

print("\nCleaned Dataset Shape:", data.shape)

# Features and target
X = data.drop(columns=["Loan_ID","Loan_Status"])
y = data["Loan_Status"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Predict sample applicant
sample = X.iloc[0].values.reshape(1,-1)

result = model.predict(sample)

print("\nPrediction for sample applicant:")
if result[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")

# Decision Tree diagram
plt.figure(figsize=(15,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Approved","Approved"],
    filled=True
)

plt.title("Loan Approval Decision Tree")
plt.show()