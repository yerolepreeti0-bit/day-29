import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("WineQT.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:", data.shape)

# Remove unnecessary column
data = data.drop("Id", axis=1)

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, pred)
print("\nMean Squared Error:", mse)

# Predict wine quality for a sample
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("\nPredicted Wine Quality Score:", prediction[0])

