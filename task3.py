import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Pumpkin_Seeds_Dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:", data.shape)

# Convert target column to numbers
data["Class"] = data["Class"].map({
    "CERCEVELIK":0,
    "URGUP_SIVRISI":1
})

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Predict sample
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("\nPredicted Pumpkin Seed Type:")

if prediction[0] == 0:
    print("CERCEVELIK")
else:
    print("URGUP_SIVRISI")

# Decision tree diagram
plt.figure(figsize=(12,6))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["CERCEVELIK","URGUP_SIVRISI"],
    filled=True
)

plt.title("Pumpkin Seed Decision Tree")
plt.show()

