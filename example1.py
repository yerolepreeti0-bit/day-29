import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create dataset
X, y = make_classification(n_features=5, n_redundant=0,
                           n_informative=5, n_clusters_per_class=1)

df = pd.DataFrame(X, columns=['col1','col2','col3','col4','col5'])
df['target'] = y

print("Dataset Shape:", df.shape)

# Row sampling function
def sample_rows(df, percent):
    return df.sample(int(percent * df.shape[0]), replace=True)

# Row sampling
df1 = sample_rows(df,0.2)
df2 = sample_rows(df,0.2)
df3 = sample_rows(df,0.2)

# Show row samples
print("\nRow Sample 1:")
print(df1)

print("\nRow Sample 2:")
print(df2)

print("\nRow Sample 3:")
print(df3)

# Train models
clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()

clf1.fit(df1.iloc[:,0:5], df1.iloc[:,-1])
clf2.fit(df2.iloc[:,0:5], df2.iloc[:,-1])
clf3.fit(df3.iloc[:,0:5], df3.iloc[:,-1])

# -------- Tree 1 --------
plt.figure(figsize=(12,8))
plot_tree(clf1, filled=True)
plt.title("Decision Tree 1")
plt.show()

# -------- Tree 2 --------
plt.figure(figsize=(12,8))
plot_tree(clf2, filled=True)
plt.title("Decision Tree 2")
plt.show()

# -------- Tree 3 --------
plt.figure(figsize=(12,8))
plot_tree(clf3, filled=True)
plt.title("Decision Tree 3")
plt.show()