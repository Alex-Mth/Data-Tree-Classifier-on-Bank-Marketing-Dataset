# Task 3: Decision Tree Classifier - Bank Marketing Dataset
# ----------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option('display.max_columns', None)

# -------------------------------------------------
# ✅ STEP 1: AUTO FILE DETECTION
# -------------------------------------------------
print("\n--- SEARCHING FOR DATASET FILE ---")

possible_paths = [
    os.getcwd(),
    os.path.dirname(__file__),
    os.path.join(os.getcwd(), "data"),
    os.path.join(os.path.dirname(__file__), "data")
]

file_path = None

for path in possible_paths:
    full_path = os.path.join(path, "bank-full.csv")
    if os.path.exists(full_path):
        file_path = full_path
        break

# ❌ Stop program if file not found
if file_path is None:
    print("\n❌ ERROR: bank-full.csv not found!")
    print("\n➡️ Place the dataset in ANY ONE of these locations:\n")
    for p in possible_paths:
        print(" -", p)
    print("\n➡️ Then run the script again.")
    exit()

print("\n✅ Dataset found at:", file_path)

# -------------------------------------------------
# ✅ STEP 2: LOAD DATASET
# -------------------------------------------------
data = pd.read_csv(file_path, sep=';')

print("\n--- Dataset Loaded ---")
print(data.head())
print("\nShape:", data.shape)

# -------------------------------------------------
# ✅ STEP 3: CHECK FOR MISSING VALUES
# -------------------------------------------------
print("\n--- Missing Values ---")
print(data.isnull().sum())

# -------------------------------------------------
# ✅ STEP 4: ENCODE CATEGORICAL COLUMNS
# -------------------------------------------------
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

print("\n--- After Encoding ---")
print(data.head())

# -------------------------------------------------
# ✅ STEP 5: SPLIT DATA INTO FEATURES & TARGET
# -------------------------------------------------
X = data.drop('y', axis=1)
y = data['y']  # Target column

# -------------------------------------------------
# ✅ STEP 6: TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------
# ✅ STEP 7: TRAIN DECISION TREE CLASSIFIER
# -------------------------------------------------
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------------------------
# ✅ STEP 8: PREDICTIONS
# -------------------------------------------------
y_pred = clf.predict(X_test)

# -------------------------------------------------
# ✅ STEP 9: EVALUATE MODEL
# -------------------------------------------------
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------
# ✅ STEP 10: VISUALIZE DECISION TREE
# -------------------------------------------------
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Classifier - Bank Marketing Dataset")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# ✅ STEP 11: CORRELATION HEATMAP
# -------------------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
