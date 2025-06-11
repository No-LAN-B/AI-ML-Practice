# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv('personality_dataset.csv')
# Drop any rows with null values
clean_data = data.dropna()
data.dropna(inplace=True)

print(data.head())
print(data.info())

# Splitting Features

feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
X = data[feature_cols]
Y = data['Personality']


# One-hot encode all categorical columns
# This will convert 'Yes'/'No', strings, etc. into dummy 0/1 columns.
X_enc = pd.get_dummies(X, drop_first=True)

# random state is for shuffle=true which is set by default 42 is just a random number
X_train, X_test, y_train, y_test = train_test_split(X_enc, Y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the model's learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
print("Predicted Outcomes Introvert/Extrovert:", y_pred)
print("Actual Outcomes:", y_test.values)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# ===  PLOT #1: Scatter-plots of each feature vs. Personality  ===
# (map your Personality labels to 0/1 so they can live on the y-axis)
personality_map = {lbl: i for i, lbl in enumerate(data['Personality'].unique())}
Y_num = data['Personality'].map(personality_map)

plt.figure(figsize=(12, 8))
n_feats = len(feature_cols)
n_cols  = 3
n_rows  = (n_feats + n_cols - 1) // n_cols
for idx, feat in enumerate(feature_cols, start=1):
    ax = plt.subplot(n_rows, n_cols, idx)
    ax.scatter(data[feat], Y_num, alpha=0.6)
    ax.set_xlabel(feat)
    ax.set_yticks(list(personality_map.values()))
    ax.set_yticklabels(list(personality_map.keys()))
    if idx % n_cols == 1:
        ax.set_ylabel('Personality')
plt.tight_layout()
plt.suptitle("Feature vs. Personality (0=Introvert, 1=Extrovert)", y=1.02)
plt.show()

# ===  PLOT #2: Overlaid histograms per feature/class  ===
labels = data['Personality'].unique()

plt.figure(figsize=(12, 8))
for idx, feat in enumerate(feature_cols, start=1):
    ax = plt.subplot(n_rows, n_cols, idx)
    for lbl in labels:
        subset = data[data['Personality'] == lbl]
        ax.hist(subset[feat], bins=15, alpha=0.5, label=lbl)
    ax.set_title(feat)
    if idx == 1:
        ax.legend()
plt.tight_layout()
plt.suptitle("Feature Distributions by Personality Class", y=1.02)
plt.show()

