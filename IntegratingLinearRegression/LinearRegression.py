# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset (house prices based on square footage)
data = {
    'SquareFootage': [800, 950, 1100, 1250, 1400, 1600, 1800, 2000, 2200, 2600, 3000, 3500],
    'Price':         [120000, 140000, 165000, 180000, 205000, 230000,
                      255000, 290000, 325000, 380000, 450000, 550000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Inspect the first few rows
print(df.head())

# Features (X) and Target (y)
X = df[['SquareFootage']]  # Feature(s)
y = df['Price']            # Target variable

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# MSE gives the average squared difference between the actual and predicted values
# (the lower, the better).

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
# R² tells you how well the model fits the data
# (1 means a perfect fit, while 0 indicates no fit).

# Display the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# -- Visualization -- #

# Plot the data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()

# Show the plot
plt.show()