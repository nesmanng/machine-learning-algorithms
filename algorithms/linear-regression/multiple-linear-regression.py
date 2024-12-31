### Linear Regression 

"""
Predicting a continuous dependent variable based on one or more indendent variables. 

y = b0 + b1x + ... + b2x + e

where y is the dependent variable, x is the independent variable, b0 is the intercept, b1...b2 are coefficients for respective features and e is the error term.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Example - Predicting house prices based on square footage

# Data Preparation
np.random.seed(42)
X = np.random.rand(100, 3) # Features: square footage, number of rooms, age of house
y = 50 + 100 * X[:,0] + 200 * X[:,1] - 150 * X[:,2] + np.random.randn(100) * 50

# Split data into training and testing set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model and train model
model = LinearRegression() # Create linear regression model
model.fit(X_train, y_train) # Train model
y_pred = model.predict(X_test) # Make predictions on testing data

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize actuals vs pred since we have multiple features
plt.scatter(y_test,y_pred)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()
