### Linear Regression 

"""
Predicting a continuous dependent variable based on one or more indendent variables. 

y = b0 + b1x + e

where y is the dependent variable, x is the independent variable, b0 is the intercept, b1 is the slope of the line and e is the error term.
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
X = 2 * np.random.rand(100, 1) # Features: square footage (100 random points between 0 and 2)
y = 4 + 3 * X + np.random.rand(100,1 ) # Target: price (linear relation with some noise)

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

# Visualzie model
plt.scatter(X_train, y_train, color = 'blue', label = 'Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test,y_pred, color='red', label='Regression Line')

plt.title("Linear Regression: Square Footage vs House Price")
plt.xlabel("Square Footage(X)")
plt.ylabel("House Prices(y)")
plt.legend()
plt.show()
