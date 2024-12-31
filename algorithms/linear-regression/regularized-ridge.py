### Ridge Regression

"""
Type of linear regression that includes a regularization term to prevent overfitting.
It modifies the ordinary least squares (OLS) loss function by adding the L2 norm of the coefficients as a penalty.

Loss function is defined as:
----------------------------
Loss = 1/2m * sum[(observed y - predicted y)^2] + lambda * sum[theta^2]

where 
lambda is the regularization parameter
theta is the coefficients (except for intercept)

Regularization term penalizes large coefficients to reduce model complexity and prevents overfitting by discouraging overreliance on any single feature.

Steps
-----
1. Standardization

Scale features so that they have a mean of 0 and standard deviation of 1.

2. Choose lambda:

Determines the strength of regularization. A larger value means more shrinkage / regularization

3. Fit the model

Use the ridge regression to minimize the loss function.

4. Evaluate the model

Using metrics like MSe, R-squared

5. Tune lambda

Using cross-validation to find the optimal lambda

"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load Diabetes dataset
# Diabetes dataset has 10 features and 1 target variable
data = load_diabetes()
X, y = data.data, data.target

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge regression with lambda = 1.0
ridge = Ridge(alpha = 1.0)
ridge.fit(X_train, y_train)

# Predictions
y_pred = ridge.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Coefficients and Intercept
print("Coefficients", ridge.coef_)
print("Intercept", ridge.intercept_)

## Tuning lambda
from sklearn.linear_model import RidgeCV

# Ridge Regression with Cross Validation
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Best alpha: {ridge_cv.alpha_}")