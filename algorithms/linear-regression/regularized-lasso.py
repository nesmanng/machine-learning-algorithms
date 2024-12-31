### Lasso Regression

"""
Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a type of linear regression that performs regularization by adding a penalty term based on the L1 norm (the sum of the absolute values of the coefficients) to the loss function.

Loss function is defined as:
----------------------------
Loss = 1/2m * sum[(observed y - predicted y)^2] + lambda * sum[abs(theta)]

where 
lambda is the regularization parameter
abs(theta) is the absolute value of the coefficients

L1 norm penalty has two main effects:
Shrinkage - reduces the magnitude of coefficients
Feature Selection - can force some coefficients to be exactly zero, effectively removing less important features from the model

Steps
-----
1. Standardization

Ensures all features are treated equally by the regularization term.

2. Choose lambda:

Determines the strength of regularization. A larger value means more shrinkage / regularization.

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
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes dataset
data = load_diabetes()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lasso Regression with lambda (alpha) = 1.0
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Predictions
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Coefficients and Intercept
print("Coefficients:", lasso.coef_)
print("Intercept:", lasso.intercept_)

## Tuning lambda

from sklearn.linear_model import LassoCV

# Lasso Regression with Cross-Validation
# Generate alpha values from 0.01 to 1.0 with a step size of 0.1
alphas = np.arange(0.01, 1.1, 0.1)
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"Best alpha: {lasso_cv.alpha_}")

