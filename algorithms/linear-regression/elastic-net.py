### Elastic Net 

"""
Elastic Net Regression is a linear regression model that combines L1 (Lasso) and L2 (Ridge) regularization. It is particularly useful when features are highly correlated or when there are many irrelevant features, as it balances the benefits of both types of regularization.

Loss Function is defined as:
----------------------------
Loss = 1/2m * sum[(observed y - predicted y)^2] + lambda_1 * sum[abs(theta_1)] + lambda_2 * sum[theta_2 ^ 2]

or

Loss = 1/2m * sum[(observed y - predicted y)^2] + alpha * (l1_ratio * sum[abs(theta_1)]) + (1-l1_ratio) * sum[theta_2 ^ 2]

where
lambda_1 is the L1 regularization parameter
lambda_2 is the L2 regularization parameter
l1_ratio is the mix of L1 and L2 regularization
alpha controls the overall strength of regularization in the Elastic Net model

Steps
-----
1. Standardization

Ensures all features are treated equally by the regularization term.

2. Choose lambda_1 and lambda_2

3. Model training

4. Model evaluation
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load diabetes dataset
data = load_diabetes()
X, y = data.data, data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Elastic Net with cross-validation
elastic_net = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-3,3,100), cv=5, random_state=42)
elastic_net.fit(X_train, y_train)

# Best hyperparameters
print(f"Best alpha: {elastic_net.alpha_}")
print(f"Best l1_ratio: {elastic_net.l1_ratio_}")

# Predictions
y_pred = elastic_net.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")