### Partial Least Squares

"""
Partial Least Squares, unlike PCA, finds directions in the data that maximizes the covariance between the predictors and response variable. Thus, it explicitly considers the relationship between the predictors and response.

It is also a dimensionlity reduction technique, but is a supervised method unlike PCA. It is often more effective than PCA for prediction tasks because it directly incorporates the relationship between predictors and response.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load 'iris' dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# DataFrame for easier handling
X_df = pd.DataFrame(X, columns=feature_names)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target to one-hot encoded format for multiple response PLS
# One-hot encoding as PLS uses the response variable, unlike PCA
y_onehot = pd.get_dummies(y).values

# Apply PLS
n_components = 4  # Maximum possible components for iris dataset
pls = PLSRegression(n_components=n_components)
X_pls = pls.fit_transform(X_scaled, y_onehot)[0]

# Create a DataFrame with PLS results
pls_df = pd.DataFrame(
    X_pls,
    columns=[f'PLS{i+1}' for i in range(X_pls.shape[1])]
)
pls_df['Species'] = pd.Categorical.from_codes(y, iris.target_names)

# Calculate explained variance for X and Y
x_scores = pls.x_scores_
x_explained_var = np.var(x_scores, axis=0) / np.sum(np.var(x_scores, axis=0))
x_cum_var = np.cumsum(x_explained_var)

y_scores = pls.y_scores_
y_explained_var = np.var(y_scores, axis=0) / np.sum(np.var(y_scores, axis=0))
y_cum_var = np.cumsum(y_explained_var)

# Print the results
print("Explained variance ratio for X by component:")
for i, var in enumerate(x_explained_var):
    print(f"PLS{i+1}: {var:.3f}")

print("\nCumulative explained variance ratio for X:")
for i, var in enumerate(x_cum_var):
    print(f"First {i+1} components: {var:.3f}")

print("\nExplained variance ratio for Y by component:")
for i, var in enumerate(y_explained_var):
    print(f"PLS{i+1}: {var:.3f}")

# Create visualization plots
plt.figure(figsize=(15, 10))

# 1. X Explained Variance Plot
plt.subplot(2, 2, 1)
plt.plot(range(1, len(x_explained_var) + 1), 
         x_explained_var, 'bo-')
plt.xlabel('PLS Component')
plt.ylabel('Explained Variance Ratio (X)')
plt.title('X Explained Variance by Component')

# 2. X Cumulative Variance Plot
plt.subplot(2, 2, 2)
plt.plot(range(1, len(x_cum_var) + 1), 
         x_cum_var, 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio (X)')
plt.title('X Cumulative Explained Variance')

# 3. Score Plot
plt.subplot(2, 2, 3)
sns.scatterplot(
    data=pls_df,
    x='PLS1',
    y='PLS2',
    hue='Species',
    style='Species'
)
plt.title('First Two PLS Components')

# 4. Loading Plot
plt.subplot(2, 2, 4)
loadings = pls.x_loadings_
for i, feat_name in enumerate(feature_names):
    plt.arrow(0, 0,
              loadings[i, 0]*3, loadings[i, 1]*3,
              color='r', alpha=0.5)
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2,
             feat_name, color='r')

plt.scatter(X_pls[:, 0], X_pls[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('PLS1')
plt.ylabel('PLS2')
plt.title('Biplot: PLS with Feature Loadings')

plt.tight_layout()
plt.show()

# Print component loadings
print("\nX loadings:")
loadings_df = pd.DataFrame(
    pls.x_loadings_,
    columns=[f'PLS{i+1}' for i in range(n_components)],
    index=feature_names
)
print(loadings_df)

# Print coefficients
print("\nRegression coefficients:")
coef_df = pd.DataFrame(
    pls.coef_,
    columns=iris.target_names,
    index=feature_names
)
print(coef_df)

# Calculate and print R² score for each response variable
y_pred = pls.predict(X_scaled)
r2_scores = np.array([
    np.corrcoef(y_onehot[:, i], y_pred[:, i])[0, 1]**2
    for i in range(y_onehot.shape[1])
])

print("\nR² scores for each class:")
for i, (name, r2) in enumerate(zip(iris.target_names, r2_scores)):
    print(f"{name}: {r2:.3f}")