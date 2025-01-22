### Principal Component Analysis (PCA)

"""
PCA is a statistical procedure that transforms a set of correlated variables into a new set of uncorrelated variables called principal components. The goal is reduce the dimensionality of the data while preserving as much of the original information as possible. It can create new features that capture the most important aspects of the data.

It finds a new set of axes (principal components) that align with the directions of maximum variance in the data. Thus, the first principal component captures the most variation in the data, followed by the second principal component and so on...
Each principal component is a linear combination of the original variables, and the principal components are orthogonal/perpendicular to each other, meaning that they are uncorrelated.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load 'iris' dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Dataframe for easier handling
df = pd.DataFrame(X, columns=feature_names)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Dataframe with PCA results
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['Species'] = pd.Categorical.from_codes(y, iris.target_names)

# Calculate explained variance ratio
explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

print("Explained variance ratio by component:")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.3f}")

print("\nCumulative explained variance ratio:")
for i, var in enumerate(cum_var):
    print(f"First {i+1} PCs: {var:.3f}")

# Visualization
plt.figure(figsize=(15,10))

# 1. Scree plot
plt.subplot(2, 2, 1)
plt.plot(range(1, len(explained_var) + 1), 
         explained_var, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

# 2. Cumulative variance plot
plt.subplot(2, 2, 2)
plt.plot(range(1, len(cum_var) + 1), cum_var, 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')

# 3. PCA scatter plot
plt.subplot(2, 2, 3)
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Species',
    style='Species'
)
plt.title('First Two Principal Components')

# 4. Biplot
plt.subplot(2, 2, 4)
loadings = pca.components_.T
for i, (feat_name, loading) in enumerate(zip(feature_names, loadings)):
    plt.arrow(0, 0, 
              loading[0]*3, loading[1]*3,  # Scale arrows for visibility
              color='r', alpha=0.5)
    plt.text(loading[0]*3.2, loading[1]*3.2, 
             feat_name, color='r')

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Species',
    style='Species',
    alpha=0.5
)
plt.title('Biplot: PCA with Feature Loadings')

plt.tight_layout()
plt.show()

# Print component loadings
print("\nComponent loadings:")
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=feature_names
)
print(loadings_df)