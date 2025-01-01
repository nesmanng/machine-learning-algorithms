### Random Forest Classification

"""
Random Forest is an ensemble learning method, primarily used for classification and regression tasks. It works by constructing a collection (or "forest") of decision trees during training and aggregating their predictions (either averaging for regression or voting for classification).

Steps:
------
1. Load dataset

Preprocess dataset (e.g., handling missing values, encoding categorical variables)

2. Split data

Divide data into training and test sets for model evaluation

3. Model training 

Use RandomForestClassifier for classification problems

4. Model Evaluation

Using metrics like accuracy, precision and recall for classification

5. Hyperparameter Tuning

Optimize model using GridSearchCV or RandomizedSearchCV
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset 
data = load_iris()
X, y = data.data, data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42) # n_estimators is the no. of trees in the forest
rf_clf.fit(X_train, y_train)

# Predict on test set
y_pred = rf_clf.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")

# Hyperparameter Tuning

param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring="neg_mean_squared_error",
                           n_jobs=1)

grid_search.fit(X_train, y_train)

print("Best parameters:" , grid_search.best_params_)
best_rf = grid_search.best_estimator_

y_test_pred = best_rf.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Tuned Test MSE: {test_mse}")
