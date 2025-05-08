### Logistic Regression

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Splitting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Fitting a logistic regression model 
model = LogisticRegression(max_iter = 10000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
cm = confusion_matrix(y_test, y_pred)
tp, fn, fp, tn = cm.ravel()
sensitivity = tp / (tp + fn) # True positive rate i.e., how many of the true positives were correctly identified
specificity = tn / (tn + fp) # True negative rate i.e., how many of the true negatives were correctly identified

accuracy = accuracy_score(y_test, y_pred) # Percentage of correct predictions

print("Confusion Matrix:\n", cm)
print("Sensitivty: ", sensitivity)
print("Specificity: ", specificity)
print("Accuracy: ", accuracy)

