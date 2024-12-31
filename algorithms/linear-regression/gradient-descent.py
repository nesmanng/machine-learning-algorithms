### Gradient Descent

"""
Gradient descent is an optimization algorithm used to find the minimum of a function, often the cost or loss function in machine learning. It's called "gradient descent" because it uses the gradient (derivative) of the function to determine the direction and size of the steps to take to reach the minimum.

Steps to perform Gradient Descent
---------------------------------
1. Initialize parameters

Start with some initial values for the parameters of the model, which are usually randomly chosen

2. Calculate Gradient

Evaluate the derivative of the cost function w.r.t each parameter at the current point.
The gradient tells you the slope of the cost function at the current position.

3. Update the parameters

To move towards the minimum, update each parameter in the opposite direction of the gradient, taking a step proportional to the learning rate.
The learning rate is a small positive value that determines the size of steps you take.
Equation: new_parameter = old_parameter - learning rate * gradient

4. Repeat steps 2-3 until convergence

Keep calculating the gradient and updating the parameters iteratively. Each step should move you closer to the min of the cost function.
The process stops when the gradient gets very close to zero, or after a fixed number of iterations.

Note: Gradient descent is a greedy algorithm - it moves in the direction of the steepest descent at the current point.
"""

from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the California housing dataset from the sklearn library
california = fetch_california_housing()
X = california.data
y = california.target.reshape(-1, 1) # Target (reshape for compatibility)

# Normalize features for better gradient descent convergence
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add a bias term (column of ones) to X
# Practically, the bias term exists as there is a baseline value e.g., inherent land value that is independent of the house's features
X = np.c_[np.ones((X.shape[0], 1)), X] # Add intercept term
print(f"Shape of X: {X.shape}, Shape of Y: {y.shape}")

# Initialize parameters to zeros, the starting point for the gradient descent
theta = np.zeros((X.shape[1], 1)) # Shape: (m + 1, 1)

# Define the cost function - mean-squared error
def compute_cost(X, y, theta):
    m = len(y) # Number of training examples
    # Shape of X: (m, n + 1), Shape of y: (n + 1, 1)
    predictions = X.dot(theta) # Resulting shape: (m, 1)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2) # Mean-squared error
    return cost

# Define Gradient Descent Algorithm
# Implement gradient descent to update theta iteratively based on the cost function gradient
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y) # Number of training examples
    cost_history = [] # To store the cost at each iteration

    for i in range(iterations):
        # Shape of X.T: (n + 1, m), Shape of X.dot(theta): (m, 1)
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y) # Compute gradients, resulting shape: (n + 1, 1)
        theta -= learning_rate * gradients # Update theta
        cost_history.append(compute_cost(X, y, theta)) # Track the cost

    return theta, cost_history

# Train the model
# Set the learning rate and number of iterations, and train the model using gradient descent

# Set hyperparameters
learning_rate = 0.01
iterations = 1000

# Model training
theta_optimal, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)
print(f"Optimal parameters(theta): \n{theta_optimal}")

# Visualize cost function
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (J)")
plt.title("Cost Function Convergence")
plt.show()

# Model Evaluation

# Predictions
predictions = X.dot(theta_optimal)

# Calculate MSE on the dataset
mse = np.mean((predictions - y) ** 2)
print(f"Mean Squared Error on the dataset: {mse}")