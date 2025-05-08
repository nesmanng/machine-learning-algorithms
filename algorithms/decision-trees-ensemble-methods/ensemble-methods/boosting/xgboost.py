### XGBoost (Extreme Gradient Boosting)

"""
XGBoost (Extreme Gradient Boosting) is a powerful and popular gradient boosting library that can be used for both regression and classification tasks. It is an optimized implementation of gradient boosting that offers several advanced features and has proven to be highly effective in many machine learning competitions and real-world applications.

- XGBoost builds an ensemble of decision trees, where each tree is trained to predict the residuals of the previous trees. The final prediction is the sum of the predictions from all the individual trees in the ensemble.
- XGBoost uses gradient boosting, which sequentially trains each tree to minimize a loss function. In each iteration, XGBoost calculates the negative gradients of the loss function w.r.t the predictions, which become the target values for the next tree.
- XGBoost incorporates regularization techniques to prevent overfitting and improve generationalization, by adding a regularization term to the loss function, penalizing complex models and encourages simpler, more robust trees.
- XGBoost uses a technique called tree pruning to limit depth and complexity of individual trees.
- XGBoost provides a way to measure the importance of each feature in the model, calculating the feature importance based on the number of times a feature is used to split data across all trees and improvement in loss function resulting from each split.
"""