# KNN Diabetes Prediction Model

## Effect of K on Model Performance

The value of K in K-Nearest Neighbors controls how many neighboring training points the model consults before assigning a class label, and it directly governs the bias-variance tradeoff.

When K is too small (overfitting): The model memorizes the training data instead of learning general patterns. At K = 1, training accuracy is perfect but test accuracy drops, meaning the model fails to generalize to new data.

When K is too large (underfitting): The model averages over too many neighbors and loses the ability to detect meaningful patterns. Both training and test accuracy decline, and the classifier becomes too simple to make useful predictions.

Best K for this dataset: Based on the training vs. testing accuracy plot, K = 23 produced the highest test accuracy, making it the optimal value for this dataset.

Why prefer odd K in binary classification: An even K can result in a tie vote — for example, 2 neighbors predict each class and the model has no clear winner. An odd K always produces a majority, guaranteeing a decisive prediction without any tiebreaking needed.
