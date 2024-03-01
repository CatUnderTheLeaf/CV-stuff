# Regularization

Regularization is a technique used in machine learning to reduce overfitting. Overfitting occurs when a model learns the training data too well and is unable to generalize to new data. Regularization works by penalizing complex models, which forces the model to learn simpler and more generalizable relationships between the features and the target variable.

### L1 (Lasso Regression)

```math

L(w^T) = Loss(w^T) + \lambda\sum_{i=1}^{m}|{w_i}|
```

The L1 norm is the sum of the absolute values of the coefficients. This penalization forces the model to shrink the coefficients towards zero, which can lead to some coefficients being set to zero exactly. This feature selection capability of Lasso regression makes it a powerful tool for variable selection.

- it is easy to implement and can be trained as a one-shot thing, meaning that once it is trained you are done with it and can just use the parameter vector and weights.
- is robust in dealing with outliers. It creates sparsity in the solution (most of the coefficients of the solution are zero), which means the less important features or noise terms will be zero. It makes L1 regularization robust to outliers.
- has an unstable solution and can possibly have multiple solutions

### L2 (Ridge Regression)

```math

L(w^T) = Loss(w^T) + \lambda\sum_{i=1}^{m}{w_i}^2
```

Regularization adds the penalty as model complexity increases. The regularization parameter (lambda) penalizes all the parameters except intercept so that the model generalizes the data and won’t overfit. Ridge regression adds “squared magnitude of the coefficient” as penalty term to the loss function.

- forces the weights to be small but does not make them zero and does not give the sparse solution.
- is not robust to outliers as square terms blow up the error differences of the outliers, and the regularization term tries to fix it by penalizing the weights.
- performs better when all the input features influence the output, and all with weights are of roughly equal size.
- can learn complex data patterns
- has a stable solution and always one solution

### L1+L2 (Elastic Net)

Elastic Net is a hybrid regularization technique that combines the L1 and L2 penalties. The Elastic Net penalty is a weighted sum of the L1 and L2 penalties, where the weight parameter controls the relative importance of the two penalties.

