# Loss Functions

This equation represents how a neural network processes the input data at each layer and eventually produces a predicted output value.
```math

\hat{y} = \sigma(w^Tx + b)\\
```
`y` - output value, `x` - input values, `wT` - weights, `b` - bias

A __loss function__ is a function that compares the target and predicted output values. It’s a method of evaluating how well your algorithm models your dataset. If your predictions are totally off, your loss function will output a higher number. If they’re pretty good, it’ll output a lower number. As you change pieces of your algorithm to try and improve your model, your loss function will tell you if you’re getting anywhere.

The hyperparameters are adjusted to minimize the average loss — we find the weights `wT` and biases `b` that minimize the value of `J` (average loss).

```math

J(w^T,b) = \frac{1}{m}\sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})
```

## Types of Loss Functions

In supervised learning, there are two main types of loss functions — these correlate to the 2 major types of neural networks:

- Regression Loss Functions — used in regression neural networks; given an input value, the model predicts a corresponding output value (rather than pre-selected labels); Ex. Mean Squared Error, Mean Absolute Error
- Classification Loss Functions — used in classification neural networks; given an input, the neural network produces a vector of probabilities of the input belonging to various pre-set categories — can then select the category with the highest probability of belonging; Ex. Binary Cross-Entropy, Categorical Cross-Entropy

| Problem Type | Output type | Loss function |
| --- | --- | --- |
| Regression | Numerical value | [Mean Squared Error (MSE)](#mean-squared-error-mse), [Mean Absolute Error (MAE)](#mean-absolute-error-mae), Huber loss, Log-cosh loss, Quantile loss|
| Classification | Binary outcome | [Binary Cross Entropy](#binary-cross-entropylog-loss) |
| Classification | Single label, multiple classes | [Cross Entropy](#categorical-cross-entropy-loss) |
| Classification | Multiple labels, multiple classes | [Binary Cross Entropy](#binary-cross-entropylog-loss) |
| Object detection | Boxes | [IoU loss](#iou-loss-function), Generalized IoU loss (GIoU loss), Smooth L1 loss, Focal loss | 
| Face recognition | ... | Contrastive loss, Triplet loss, Center loss, Angular softmax loss (A-Softmax loss), Additive margin softmax loss (AM-Softmax loss), Additive angular margin loss (ArcFace loss) |
| Unsupervised learning | Numerical value | [Mean Squared Error (MSE)](#mean-squared-error-mse), Distance error, Reconstruction error, Negative variance |


### Mean Squared Error (MSE)

One of the most popular loss functions, MSE finds the average of the squared differences between the target and the predicted outputs

```math

MSE = \frac{1}{n}\sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2
```
This function has numerous properties that make it especially suited for calculating loss. The difference is squared, which means it does not matter whether the predicted value is above or below the target value; however, values with a large error are penalized. MSE is also a convex function (as shown in the diagram above) with a clearly defined global minimum — this allows us to more easily utilize gradient descent optimization to set the weight values.

However, one disadvantage of this loss function is that it is very sensitive to outliers; if a predicted value is significantly greater than or less than its target value, this will significantly increase the loss.

### Mean Absolute Error (MAE)

MAE finds the average of the absolute differences between the target and the predicted outputs.

```math

MAE = \frac{1}{n}\sum_{i=1}^{n} |y^{(i)} - \hat{y}^{(i)}|
```

This loss function is used as an alternative to MSE in some cases. As mentioned previously, MSE is highly sensitive to outliers, which can dramatically affect the loss because the distance is squared. MAE is used in cases when the training data has a large number of outliers to mitigate this.

It also has some disadvantages; as the average distance approaches 0, gradient descent optimization will not work, as the function's derivative at 0 is undefined (which will result in an error, as it is impossible to divide by 0).

Because of this, a loss function called a __Huber Loss__ was developed, which has the advantages of both MSE and MAE.
> If the absolute difference between the actual and predicted value is less than or equal to a threshold value, 𝛿, then MSE is applied. Otherwise — if the error is sufficiently large — MAE is applied.

### Likelihood loss

The likelihood function is also relatively simple, and is commonly used in classification problems. The function takes the predicted probability for each input example and multiplies them. And although the output isn’t exactly human-interpretable, it’s useful for comparing models.

```math

L = \frac{1}{n}\prod_{i=1}^{n} (y_i*p_i + (1 - y_i)*(1 - p_i))
```

For example, consider a model that outputs probabilities of [0.4, 0.6, 0.9, 0.1] for the ground truth labels of [0, 1, 1, 0]. The likelihood loss would be computed as (0.6) * (0.6) * (0.9) * (0.9) = 0.2916. Since the model outputs probabilities for TRUE (or 1) only, when the ground truth label is 0 we take (1-p) as the probability. In other words, we multiply the model’s outputted probabilities together for the actual outcomes.

### Binary Cross-Entropy/Log Loss

This is the loss function used in binary classification models — where the model takes in an input and has to classify it into one of two pre-set categories.

```math

BCE Loss = \frac{1}{n}\sum_{i=1}^{n} -(y_i*\log(p_i) + (1 - y_i)*\log(1 - p_i))
```

Classification neural networks work by outputting a vector of probabilities — the probability that the given input fits into each of the pre-set categories; then selecting the category with the highest probability as the final output.

In binary classification, there are only two possible actual values of y — 0 or 1. Thus, to accurately determine loss between the actual and predicted values, it needs to compare the actual value (0 or 1) with the probability that the input aligns with that category (p(i) = probability that the category is 1; 1 — p(i) = probability that the category is 0)

### Categorical Cross-Entropy Loss

In cases where the number of classes is greater than two, we utilize categorical cross-entropy — this follows a very similar process to binary cross-entropy.

```math

CCE Loss = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m} y_{ij}*\log(p_i)
```

The most commonly used loss function in image classification is cross-entropy loss/log loss (binary for classification between 2 classes and sparse categorical for 3 or more), where the model outputs a vector of probabilities that the input image belongs to each of the pre-set categories. This output is then compared to the actual output, represented by a vector of equal size, where the correct category has a probability of 1 and all others have a probability of 0.

### IoU loss function

Intuitively, IoU loss maximizes the coincidence between the predicted box and the ground truth box. 

```math

L IoU = -ln\frac{I}{U}
```

`I` is the intersection area of two boxes, `U` is the union area of two boxes.

From the formula point of view, when calculating the area of intersection and union of two boxes, four variables of measuring each box are used at the same time. Therefore, this loss function regards a box as a whole for training, and can get more accurate predicted box. In addition, regardless of the scale of the ground truth, IoU is normalized to [0, 1], which can prevent the model from focusing too much on large objects and ignoring small ones. Use of IoU loss not only makes the location more accurate, but also speeds up the convergence rate.
