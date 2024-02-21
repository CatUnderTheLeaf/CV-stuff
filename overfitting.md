### What is overfitting?

The word overfitting refers to a model that models the training data too well. Instead of learning the genral distribution of the data, the model learns the expected output for every data point.

This is the same a memorizing the answers to a maths quizz instead of knowing the formulas. Because of this, the model cannot generalize. Everything is all good as long as you are in familiar territory, but as soon as you step outside, you’re lost.

### How to detect overfitting?

As stated above, overfitting is characterized by the inability of the model to generalize. To test this ability, a simple method consists in splitting the dataset into two parts: the training set and the test set. When selecting models, you might want to split the dataset in three, I explain why here.

1. The training set represents about 80% of the available data, and is used to train the model (you don’t say?!).
2. The test set consists of the remaining 20% of the dataset, and is used to test the accuracy of the model on data it has never seen before.

With this split we can check the performance of the model on each set to gain insight on how the training process is going, and spot overfitting when it happens. This table shows the different cases.
|                     | Low training error |   High training error  |
| ------------------- | ------------------ | ---------------------- |
|__Low testing error__| Model is learning  |     Error in code?     |
|__High testing error__|    Overfitting    | Model is not learning  |

Overfitting can be seen as the difference between the training and testing error.

> Note: for this technique to work, you need to make sure both parts are representative of your data. A good practice is to shuffle the order of the dataset before splitting.

### How to prevent overfitting - Model & Data

#### 1. Gather more data

You model can only store so much information. This means that the more training data you feed it, the less likely it is to overfit. The reason is that, as you add more data, the model becomes unable to overfit all the samples, and is forced to generalize to make progress.

Collecting more examples should be the first step in every data science task, as more data will result in an increased accuracy of the model, while reducing the chance of overfitting.

The more data you get, the less likely the model is to overfit.

#### 2. Data augmentation & Noise

Collecting more data is a tedious and expensive process. If you can’t do it, you should try to make your data appear as if it was more diverse. To do that, use data augmentation techniques so that each time a sample is processed by the model, it’s slightly different from the previous time. This will make it harder for the model to learn parameters for each sample.

Each iteration sees as different variation of the original sample.

Another good practice is to add noise:
- To the input: This serves the same purpose as data augmentation, but will also work toward making the model robust to natural perturbations it could encounter in the wild.
- To the output: Again, this will make the training more diversified.
> Note: In both cases, you need to make sure that the magnitude of the noise is not too great. Otherwise, you could end up respectively drowning the information of the input in the noise_, or_ make the output incorrect. Both will hinder the training process.

#### 3. Simplify the model

If, even with all the data you now have, your model still manages to overfit your training dataset, it may be that the model is too powerful. You could then try to reduce the complexity of the model.

As stated previously, a model can only overfit that much data. By progressively reducing its complexity — # of estimators in a random forest, # of parameters in a neural network etc. — you can make the model simple enough that it doesn’t overfit, but complex enough to learn from your data. To do that, it’s convenient to look at the error on both datasets depending on the model complexity.

This also has the advantage of making the model lighter, train faster and run faster.

### How to prevent overfitting - Training Process

#### 4. Early Termination

In most cases, the model starts by learning a correct distribution of the data, and, at some point, starts to overfit the data. By identifying the moment where this shift occurs, you can stop the learning process before the overfitting happens. As before, this is done by looking at the training error over time.

When the testing error starts to increase, it’s time to stop!

### How to prevent overfitting — Regularization

Regularization is a process of constraining the learning of the model to reduce overfitting. It can take many different forms, and we will see a couple of them.

#### 5. L1 and L2 regularization

One of the most powerful and well-known technique of regularization is to add a penalty to the loss function:

- The L1 penalty aims to minimize the absolute value of the weights
- The L2 penalty aims to minimize the squared magnitude of the weights.

With the penalty, the model is forced to make compromises on its weights, as it can no longer make them arbitrarily large. This makes the model more general, which helps combat overfitting.

The L1 penalty has the added advantage that it enforces feature selection, which means that it has a tendency to set to 0 the less useful parameters. This helps identify the most relevant features in a dataset. The downside is that it is often not as computationally efficient as the L2 penalty.

Another possibility is to add noise to the parameters during the training, which helps generalization.

#### 6. For Deep Learning: Dropout and Dropconnect
This extremely effective technique is specific to Deep Learning, as it relies on the fact that neural networks process the information from one layer to the next. The idea is to randomly deactivate either neurons (dropout) or connections (dropconnect) during the training.

This forces the network to become redundant, as it can no longer rely on specific neurons or connections to extract specific features. Once the training is done, all neurons and connections are restored. It has been shown that this technique is somewhat equivalent to having an ensemble approach, which favorises generalization, thus reducing overfitting.
