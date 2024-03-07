# Optimization algorithms

**Gradient-based optimizers**:
- [Gradient Descent](#gradient-descent)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)

**Momentum-Based Optimizers**:
- [Momentum SGD](#momentum-sgd)
- [Nesterov Accelerated Gradient (NAG)](#nesterov-accelerated-gradient-nag)

**Adaptive Learning Rate Optimizers**:
- [Adaptive Gradient (Adagrad)](#adaptive-gradient-adagrad)
- [RMSprop](#rmsprop)
- [Adam (Adaptive Moment Estimation)](#adam-adaptive-moment-estimation)

**Population-Based Optimizers**:
- [Genetic Algorithms](#genetic-algorithms)
- [Particle Swarm Optimization (PSO)](#particle-swarm-optimization-pso)

**Quasi-Newton Optimizers**:
- [Limited Memory Broyden Fletcher Goldfarb Shanno (L-BFGS)](#limited-memory-broyden-fletcher-goldfarb-shanno-l-bfgs)
- [Newton’s Method](#newtons-method)

[What is the Learning Rate?](#learning-rate)

**General guidelines that may help you make an informed decision**:
- If you are **new** to neural network training or want a **simple and reliable** optimizer, you can start with SGD, or mini-batch gradient descent.
- If you want to **speed up** your training process and **avoid getting stuck** in local minima or plateaus, you can try momentum-based optimizers such as momentum or NAG.
- If you want to **fine-tune your learning rate** for each parameter and achieve more **stable results**, you can try adaptive learning rate optimizers such as Adagrad, RMSprop, or Adam.
- If you have a **small-scale problem** or want to achieve **high accuracy and efficiency**, you can try second-order optimizers such as Newton’s method or L-BFGS.
- If you want to explore a **diverse and complex solution space**, you can try population-based optimizers such as genetic algorithms or PSO.

| Algorithm | Benefits | Disadvantages |
| --------- | -------- | ------------- |
| [Gradient Descent](#gradient-descent) | + Generality: can be applied to almost any function<br> + Simplicity and Ease of Implementation: Despite its effectiveness, GD remains relatively simple to understand and implement | - Size of the learning rate: if too small, it might take a long time to reach the bottom. If it’s too large, you might overshoot the lowest point<br> - Risk of Local Minima and Saddle Points: In complex models, SGD can get stuck in local minima or saddle points, especially in high-dimensional spaces. |
| [Stochastic Gradient Descent](#stochastic-gradient-descent) | + Scalability: Since it updates parameters using only a single data point (or a small batch) at a time, it is much less memory-intensive than algorithms requiring the entire dataset for each update<br> + Speed: By using only a small subset of data at a time, SGD can make rapid progress in reducing the loss, especially for large datasets<br> + Escape from Local Minima: The randomness helps SGD to potentially escape local minima, a common problem in complex optimization problems<br> + Online Learning: SGD is well-suited for online learning, where the model needs to be updated as new data comes in, due to its ability to update the model incrementally<br> + Handling Non-Static Datasets: For datasets that change over time, SGD’s incremental update approach can adjust to these changes more effectively than batch methods<br> + Improved Generalization: By updating the model frequently with a high degree of variance, SGD can often lead to models that generalize better on unseen data. This is because the algorithm is less likely to overfit to the noise in the training data | - Choosing the Right Learning Rate: Same as Gradient Descent<br> - Dealing with Noisy Updates:  The stochastic nature of SGD leads to noisy updates, which can cause the algorithm to be less stable and take longer to converge<br> - Risk of Local Minima and Saddle Points: Same as Gradient Descent<br> - Sensitivity to Feature Scaling: SGD is sensitive to the scale of the features, and having features on different scales can make the optimization process inefficient<br> - Hyperparameter Tuning: SGD requires careful tuning of hyperparameters, not just the learning rate but also parameters like momentum and the size of the mini-batch<br> - Overfitting: Like any machine learning algorithm, there’s a risk of overfitting, where the model performs well on training data but poorly on unseen data |
| [Mini-Batch Gradient Descent](#mini-batch-gradient-descent) | + More stable convergence: Instead of using the entire dataset (as in batch GD) or a single sample (as in SGD), it uses a mini-batch of samples, which reduces the variance of the parameter updates<br> + More computationally efficient: It can also take advantage of optimized matrix operations | - same? |
| [Momentum SGD](#momentum-sgd) | + Faster convergence and reduces oscillations: It does this by adding a fraction of the previous update vector to the current update | - same? |
| [Nesterov Accelerated Gradient (NAG)](#nesterov-accelerated-gradient-nag) | + Speed up convergence and improve the performance of the algorithm: makes a more informed update by calculating the gradient of the future approximate position of the parameters | - same? |
| [Adaptive Gradient (Adagrad)](#adaptive-gradient-adagrad) | + Adapts the learning rate to each parameter, giving parameters that are updated more frequently a lower learning rate | - Radically diminishing learning rates |
| [RMSprop](#rmsprop) | + Addresses radically diminishing learning rates of Adagrad. It uses a moving average of squared gradients to normalize the gradient | - same? |
| [Adam (Adaptive Moment Estimation)](#adam-adaptive-moment-estimation) | + Dealing with Sparse Data: is particularly effective when working with data that leads to sparse gradients<br> + Training Large-Scale Models: Its adaptive learning rate helps navigate the complex optimization landscapes of such models efficiently<br> + Achieving Rapid Convergence: adaptive learning guarantees a faster convergence compared to its rival SGD<br> + For Online and Batch Training | - Tuning Hyperparameters: choosing an appropriate initial learning rate is still crucial. A too-high learning rate may lead to instability, while too low a rate can slow down the training process<br> - Handling Noisy Data and Outliers: extreme outliers or highly noisy datasets might impact its performance<br> - Choice of Loss Function: The efficiency of Adam can vary with different loss functions<br> - Computational Considerations: requires more memory than simple gradient descent algorithms because it maintains moving averages for each parameter

### Gradient Descent

In machine learning, Gradient Descent is a star player. It’s an optimization algorithm used to minimize a function by iteratively moving towards the steepest descent as defined by the negative of the gradient. Imagine you’re at the top of a mountain, and your goal is to reach the lowest point. Gradient Descent helps you find the best path down the hill.

The beauty of Gradient Descent is its simplicity and elegance. Here’s how it works, you start with a random point on the function you’re trying to minimize, for example a random starting point on the mountain. Then, you calculate the gradient (slope) of the function at that point. In the mountain analogy, this is like looking around you to find the steepest slope. Once you know the direction, you take a step downhill in that direction, and then you calculate the gradient again. Repeat this process until you reach the bottom.

The size of each step is determined by the learning rate. However, if the learning rate is too small, it might take a long time to reach the bottom. If it’s too large, you might overshoot the lowest point. Finding the right balance is key to the success of the algorithm.

One of the most appealing aspects of Gradient Descent is its generality. It can be applied to almost any function, especially those where an analytical solution is not feasible. This makes it incredibly versatile in solving various types of problems in machine learning, from simple linear regression to complex neural networks.

### Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) adds a twist to the traditional gradient descent approach. The term ‘stochastic’ refers to a system or process that is linked with a random probability. Therefore, this randomness is introduced in the way the gradient is calculated, which significantly alters its behavior and efficiency compared to standard gradient descent.

In traditional batch gradient descent, you calculate the gradient of the loss function with respect to the parameters for the entire training set. As you can imagine, for large datasets, this can be quite computationally intensive and time-consuming. This is where SGD comes into play. Instead of using the entire dataset to calculate the gradient, SGD randomly selects just one data point (or a few data points) to compute the gradient in each iteration.

Think of this process as if you were again descending a mountain, but this time in thick fog with limited visibility. Rather than viewing the entire landscape to decide your next step, you make your decision based on where your foot lands next. This step is small and random, but it’s repeated many times, each time adjusting your path slightly in response to the immediate terrain under your feet.

This stochastic nature of the algorithm provides several benefits:

- Speed: By using only a small subset of data at a time, SGD can make rapid progress in reducing the loss, especially for large datasets.
- Escape from Local Minima: The randomness helps SGD to potentially escape local minima, a common problem in complex optimization problems.
- Online Learning: SGD is well-suited for online learning, where the model needs to be updated as new data comes in, due to its ability to update the model incrementally.
  
However, the stochastic nature also introduces variability in the path to convergence. The algorithm doesn’t smoothly descend towards the minimum; rather, it takes a more zigzag path, which can sometimes make the convergence process appear erratic.

#### Why Choose SGD?

- Scalability: One of the primary advantages of SGD is its efficiency in handling large-scale data. Since it updates parameters using only a single data point (or a small batch) at a time, it is much less memory-intensive than algorithms requiring the entire dataset for each update.
- Speed: By frequently updating the model parameters, SGD can converge more quickly to a good solution, especially in cases where the dataset is enormous.
- Online Learning: SGD’s ability to update the model incrementally makes it well-suited for online learning, where the model needs to adapt continuously as new data arrives.
- Handling Non-Static Datasets: For datasets that change over time, SGD’s incremental update approach can adjust to these changes more effectively than batch methods.
- Overcoming Challenges of Local Minima: The stochastic nature of SGD helps it to potentially escape local minima, a significant challenge in many optimization problems. The random fluctuations allow the algorithm to explore a broader range of the solution space.
- General Applicability: SGD can be applied to a wide range of problems and is not limited to specific types of models. This general applicability makes it a versatile tool in the machine learning toolbox.
- Simplicity and Ease of Implementation: Despite its effectiveness, SGD remains relatively simple to understand and implement. This ease of use is particularly appealing for those new to machine learning.
- Improved Generalization: By updating the model frequently with a high degree of variance, SGD can often lead to models that generalize better on unseen data. This is because the algorithm is less likely to overfit to the noise in the training data.
- Compatibility with Advanced Techniques: SGD is compatible with a variety of enhancements and extensions, such as momentum, learning rate scheduling, and adaptive learning rate methods like Adam, which further improve its performance and versatility.

#### Disadvantages and challenges

- Choosing the Right Learning Rate: Selecting an appropriate learning rate is crucial for SGD. If it’s too high, the algorithm may diverge; if it’s too low, it might take too long to converge or get stuck in local minima. Use a learning rate schedule or adaptive learning rate methods. Techniques like learning rate annealing, where the learning rate decreases over time, can help strike the right balance.
- Dealing with Noisy Updates:  The stochastic nature of SGD leads to noisy updates, which can cause the algorithm to be less stable and take longer to converge. Implement mini-batch SGD, where the gradient is computed on a small subset of the data rather than a single data point. This approach can reduce the variance in the updates.
- Risk of Local Minima and Saddle Points: In complex models, SGD can get stuck in local minima or saddle points, especially in high-dimensional spaces. Use techniques like momentum or Nesterov accelerated gradients to help the algorithm navigate through flat regions and escape local minima.
- Sensitivity to Feature Scaling SGD is sensitive to the scale of the features, and having features on different scales can make the optimization process inefficient. Normalize or standardize the input features so that they are on a similar scale. This practice can significantly improve the performance of SGD.
- Hyperparameter Tuning: SGD requires careful tuning of hyperparameters, not just the learning rate but also parameters like momentum and the size of the mini-batch. Utilize grid search, random search, or more advanced methods like Bayesian optimization to find the optimal set of hyperparameters.
- Overfitting: Like any machine learning algorithm, there’s a risk of overfitting, where the model performs well on training data but poorly on unseen data. Use regularization techniques such as L1 or L2 regularization, and validate the model using a hold-out set or cross-validation.

### Mini-Batch Gradient Descent
This is a blend of batch gradient descent and stochastic gradient descent. Instead of using the entire dataset (as in batch GD) or a single sample (as in SGD), it uses a mini-batch of samples.
It reduces the variance of the parameter updates, which can lead to more stable convergence. It can also take advantage of optimized matrix operations, which makes it more computationally efficient.

### Momentum SGD
Momentum is an approach that helps accelerate SGD in the relevant direction and dampens oscillations. It does this by adding a fraction of the previous update vector to the current update.
It helps in faster convergence and reduces oscillations. It is particularly useful for navigating the ravines of the cost function, where the surface curves much more steeply in one dimension than in another.

### Nesterov Accelerated Gradient (NAG)
A variant of momentum SGD, Nesterov momentum is a technique that makes a more informed update by calculating the gradient of the future approximate position of the parameters.
It can speed up convergence and improve the performance of the algorithm, particularly in the context of convex functions.

### Adaptive Gradient (Adagrad)
Adagrad adapts the learning rate to each parameter, giving parameters that are updated more frequently a lower learning rate.
It’s particularly useful for dealing with sparse data and is well-suited for problems where data is scarce or features have very different frequencies.

### RMSprop
RMSprop (Root Mean Square Propagation) modifies Adagrad to address its radically diminishing learning rates. It uses a moving average of squared gradients to normalize the gradient.
It works well in online and non-stationary settings and has been found to be an effective and practical optimization algorithm for neural networks.

### Adam (Adaptive Moment Estimation)
Adam combines ideas from both Momentum and RMSprop. It is often considered as a default optimizer due to its effectiveness in a wide range of applications. It’s particularly good at solving problems with noisy or sparse gradients.

At its core, Adam is designed to adapt to the characteristics of the data. It does this by maintaining individual learning rates for each parameter in your model. These rates are adjusted as the training progresses, based on the data it encounters.

Think of it as if you’re driving a car over different terrains. In some places, you accelerate (when the path is clear and straight), and in others, you decelerate (when the path gets twisty or rough). Adam modifies its speed (the learning rate_ based on the road (the gradient’s nature) ahead.

Indeed, the algorithm can remember the previous actions (gradients), and the new actions are guided by the previous ones. Therefore, Adams keeps track of the gradients from previous steps, allowing it to make informed adjustments to the parameters. This memory isn’t just a simple average; it’s a sophisticated combination of recent and past gradient information, giving more weight to the recent.

Moreover, in areas where the gradient (the slope of the loss function) changes rapidly or unpredictably, Adam takes smaller, more cautious steps. This helps avoid overshooting the minimum. Instead, in areas where the gradient changes slowly or predictably, it takes larger steps. This adaptability is key to Adam’s efficiency, as it navigates the loss landscape more intelligently than algorithms with a fixed step size.

#### Why Opt for Adam?

- Dealing with Sparse Data - Adam is particularly effective when working with data that leads to sparse gradients. This situation is common in models with large embedding layers or when dealing with text data in natural language processing tasks.
- Training Large-Scale Models - Adam is well-suited for training models with a large number of parameters, such as deep neural networks. Its adaptive learning rate helps navigate the complex optimization landscapes of such models efficiently. However, this is not always the case, as we can see when we applied Adam to Linear Regression.
 - Achieving Rapid Convergence - When we don’t have much time for the convergence to happen, Adam comes to help. This is thanks to its adaptive learning, which guarantees a faster convergence compared to its rival SGD.
 - For Online and Batch Training - It’s versatile enough to be used in both online learning scenarios (where the model is updated continuously as new data arrives) and batch learning.

#### Addressing Adam’s Limitations

 - Tuning Hyperparameters - While Adam is less sensitive to learning rate changes compared to other optimizers, choosing an appropriate initial learning rate is still crucial. A too-high learning rate may lead to instability, while too low a rate can slow down the training process.
The default values of β1​ and β2​ (typically 0.9 and 0.999, respectively) work well in most cases, but in some scenarios, adjusting them can yield better results.
 - Handling Noisy Data and Outliers - While Adam is generally robust to noisy data, extreme outliers or highly noisy datasets might impact its performance. Preprocessing data to remove or diminish the impact of outliers can be beneficial.
 - Choice of Loss Function - The efficiency of Adam can vary with different loss functions. Make sure that the loss function resonates with the problem you are solving, and experiment with a few of them to see which one works best.
 - Computational Considerations - Adam typically requires more memory than simple gradient descent algorithms because it maintains moving averages for each parameter. This should be considered when working with very large models or limited computational resources.

### Genetic Algorithms

Genetic algorithms are inspired by the process of natural selection. They create a population of potential solutions (parameter sets) and iteratively evolve them over generations by selecting, recombining, and mutating individuals.

### Particle Swarm Optimization (PSO)

PSO simulates the behavior of particles in a search space. Each particle represents a potential solution, and particles adjust their positions based on their own best-known solution and the collective best-known solution among all particles.

### Limited Memory Broyden Fletcher Goldfarb Shanno (L-BFGS)
L-BFGS is an approximation of Newton’s method that avoids the computational cost of calculating and inverting the full Hessian matrix. Instead, it maintains a limited-memory approximation of the Hessian. L-BFGS is more memory-efficient and still converges relatively quickly.

### Newton’s Method
Newton’s method is a classic second-order optimizer that uses the Hessian matrix, which is the matrix of second derivatives of the loss function, to find the optimal parameters. It can converge very quickly, but it requires computing and inverting the Hessian matrix, which can be very costly and impractical for large-scale problems.

------------------

### Learning Rate

One of the most crucial hyperparameters in the Stochastic Gradient Descent (SGD) algorithm is the learning rate. This parameter can significantly impact the performance and convergence of the model. Understanding and choosing the right learning rate is a vital step in effectively employing SGD.

At this point you should have an idea of what learning rate is, but let’s better define it for clarity. The learning rate in SGD determines the size of the steps the algorithm takes towards the minimum of the loss function. It’s a scalar that scales the gradient, dictating how much the weights in the model should be adjusted during each update. If you visualize the loss function as a valley, the learning rate decides how big a step you take with each iteration as you walk down the valley.

#### Too High Learning Rate
If the learning rate is too high, the steps taken might be too large. This can lead to overshooting the minimum, causing the algorithm to diverge or oscillate wildly without finding a stable point.
Think of it as taking leaps in the valley and possibly jumping over the lowest point back and forth.

#### Too Low Learning Rate
On the other hand, a very low learning rate leads to extremely small steps. While this might sound safe, it significantly slows down the convergence process.
In a worst-case scenario, the algorithm might get stuck in a local minimum or even stop improving before reaching the minimum.
Imagine moving so slowly down the valley that you either get stuck or it takes an impractically long time to reach the bottom.

#### Finding the Right Balance
The ideal learning rate is neither too high nor too low but strikes a balance, allowing the algorithm to converge efficiently to the global minimum.
Typically, the learning rate is chosen through experimentation and is often set to decrease over time. This approach is called learning rate annealing or scheduling.

#### Learning Rate Scheduling
Learning rate scheduling involves adjusting the learning rate over time. Common strategies include:

- Time-Based Decay: The learning rate decreases over each update.
- Step Decay: Reduce the learning rate by some factor after a certain number of epochs.
- Exponential Decay: Decrease the learning rate exponentially.
- Adaptive Learning Rate: Methods like AdaGrad, RMSProp, and Adam adjust the learning rate automatically during training.

-----
Each of these variants has its own strengths and is suited for specific types of problems. Their development reflects the ongoing effort in the machine learning community to refine and enhance optimization algorithms to achieve better and faster results. Understanding these variants and their appropriate applications is crucial for anyone looking to delve deeper into machine learning optimization techniques.
