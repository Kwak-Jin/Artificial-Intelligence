# Deep Learning
### DL and ML
1. DL: finds and learns the characteristics of learning data by itself
   1. Object detection
   2. Semantic segmentation
   3. Pose estimation
   4. Anomaly detection
   - ANN
   - DNN
   - CNN
2. ML: need manual feature extraction/selection for the model
   1. Supervised
   2. Unsupervised
   3. Semi-supervised
   4. Reinforcement
## Theory
### Perceptron
   1. Binary linear classifier
   $$z= w^Tx + b, x: input, w: weight, b: bias $$
   2. Decision boundary $$y = 0 \ if \ \ z<= 0 \ else  \ 1$$
   3. Linear Decision Boundary $\rightarrow$ Activation Function(Step) $\rightarrow$ Classification

### MLP/Forward Network
   1. MLP: Multi-Layer Perceptron
      1. Started from XOR gate (Linear Decision boundary is not suitable for XOR gate boundary)
      2. XOR gate can be splitted into $$y = (x_{1}x_{2})'(x_{1}+x_{2}) = AND(NAND(x_{1},x_{2}),OR(x_{1},x_{2}))$$
      3. MLP: Non-linear classifier / Also use non-linear activation function (ReLU, tanh, sigmoid)
   2. Forward Network 
      1. The structure and operation of traditional neural networks where data moves in one direction—from input to output.

### Activation functions
   What is activation function?   

   - Transform input signal of a node in a neural network into an output signal that is then passed on to the next layer.
   - Without activation function, neural networks would be restricted to modeling only linear relationships between I/O. e.g. Matrix multiplication
   - Enables networks to learn **non-linear relationships by introducing non-linear behaviors** through activation functions. Thus, increases flexibility and power of neural networks to model complex data
   - Activating neurons ensures that **backpropagation works effectively**, as these functions influence the calculated gradients during training. Thus, gradient descent optimization.
   - **Decision boundaries**: Help define complex decision boundaries.

   Why gradient of activation function important?

   - Crucial role in **training** process
   - **Weight update** using derivatives
   - **Learning signal** direction & magnitude of the updates
   - **Gradient flow**

   <img src="img/activation_function.png" alt="img.png" style="zoom:67%;" />

   **Types of activation functions**   

   1. Sigmoid: output limit [0,1]
      - vanishing gradient
      - not zero centered
      - not used in practice
      - sigmoid is used in logistic regression
      - simple gradient function

   2. tanh
      - vanishing gradient

   3. ReLU: Rectified linear Unit
      - Not zero centered

### Loss functions

   - Minimize classification/ regression error
   - Avoid over-fitting in training

   <img src="img/model_complexity_and_error.png" alt="img.png" style="zoom:67%;" />

   - Loss function calculates error between prediction and ground truth
   - Average error from all datasets

   <img src="img/model_complexity_and_error2.png" alt="img.png" style="zoom:80%;" />

   - Common loss functions
     - Mean Squared Error(MSE): used for regression tasks, calculates average squared difference between predicted and actual values
     - Binary cross-entropy: Binary classification problems, it measures the difference between two probability distributions
     - Categorical **cross entropy**: multi-class classification problems, it extends binary cross-entropy to handle multiple classes
     
     **Cross entropy**
     
     - model predicts a probability distribution across multiple classes
     
     $$\text{Categorical Cross-Entropy} = -\sum_{i=1}^{C} y_i \log(p_i)$$ 
     
     - **Probabilistic Interpretation**: Cross-entropy provides a measure based on probabilities, making it suitable for models that output probabilities (like those using the softmax function in multi-class problems or the sigmoid function in binary cases).
     - **Gradient Descent Optimization**: It works well with gradient-based optimization algorithms, providing smooth gradients that lead to effective weight updates during training.
     - **Sensitivity**: Cross-entropy is sensitive to misclassification. It penalizes wrong predictions heavily, especially when the model is confident about its incorrect predictions (e.g., predicting a probability close to 1 for the wrong class).
     - **Encourages Higher Confidence**: By minimizing cross-entropy, models are encouraged to assign high probabilities to true classes while assigning low probabilities to false classes, thereby improving overall classification performance.

   - **Regularization**

     Regularization is a technique used in machine learning and neural networks to **prevent overfitting**, which occurs when a model learns the training data too well and performs poorly on unseen data. Regularization introduces additional information or constraints on the model to ensure that it generalizes better to new data. Here are some common regularization techniques

     **Promotes Simplicity**: Encourages simpler models with fewer complex relationships, which are often more interpretable and robust.

     - Curse of dimensionality (Interpolation과 같은 상황에서 실제 경향보다 차수가 훨씬 높을 때, 발생하는 문제)

       - Overfitting: high model complexity/ high variance
       - Underfitting: small datasets, over-generalizing/ high bias

     - L1 regularization
       - Adds the absolute value of the weights to the loss function as a penalty term. 
       - tendency to produce sparse weights
     - L2 regularization
       - Adds the square of the weights to the loss function produce small weights.
       - L2 regularization is fine with SGD, momentum optimization, and Nesterov momentum optimization, but not with Adam and its variants.
       - To use Adam with weight decay, then use AdamW instead
     - Dropout
       - A technique where a random subset of neurons is "dropped out" (set to zero) during each training iteration. This prevents the model from becoming overly reliant on any specific set of features. 
       - It effectively creates a different architecture for each training iteration, helping to improve generalization. 
       - Since dropout is only active during training, comparing the training loss and the validation loss can be misleading.  
       - Large network는 overfit에 취약하다. 따라서 training 도중, network의 random node deactivation를 일부러 시켜서 남은 노드들의 영향력을 확인할 수 있다.
     - Early stopping
       - A form of regularization that involves monitoring the validation loss during training. 
       - When the validation loss starts to increase after initially decreasing, training is stopped to prevent overfitting.
     - Data augmentation

       - Generating additional training data through various transformations of the existing data (e.g., rotations, translations, scaling for images)
       - 사진이면 돌려도 보고, 키워도 보고, 위치도 바꿔보고 등등 짜집어서 실제 test 혹은 validation에서 강인한 인지를 할 수 있게 만드는 기술이다.
       - It artificially increases the size and diversity of the training set, helping the model learn more robust features

     - **Batch normalization**
       Although primarily used to speed up training and improve convergence, batch normalization can also have a regularizing effect by adding some noise to the inputs of each layer during training
       
       Complex and deep layers have tendency to have vanishing gradient problem (Layers near input remains relatively unchanged)
       
       Explosive gradient: larger error gradients sequentially accumulate causing unstable update

   - **Batch**

     - Dividing the whole dataset into number, 데이터셋의 일부
     - Iteration 당 사용되는 양
     - 전체 데이터셋을 한번에 처리하는 것이 아닌 부분 부분으로 나눠 관리 가능한 크기로 처리함
     - Instead of using the entire dataset to compute the gradients for weight updates during training, which can be computationally expensive and inefficient, the dataset is divided into smaller, manageable sections, or batches.
     - Common batch sizes can range from 1 (in stochastic gradient descent, where each sample is processed individually) to larger sizes such as 32, 64, 128, or 256, depending on the dataset and available **computational resources**
     - **훈련 과정**: 훈련 시 일반적으로 다음과 같은 단계가 진행됩니다:
       - 일정량의 입력 데이터를 포함하는 배치가 모델에 입력됩니다.
       - 모델은 해당 입력을 기반으로 예측을 수행합니다.
       - 손실 함수가 예측값과 실제 레이블 간의 오류를 계산합니다.
       - 손실에 대한 모델 파라미터의 그래디언트(기울기)가 계산됩니다.
       - 이 그래디언트를 사용하여 모델의 파라미터(가중치)가 업데이트됩니다. 일반적으로 경량 하강법(stochastic gradient descent) 같은 최적화 알고리즘을 사용합니다
     - Large batch = larger learning rate
     - Small batch = small learning rate 
     - Usually, small batch size perform better

   - Gradient Descent

     - **전체 배치 경량법(Batch Gradient Descent)**: 전체 데이터셋을 이용해 그래디언트를 계산합니다. 대규모 데이터셋에서는 느리고 메모리 집약적일 수 있습니다.
     - **확률적 경량 하강법(Stochastic Gradient Descent, SGD)**: 배치 크기로 1을 사용하여 각 개별 샘플에 대해 가중치를 업데이트합니다. 이 방법은 잡음이 더 많지만, 더 자주 업데이트하기 때문에 빠른 속도를 제공합니다.
     - **미니배치 경량 하강법(Mini-Batch Gradient Descent)**: 소규모 고정 배치(예: 32 또는 64 샘플)를 사용하여 가중치를 업데이트합니다. 이 방법은 일반적인 방식으로, 전체 배치와 확률적 경량 하강법의 장점을 결합한 것입니다.

### Backpropagation

   [Video for backpropagation 1](https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=11s)

   [Video for backpropagation 2](https://www.youtube.com/watch?v=tIeHLnjs5U8)

   <img src="img/back_prop.png" alt="img.png" style="zoom:90%;" />

### Optimization

   - Finding optimal W that minimizes loss function $$L(W)$$
   - Gradient descent: follows opposite gradient of loss function
     - update parameter W until it reaches the minima
     - control step rate(learning rate)
     - $$W_{k+1} = W_{k}- \eta {\partial L(W)\over\partial W}$$
   - Examples of gradient descent methods
     - SGD(Stochastic gradient descent)
     - mini-batch gradient descent
     - adagrad: achieves this correction by scaling down the gradient vector along the steepest dimension. runs the risk of slowing down a bit too fast and never converging to the global optimum
     - momentum
     - Nesterov Accelerated Gradient
     - RMSProp: 최근 iteration에서의 gradient를 쌓아서 adagrad보다 안정적인 optimizer
     - Adam(adaptive moment estimation): Momentum optimizer + RMSProp
     - AdaMax
   - Issues
     - Convergence: there may be multiple minima
       - depending on initial value, learning rate
     - How to calculate the gradient of loss function?
       - Use **backpropagation**

### Convolution

### Performance metrics
#### Confusion matrix

#### recall
recall/ sensitivity/ true positive rate(TPR)    
$$\text{Recall} ={TP \over TP+FN}$$ 
 
#### precision
the accuracy of the positive predictions
$$\text{Precision} ={TP\over TP+FP}$$

#### The Precision/Recall Trade-off
Recall과 Precision은 Trade-off 관계     
<img src="img/precision_recall_curve.png" alt="img.png" style="zoom:80%;" />    
Recall만 보는 것/ Precision만 보는 것은 바람직하지 않은 방식이다.   

### Misc

1. **Iteration**:
   - An iteration refers to one update of the model’s parameters during training. It corresponds to the number of batches processed. For example, if you use a mini-batch gradient descent with a batch size of 32 and your training dataset contains 1,000 samples, you would have 31 iterations (with the last one possibly processing fewer samples if they don't evenly fit).
2. **Epoch**:
   - An epoch refers to one complete pass through the entire training dataset. During an epoch, the model sees and processes every training example once. For the same example above, with a dataset of 1,000 samples and a batch size of 32, it would take 31 iterations to complete one epoch.

**Deep Learning Training Process**

1. **Initialization**: 
   - Initialize model parameters (weights and biases) randomly or using specific initialization techniques.
2. **Epoch Loop**:
   - For each epoch:
     1. **Shuffling**: Shuffle the training data to promote a better learning pattern and avoid overfitting.
     2. Batch Loop:
        - For each batch of data within the epoch:
          1. **Forward Pass**: Input the batch into the model and compute predictions using the current weights.
          2. **Loss Calculation**: Compute the loss by comparing the predictions to the true labels using a loss function.
          3. **Backward Pass**: Perform backpropagation to calculate gradients of the loss with respect to model parameters.
          4. **Parameter Update**: Update the model parameters using an optimization algorithm (like SGD, Adam, etc.) based on the computed gradients.
     3. **End of Batch Loop**: At the end of all batches in the epoch, the model has seen all training samples once.
3. **Validation**:
   - After each epoch, evaluate the model on a validation dataset to monitor its performance and check for overfitting.
4. **Stopping Criterion**:
   - Repeat the epoch loop for a predetermined number of epochs or until a stopping criterion is met (like early stopping if validation performance is not improving).
5. **Final Model**: 
   - Once training is complete, save the final trained model for future use or inference.