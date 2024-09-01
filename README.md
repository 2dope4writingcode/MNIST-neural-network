# MNIST-neural-network
This repository contains a simple feedforward neural network implemented in Python for classifying handwritten digits from the MNIST dataset. The network architecture consists of a single hidden layer and uses the backpropagation algorithm for training.

## Features
- Input layer with 784 neurons (one for each pixel in the 28x28 MNIST images).
- A hidden layer with 20 neurons using the sigmoid activation function.
- Output layer with 10 neurons, each representing a digit from 0 to 9.
- Mean Squared Error (MSE) as the loss function.
- Backpropagation for training the model over multiple epochs.

## Mathematical Explanation

### Neural Network Architecture

- **Input Layer**: The input is a vector **x** of size 784x1 (flattened 28x28 image).
- **Hidden Layer**:
  - The weighted input to the hidden layer **z_h** is calculated as:
    **z_h = W_ih * x + b_ih**
  - The activation **h** of the hidden layer is then:
    **h = sigma(z_h) = 1 / (1 + exp(-z_h))**
    where **sigma(·)** is the sigmoid function applied element-wise.
- **Output Layer**:
  - The weighted input to the output layer **z_o** is calculated as:
    **z_o = W_ho * h + b_ho**
  - The activation **o** of the output layer is:
    **o = sigma(z_o) = 1 / (1 + exp(-z_o))**

### Loss Function

The loss function used is the Mean Squared Error (MSE), defined as:

**MSE = (1/n) * sum((o_i - l_i)^2)**

where:
- **n** is the number of output neurons.
- **o_i** is the predicted output for the i-th neuron.
- **l_i** is the actual label for the i-th neuron.

### Backpropagation

During backpropagation, the gradients are calculated and used to update the network's weights and biases:

1. **Output Layer Gradient**:
   **delta_o = o - l**
   where **l** is the label vector.

2. **Hidden Layer Gradient**:
   **delta_h = (W_ho^T * delta_o) * (h * (1 - h))**
   where **\* ** denotes element-wise multiplication.

3. **Weight and Bias Updates**:
   - For weights between the input and hidden layers:
     **W_ih = W_ih - eta * (delta_h * x^T)**
   - For biases in the hidden layer:
     **b_ih = b_ih - eta * delta_h**
   - For weights between the hidden and output layers:
     **W_ho = W_ho - eta * (delta_o * h^T)**
   - For biases in the output layer:
     **b_ho = b_ho - eta * delta_o**
   where **eta** is the learning rate.

## Potential Extensions

Here are some ways to extend this simple neural network:

- **Change the Network Architecture**: Increase the number of neurons in the hidden layer or add more hidden layers to create a deeper network.
- **Use a Different Activation Function**: Experiment with ReLU, Leaky ReLU, or Tanh instead of Sigmoid.
- **Use a Different Loss Function**: Try Cross-Entropy Loss, which is more common for classification problems.
- **Add Dropout**: Implement dropout to prevent overfitting, especially if you increase the network's complexity.
- **Implement Regularization**: Add L2 regularization to the loss function to penalize large weights.
- **Train on a Subset of the Dataset**: Train on a smaller subset of the MNIST dataset to speed up training during experimentation.

## License

This project is licensed under the MIT License.

