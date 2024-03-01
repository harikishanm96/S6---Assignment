# S6---Assignment

![image](https://github.com/harikishanm96/S6---Assignment/assets/53985105/d335718a-e6dd-4e5c-a82a-29d0bebc20cd)

1. Network architecture and forward pass:
   Define the network with two input neurons, two hidden neurons and two output neurons
   Compute the activations of hidden neurons using weighted sums of the input neurons
   Apply the sigmoid activation function to each hidden neuron
   Compute the output of the network as weighted sums of the hidden neuron activations
   Apply the sigmoid activation function to each output neuron

2. Define the error function:
   Define the total error as the sum of errors from individual output neurons
   Define individual errors for each output neuron using squared error

3. Calculate the gradients with respect to weights:
   Use the chain rule to compute gradients of the total error with respect to weights
   Express each derivative in terms of smaller derivatives, eventually reaching down to the weights

4. Summarize the gradients:
   Summarize the gradients for each weight and activation in the network
   Express the gradients in terms of activations and weights
   
5. Partial derivatives with respect to weights:
   Compute the partial derivatives of the total error with respect to each weight.
   Use the chain rule to break down each derivative into smaller derivatives, involving activations and weights.

6. Simplify the expressions:
   Simplify the expressions by substituting known values and reorganizing terms.
   Express the gradients in terms of known quantities such as activations, weights, and input values.
   
These steps are crucial for understanding how the gradients of the error function with respect to the weights are calculated, which is essential for updating the weights during the training process using optimization algorithms like gradient descent.
