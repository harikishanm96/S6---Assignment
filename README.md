# S6---Assignment

## PART - 1

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

# Learning rate: Error graph for the learning rate changes from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 
![Loss = 0 1](https://github.com/harikishanm96/S6---Assignment/assets/53985105/6e6a4388-67af-4274-ab63-acf67c56e092)
![Loss = 0 2](https://github.com/harikishanm96/S6---Assignment/assets/53985105/3113fce6-afb0-4b54-bc73-c0c5aa42f302)
![Loss = 0 5](https://github.com/harikishanm96/S6---Assignment/assets/53985105/700b71ed-d884-4659-a98d-baac7c3c1a7b)
![Loss = 0 8](https://github.com/harikishanm96/S6---Assignment/assets/53985105/f61892ff-d50e-4327-b90c-7dd730839dbc)
![Loss = 1 0](https://github.com/harikishanm96/S6---Assignment/assets/53985105/a3c11f47-3144-41dc-936d-f5092796d513)
![Loss = 2 0](https://github.com/harikishanm96/S6---Assignment/assets/53985105/f445d772-6ab4-43d0-b720-2b7756c036f8)

## PART - 2

Model:

The network, contains seven Convolution layer, two Max pooling layer two Transition layer followed by Avg pooling layer.

2-D convolutional layers are used. As activation function is rectified linear units (ReLUs in short). Max pooling of kernel size 2x2 to reduce channel size into half.

The forward() pass defines the way to compute our output using the given layers and functions.

x.view(-1, 10) The view method returns a tensor with the same data as the self tensor (which means that the returned tensor has the same number of elements), but with a different shape. First parameter represent the batch_size in our case batch_size is 128 if you don't know the batch_size pass -1 tensor.view method will take care of batch_size for you. Second parameter is the column or the number of neurons you want. Look at the code below.

SGD optimizer with lr=0.08, momentum=0.9 is used.

20 epochs for training of the network.

Final validation accuracy is 99.37%












