from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


# Note: I added the train and predict modules. 

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # refer to the notes on Dec 28, 2022 & Jan 5, 2023
        # generate weights and values with required elements
        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        b1 = np.zeros((hidden_dim,))
        W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b2 = np.zeros((num_classes,))
        
        # store the parameters
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def train(self, X, y, learning_rate, reg, num_iters=1000):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        
        ############################################################################
        # TODO: Implement the trainning of model.                                  #
        ############################################################################
        
        num_examples = X.shape[0] 
        W1 = self.params['W1'] 
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        for i in range(num_iters):
        
            # evaluate class scores, (N, C)
            hidden_layer = np.maximum(0, np.dot(X, W1) + b1) # ReLU activation
            scores = np.dot(hidden_layer, W2) + b2
            
            # compute the class probabilities (SoftMax)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (N, C)
            
            # compute the loss: average cross-entropy loss and regularization
            correct_logprobs = -np.log(probs[range(num_examples),y])
            data_loss = np.sum(correct_logprobs)/num_examples
            reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
            loss = data_loss + reg_loss
            
            if i % 100 == 0:
                print ("iteration %d: loss %f" % (i, loss))
    
            # compute the gradient on scores
            dscores = probs
            dscores[range(num_examples),y] -= 1
            dscores /= num_examples
            
            # backpropate the gradient to the parameters
            # first backprop into parameters W2 and b2
            dW2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            # next backprop into hidden layer
            dhidden = np.dot(dscores, W2.T)
            # backprop the ReLU non-linearity
            dhidden[hidden_layer <= 0] = 0
            # finally into W1,b1
            dW1 = np.dot(X.T, dhidden)
            db1 = np.sum(dhidden, axis=0, keepdims=True)
            
            # add regularization gradient contribution
            dW2 += reg * W2
            dW1 += reg * W1
            
            # perform a parameter update
            W1 += -learning_rate * dW1
            b1 += -learning_rate * db1
            W2 += -learning_rate * dW2
            b2 += -learning_rate * db2
            

        # store the parameters and end the trainning
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################



    def predict(self, X_to_be_predicted):
        
        W1 = self.params['W1'] 
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        # evaluate class scores, (N, K)
        hidden_layer = np.maximum(0, np.dot(X_to_be_predicted, W1) + b1) # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2
        
        # generate predicted calss
        predicted_class = np.argmax(scores, axis=1)
        
        return predicted_class
        
        
        
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        W1 = self.params['W1'] 
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        reg = self.reg
        
        fc1, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(fc1, W2, b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Compute loss
        data_loss, dscores = softmax_loss(scores, y)

        reg_loss = 0.0

        # Don't regularize bias terms
        reg_loss += np.sum(np.square(W1))
        reg_loss += np.sum(np.square(W2))

        loss = data_loss + 0.5 * reg * reg_loss

        # Compute grads
        dfc1, dW2, db2 = affine_backward(dscores, cache2)
        _, dW1, db1 = affine_relu_backward(dfc1, cache1)

        # Do not forget the gradient of regularization term
        grads['W1'] = dW1 + reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + reg * W2
        grads['b2'] = db2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads