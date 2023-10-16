from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_examples = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_examples):
        scores = np.dot(X[i].T, W)
        
        # a trick to prevent numeric instability: shift the socres! 
        # refer to Notes of Study on Dec 7, 2022 (Wed) - Dec 8, 2022 (Thu), or CS231n Linear Classification
        # at the part of numeric stability
        shift_scores = scores - np.max(scores)
        loss_i = - shift_scores[y[i]] + np.log(np.sum(np.exp(shift_scores)))
        loss += loss_i
        
        for j in xrange(num_classes):
            softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:,j] += (-1 + softmax_output) *X[i] 
            else: 
                dW[:,j] += softmax_output *X[i] 

    loss /= num_examples
    dW /= num_examples
    
    loss +=  0.5* reg * np.sum(W * W)
    dW += reg* W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Refer to Notes on DEc 28, 2022 & Jan 5, 2023
    
    # compute class scores for a linear classifier
    scores = np.dot(X, W)
    
    num_examples = X.shape[0] # 300
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # get Li for each data point, with a total size of [300x1]
    correct_logprobs = -np.log(probs[range(num_examples),y])

    # compute the loss: average cross-entropy loss and regularization
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    
    dscores = probs
    # ∂Li/∂fk = pk − 𝟙(yi=k). -1 only at its true class
    # loss decreases while scores increase
    dscores[range(num_examples),y] -= 1 
    # the following division by the number of data has two benefits: 
    # 1. Normalization the batch size
    # 2. Let the gradient changes smaller (sort of regularization) 
    # https://stackoverflow.com/questions/65275522/why-is-softmax-classifier-gradient-divided-by-batch-size-cs231n
    dscores /= num_examples

    # since scores = np.dot(X, W) and now we have dscores, then by backpropagation: 
    dW = np.dot(X.T, dscores)
    dW += reg*W # remember the regularization loss 12λklWk,l2 and its derivative? 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
