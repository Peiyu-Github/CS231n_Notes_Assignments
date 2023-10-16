from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # reshape the input into a vector of dimension D = d_1 * ... * d_k
    x_shaped = np.reshape(x, (x.shape[0], -1))
    # compute the output (scores after per layer)
    out = np.dot(x_shaped, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx = np.dot(dout, w.T)  # (N, D)
    dx = dx.reshape(x.shape) # (N, d1, ..., d_k)
    
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout) # (D, N) * (N, M) -> (D, M)
    
    db = np.sum(dout, axis=0, keepdims=False) # (M,)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x) 
    # Note: 
    # np.max(arr) returns a single value, while
    # np.maximum(arr) returns an array of the same size as arr

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    mask = x > 0    # return a matrix of bool elements
    dx = dout * mask

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(scores, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - scores: Input data, of shape (N, C) where scores[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for scores[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dscores: Gradient of the loss with respect to scores
    """
    loss, dscores = 0.0, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # refer to cs231n/classifiers/linear_svm.py
    N = scores.shape[0]  # x: (N, C)
    delta = 1

    # to get the correct scores for each test point, in size of (N, ): 
    correct_scores = scores[np.arange(N), y]  # (N, )
    # although it may look the same for scores[np.arange(N), y] and scores[:, y] since scores is of size [N, C],
    # scores[:, y] would return an array of size [N, y.shape] where y.shape = [N, ]. 
    # numpy objects has a flexible and different logic than list. 
    
    margins = np.maximum(scores - correct_scores.reshape(N, 1) + delta, 0)  # (N, C)
    # reshape is essential because size (N, ) and (N, 1) is totally different in numpy
    margins[np.arange(N), y] = 0
    
    # don't forget to do the division by # trainning data (N)
    loss += np.sum(margins) / N
    # forget the regularization loss... we're doing the trainning loss here
    
    # starting from socres = np.dot(X, W), propogate to dW = np.dot(X.T, dscores)
    # answer to this TODO part is still kinda confusing... have to fighure it out anyway
    dscores = np.zeros_like(scores)  # (N, C)
    dscores[margins > 0] = 1  
    dscores[np.arange(N), y] -= np.sum(dscores, axis=1)   #  (N, 1) = (N, 1)
    
    # the following division by the number of data has two benefits: 
    # 1. Normalization the batch size
    # 2. Let the gradient changes smaller (sort of regularization) 
    # https://stackoverflow.com/questions/65275522/why-is-softmax-classifier-gradient-divided-by-batch-size-cs231n
    dscores /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dscores


def softmax_loss(scores, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - scores: Input data, of shape (N, C) where scores[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for scores[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dscores: Gradient of the loss with respect to scores
    """
    loss, dscores = 0.0, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # refer to cs231n/classifiers/softmax.py
    N = scores.shape[0]
    
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # get Li for each data point, with a total size of [300x1]
    correct_logprobs = -np.log(probs[range(N),y])

    # compute the loss: average cross-entropy loss
    # forget the regularization loss... we're doing the trainning loss here
    loss = np.sum(correct_logprobs)/N
    
    dscores = probs
    # âˆ‚Li/âˆ‚fk = pk âˆ’ ðŸ™(yi=k). -1 only at its true class
    # loss decreases while scores increase
    dscores[range(N),y] -= 1 
    # the following division by the number of data has two benefits: 
    # 1. Normalization the batch size
    # 2. Let the gradient changes smaller (sort of regularization) 
    # https://stackoverflow.com/questions/65275522/why-is-softmax-classifier-gradient-divided-by-batch-size-cs231n
    dscores /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dscores
