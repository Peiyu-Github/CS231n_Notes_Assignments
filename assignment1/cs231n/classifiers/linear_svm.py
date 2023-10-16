from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the cross-entropy loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                
                # add the analytical gradient change
                dW[:,j] += X[i].T   
                dW[:,y[i]] -= X[i].T 
                # why X[i].T? 
                # Recall X is of size [N, D], thus X[i] is of size [1, D], X[i].T becomes [D,1]. 
                # And W is of size [D, C], so is dW. Thus, at each classification level, dw[class] if of size [D,1]. 
                # Therefore X[i].T. 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train # add it

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W   # add it


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. simply count the number of classes that didn’t meet the desired margin, i.e. margin > 0, 
    # and then the data vector xi scaled by this number is the gradient.
    
    # 2. when the desired margin is met, 
    # dW[True Class] -= X
    # dW[Not True Class] += X
    
    # 3. remember to divide the loss and gradient by num_train 
    # 4. and don't forget the regularization loss and change
    
    # refer to the analytical gradient change above

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    # Vectorized means implementing the computation without loops.              #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    C = W.shape[1]  # W: [D, C]
    N = X.shape[0]  # X: [N, D]
    delta = 1

    scores = np.dot(X, W)    # (N, C)
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
    # don't forget the regularization loss! 
    loss += 0.5 * reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
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
    
    # backpropate the gradient to the parameters
    dW += np.dot(X.T, dscores)
    # and don't forget the regularization gradient! it changes weight too. 
    dW += reg * W
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


    