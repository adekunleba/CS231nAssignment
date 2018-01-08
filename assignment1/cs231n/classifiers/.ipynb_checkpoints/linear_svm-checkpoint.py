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
  dW = np.zeros(W.shape) # initialize the gradient as zero
    
  '''
  simple Note on computing Gradient.
  Understand that for an example Xi of dim 1 * 3072 is multiplied by a Weight for all class
  W = 10 * 3072 and the result is 10 * 1 (10 * 3072 multiply by 3072 * 1) score for all class
  '''

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    diff_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        diff_count += 1
        #Update the loss when margin is greater than 0
        dW[:, j] += X[i]
        loss += margin
    
    #Update for the correct class
    dW[:, y[i]] += -diff_count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  delta = 1.0
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_score = scores[np.arange(num_train), y]
  #To get the max of errors, Using Broadcasting approach will subtract two arrays
  # of different dimension. np.axis just increase by 1 the dimension of an array
  # passed to it ###
  margin = np.maximum(0, scores - correct_score[:, np.newaxis] + delta)
  #Sum all non_zeros and add to loss
  #Convert all index where correct score is to 0
  margin[np.arange(num_train), y] = 0
  loss = np.sum(margin)

  loss /= num_train # get mean
  loss += 0.5 * reg * np.sum(W * W) #Add Regularization
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #Create a mask for X
  X_mask = np.zeros(margin.shape)
  #Replace all index where margins is greater than 0 with 1 in X_mask
  X_mask[margin > 0] = 1
  #Get the total incorrect classifications for each datapoints.
  incorrect_counts = np.sum(X_mask, axis=1)
  #Replace all places where y existed in X_mask with -incorrect_counts
  X_mask[np.arange(num_train), y] = -incorrect_counts
  #Multiply with X gives your gradient
  dW = X.T.dot(X_mask) #returns an arran 1032 * 10 same as Weight
  #Average out Weight
  dW /= num_train #Perform an element wise division over dW
  #Regularize dW
  dW += reg*W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
