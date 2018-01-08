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
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #Calculate y for each
  #Divide over total 
  #Do the log
  for i in range(num_train):
    val = X[i].dot(W) #Calculate the score
    
  #Do shifting
    val -= np.max(val)
    sfm = np.exp(val) / np.sum(np.exp(val)) #Do Softmax
  #Do Loss and do Gradient 
    for j in xrange(num_class):
      if j == y[i]:
        loss += -np.log(sfm[j])
      p = np.exp(val[j]) / np.sum(np.exp(val))
      dW[:, j] += (p - (j == y[i])) * X.T[:, i]

        
  #Compute Average  
  loss /= num_train
  dW /= num_train

  #Do Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  
  ###Smooth approach using Lambda 
  # sum_val = np.sum(np.exp(val))
  #p = lambda k : np.exp(val[k]) / sum_val -- generates an iterable
  # loss += -np.log(p(y[i]))
  #Check my Implementations
  #num_classes = W.shape[1]
  #for i in xrange(num_train):
  #      scores = X[i].dot(W)
  #      scores -= np.max(scores)
  #      softmax = np.exp(scores) / np.sum(np.exp(scores),keepdims = True)
  #      for j in xrange(num_classes):
  #          if j == y[i]:
  #              loss += -(np.log(softmax[j]))
  #              dW[:,j] += (softmax[j]-1) * X[i].T
  #          else:
  #              dW[:,j] += (softmax[j]) * X[i].T
  #              
  #loss = loss / num_train          
  #loss += reg*np.sum(W*W)
  #dW = dW / num_train
  #dW = dW + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  val = X.dot(W)
  val -= np.max(val, axis=1)[:, np.newaxis]
  sfm = np.exp(val) / np.sum(np.exp(val), axis=1)[:, np.newaxis]
  loss = -np.sum(np.log(sfm[np.arange(num_train), y]))
  #Implement dW
  sfm[range(num_train), y] -=1 
  dW = X.T.dot(sfm)
  

  #Compute Average  
  loss /= num_train
  dW /= num_train

  #Do Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

