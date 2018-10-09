import numpy as np
from random import shuffle

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
  N, D = X.shape
  C = W.shape[1]
  score = np.zeros([N, C])
  for i in range(N):
      score[i] = np.dot(X[i, :], W)
      correct_score = score[i, y[i]]
      loss += (-correct_score + np.log(np.sum(np.exp(score[i]))))
      for j in range(C):
          if j == y[i]:
              dW[:, j] -= X[i, :]
          dW[:, j] += np.exp(score[i, j]) / np.sum(np.exp(score[i])) * X[i, :]
  loss /= N
  dW /= N
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D = X.shape
  C = W.shape[1]
  score = X.dot(W)
  correct_score = score[np.arange(N), y]
  loss = -np.sum(correct_score) + np.sum(np.log(np.sum(np.exp(score), axis=1)))
  I = np.zeros([N, C])
  I[np.arange(N), y] -= 1
  grad1 = np.exp(score) / np.tile(np.sum(np.exp(score), axis=1).reshape(N, 1), [1, C])
  I += grad1
  dW = np.dot(X.T, I)
  loss /= N
  dW /= N
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

