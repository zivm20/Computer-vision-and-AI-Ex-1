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
    extra = []
    for i in range(X.shape[0]):
      # Softmax Loss
      s = [sum([X[i,j]*W[j,k] for j in range(X.shape[1])]) for k in range(W.shape[1])]
      
      expS = np.array([np.exp(s_k) for s_k in s])
      
      preds = expS/np.sum(expS)
      #print("s: ",s)
      extra.append(preds)
      #print("expS: ",expS)
      #print("np.sum(expS): ",np.sum(expS))
      #print("preds: ",preds)
     
      
      for c in range(W.shape[1]):
        err = (preds[c]-1) if c == y[i] else preds[c]
        
        dW[:,c] += (X[i,:]*err/X.shape[0]) 

      loss += -np.log(preds[y[i]])/X.shape[0]
      #print("loss: ", -np.log(preds[y[i]])/X.shape[0])
      #print("___________________")
    # Regularization
    for c in range(W.shape[1]):
      for d in range(W.shape[0]):
        loss += (reg/2)*(W[d,c]**2)
        dW[d,c]+= reg*W[d,c]
    
    

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
    
    s = np.exp(X@W)
    target = np.eye(W.shape[1])[y]
    preds = s/s.sum(axis=1)[:,None]
    loss = np.mean(-np.sum(np.log(preds)*target,axis=1)) + np.sum((reg/2)*W**2)
    dW = reg*W + (X.T@(preds-target))/X.shape[0]
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
