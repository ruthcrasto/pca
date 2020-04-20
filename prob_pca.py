import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ProbabilisticPCA:
  """ An implementation of Probabilistic PCA by Ruth Crasto.
  """
  
  def __init__(self, X, k=2):
    """ Initialize parameters.

    Arguments:
        X: a (N x d) or (N x w x h) data matrix (NumPy array)
    """
    self.data = reshape_data(X)
    # hyperparameters
    self.num_iters = 5
    self.embed_dim = k
    # initialize parameters
    self.sigma_squared = ... # scalar
    self.W = ... # d x k
    self.mu = ... # d x 1

  def reshape_data(self, X):
    """ Reshape 3D image data to 2D data.

    Arguments:
      X: a (N x d) or (N x w x h) data matrix (NumPy array)

    Returns:
      data matrix reshaped to (N x w*h) if X was 3-dimensional, otherwise X unchanged
    """
    if len(self.original_dims) == 3:
      return X.reshape(X.shape[0],-1)
    return X

  def train(self):
    """ Determine principal components using SVD.
    """
    X_centered = self.data - np.mean(self.data, axis=0)
    N = X_centered.shape[0]
    for _ in range(self.num_iters):
        M = self.W.T @ self.W + self.sigma * np.eye(self.embed_dim)
        M_inv = np.linalg.inv(M)

        # E step
        Z = M_inv @ W.T @ X_centered # N x k
        temp = np.dot(Z.expand_dims(2), Z.expand_dims(1))
        #temp = np.zeros(N, self.embed_dim, self.embed_dim)
        #for i in range(N):
            #temp[i] = Z[i].reshape(-1,1) @ Z[i].reshape(1,-1)
        Z_ = self.sigma * M_inv +  temp # N x k x k

        # M step
        self.W = np.sum(np.dot(X_centered.expand_dims(2), Z.expand_dims(1)), axis=0)
        

