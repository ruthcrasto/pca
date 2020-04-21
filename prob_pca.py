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
    X = X[:1000,:]
    self.original_dims = X.shape
    self.data = self.reshape_data(X)
    d = self.data.shape[1]
    # hyperparameters
    self.num_iters = 10
    self.embed_dim = k
    # initialize parameters
    self.sigma = np.random.randn()
    self.W = np.random.randn(d,k)
    self.train()

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
    """ Determine principal components using EM.
    """
    X_bar = self.data - np.mean(self.data, axis=0)
    N,d = X_bar.shape
    for _ in range(self.num_iters):
        M = self.W.T @ self.W + self.sigma * np.eye(self.embed_dim)
        M_inv = np.linalg.inv(M)

        # E step
        Z = X_bar @ self.W @ M_inv.T  # N x k
        temp = np.expand_dims(Z,2) @ np.expand_dims(Z,1) # N x k x k
        Z_ = self.sigma * M_inv + temp # N x k x k
      
        # M step
        self.W = np.sum(np.expand_dims(X_bar,2) @ np.expand_dims(Z,1), axis=0)
        self.W = self.W @ np.linalg.inv(np.sum(Z_, axis=0))
        self.sigma = np.sum(np.linalg.norm(X_bar,axis=1)**2)
        self.sigma -= np.sum(np.diagonal(2 * Z @ self.W.T @ X_bar.T))
        self.sigma += np.sum(np.trace((Z_ @ self.W.T @ self.W).T))
        self.sigma /= N * d

  def get_principal_components(self):
    if len(self.original_dims) == 3:
      return self.W.T.reshape(-1, self.original_dims[1], self.original_dims[2])
    return self.W.T

  def project_all(self, X):
    projector = self.W
    return X @ self.W

  def visualize(self, X, labels=None):
    """ Visualize input 2D embeddings.

    Arguments:
      X: a (N' x d) data matrix (NumPy array)
      labels: a (N' x 1) matrix of numeric labels (NumPy array) - optional
    """
    X = self.reshape_data(X)
    projections = self.project_all(X)
    x = projections[:,0]
    y = projections[:,1]
    colors = list(mcolors.get_named_colors_mapping().keys())[-19:-9]
    
    if labels is not None:
      for i in range(len(labels)):
        plt.scatter(x[i],y[i],color=colors[int(labels[i])])
        plt.annotate(labels[i], (x[i],y[i]))
    else:
      plt.scatter(x,y)
    plt.show()
    
