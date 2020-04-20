import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class PCA:
  """ An implementation of linear PCA by Ruth Crasto.
  """
  
  def __init__(self, X):
    """ 
    Arguments:
        X: a (N x d) or (N x w x h) data matrix (NumPy array)
    """
    assert X is not None
    self.original_dims = X.shape
    self.data = self.reshape_data(X)
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
    """ Determine principal components using SVD.
    """
    # Center data
    mean = np.mean(self.data, axis=0)
    X_bar = self.data - mean
    # Get principal components
    U, _, _ = np.linalg.svd(X_bar.T @ X_bar)
    self.all_eigenvecs = U

  def get_principal_components(self):
    """ Return all principal components reshaped to dimensions of original dataset.
    """
    if len(self.original_dims) == 3:
      return self.all_eigenvecs.T.reshape(-1, self.original_dims[1], self.original_dims[2])
    return self.all_eigenvecs.T
  
  def project(self, x, k=2):
    """ Project a single vector onto subspace spanned by top k principal components.

    Arguments:
      x: a d-dimensional vector (NumPy array)
      k: dimensionality of embedding (int) - default is 2

    Returns:
      k-dimensional projection of x (NumPy array)
    """
    projector = self.all_eigenvecs[:,:k] # top k
    return projector.T @ x

  def project_all(self, X, k=2):
    """ Project a collection of vectors onto subspace spanned by top k principal components.

    Arguments:
      X: a (N' x d) matrix (NumPy array)
      k: dimensionality of embedding (int) - default is 2

    Returns:
      (N' x k) matrix whose rows are the projections (NumPy array)
    """
    projector = self.all_eigenvecs[:,:k]
    return X @ projector

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
