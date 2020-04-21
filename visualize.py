import argparse
from pca import PCA
from prob_pca import ProbabilisticPCA
import mnist
import numpy as np


if __name__ == "__main__":
    # TODO: refactor methods into a superclass
    
    methods = {"pca": PCA, "prob_pca": ProbabilisticPCA}
    train_images, train_labels, test_images, test_labels = mnist.load()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="The method for dimensionality reduction", default="pca")    
    args = parser.parse_args()
    dim_reducer = methods[args.method](train_images)
    dim_reducer.visualize(test_images[:80], test_labels[0:80])
    mnist.show(dim_reducer.get_principal_components())
