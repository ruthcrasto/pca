import numpy as np
import os
import gzip, struct, array
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parse_labels(filename):
    with gzip.open(filename, 'rb') as fh:
        magic, num_data = struct.unpack(">II", fh.read(8))
        return np.array(array.array("B", fh.read()), dtype=np.uint8)

def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
        magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
        return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

def load():
    train_images = parse_images('mnist_data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('mnist_data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('mnist_data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('mnist_data/t10k-labels-idx1-ubyte.gz')
    return train_images, train_labels, test_images, test_labels

def show(data, count=9):
    count = min(count, data.shape[0])
    rows = np.floor(np.sqrt(count))
    cols = np.ceil(count // rows)
    fig = plt.figure(figsize=(2*rows,2*cols))
    for i in range(1,count+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(data[i-1])
    plt.show()

def plot2D(X, labels=None):
    assert X.shape[1] == 2
    x = X[:,0]
    y = X[:,1]
    colors = list(mcolors.get_named_colors_mapping().keys())[-19:-9]
    
    if labels is not None:
      for i in range(len(labels)):
        plt.scatter(x[i],y[i],color=colors[int(labels[i])])
        plt.annotate(labels[i], (x[i],y[i]))
    else:
      plt.scatter(x,y)
    plt.show()

if __name__ == "__main__":
    data,_,_,_ = load()
    show(data)
