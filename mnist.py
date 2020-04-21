import numpy as np
import os
import gzip, struct, array
import matplotlib.pyplot as plt

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
