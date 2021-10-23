import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # Dataset setting
    dataset = pd.read_csv('./data/fashion-mnist_train.csv')
    testset = pd.read_csv('./data/fashion-mnist_test.csv')

    labels_train = dataset['label']
    images_train = dataset.iloc[:, 1:]

    # combining train and validation
    x_train = images_train.values
    y_train = labels_train.values

    labels_test = testset['label']
    images_test = testset.iloc[:, 1:]
    x_test = images_test.values
    y_test = labels_test.values

    # Learning
    print("Start")
    svm_mnist = svm.LinearSVC(C=0.0001, max_iter=10000)
    # svm_mnist = svm.SVC(C=0.0001, kernel='linear', max_iter=10000)
    svm_mnist.fit(x_train, y_train)

    # Visualization
    label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plot = ConfusionMatrixDisplay.from_estimator(svm_mnist, x_test, y_test, display_labels=label, normalize='true')
    plot.ax_.set_title('Confusion Matrix')
    print(plot.confusion_matrix)
    plt.show()


