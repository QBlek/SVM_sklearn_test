import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    # Dataset setting
    dataset = pd.read_csv('./data/fashion-mnist_train.csv')
    testset = pd.read_csv('./data/fashion-mnist_test.csv')

    labels_train = dataset['label']
    images_train = dataset.iloc[:, 1:]

    x_temp = images_train.values
    y_temp = labels_train.values

    index_list = np.arange(0, dataset.shape[0])
    train_length = int(0.8 * len(index_list))
    train_indices = index_list[:train_length]
    validation_indices = index_list[train_length:]

    x_train = x_temp[train_indices]
    y_train = y_temp[train_indices]
    x_validation = x_temp[validation_indices]
    y_validation = y_temp[validation_indices]

    labels_test = testset['label']
    images_test = testset.iloc[:, 1:]
    x_test = images_test.values
    y_test = labels_test.values

    # Learning
    acc_result = []
    for _ in range(4):
        acc_result.append([])

    C = 0.0001

    for i in range(9):
        print("Fitting.", i+1)

        # svm_mnist = svm.SVC(C=C, kernel='linear', max_iter=1000)
        # svm_mnist = svm.SVC(C=C, kernel='linear', max_iter=10000)
        # svm_mnist = svm.LinearSVC(C=C, max_iter=2000)
        svm_mnist = svm.LinearSVC(C=C, max_iter=10000)


        svm_mnist.fit(x_train, y_train)

        # print(svm_mnist.n_supports)
        # In LinearSVC, there is no toolkit for counting number of support vectors

        acc_result[0].append(str(C))
        """
        acc_result[1].append(svm_mnist.score(x_train, y_train))
        acc_result[2].append(svm_mnist.score(x_validation, y_validation))
        acc_result[3].append(svm_mnist.score(x_test, y_test))
        """
        acc_result[1].append(accuracy_score(svm_mnist.predict(x_train), y_train))
        acc_result[2].append(accuracy_score(svm_mnist.predict(x_validation), y_validation))
        acc_result[3].append(accuracy_score(svm_mnist.predict(x_test), y_test))

        # np.insert(train_result, i, accuracy_score(svm_mnist.predict(x_train), y_train))
        C *= 10



    # Visualization
    plt.title("SVM_mnist")
    plt.plot(acc_result[0], acc_result[1], label='Train')
    plt.plot(acc_result[0], acc_result[2], label='Validation')
    plt.plot(acc_result[0], acc_result[3], label='Test')
    plt.xlabel('Value of C')
    # plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
