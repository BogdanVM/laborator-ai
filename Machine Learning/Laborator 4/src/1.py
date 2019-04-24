from sklearn import preprocessing
from sklearn import svm
import sklearn.metrics as sm
import numpy as np


def normalize_data(train_data, test_data, type = None):
    if type is None:
        print('Type is None')
        return train_data, test_data

    if type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)

        scaled_train = scaler.transform(train_data)
        scaled_test = scaler.transform(test_data)
        return scaled_train, scaled_test

    if type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_data)

        scaled_train = scaler.transform(train_data)
        scaled_test = scaler.transform(test_data)
        return scaled_train, scaled_test

    if type == 'l1':
        train_data /= np.sum(np.abs(train_data), axis = 1, keepdims = True)
        test_data /= np.sum(np.abs(test_data), axis = 1, keepdims = True)

    elif type == 'l2':
        train_data /= np.sqrt(np.sum(train_data ** 2, axis = 1, keepdims = True))
        test_data /= np.sqrt(np.sum(test_data ** 2, axis = 1, keepdims = True))

    return train_data, test_data


def svm_classifier(train_data, train_labels, test_data, C):
    # definim modelul
    model = svm.SVC(C, 'linear')

    # antrenam modelul
    model.fit(train_data, train_labels)

    # facem predictia pe train_labels si test_labels
    predicted_train_labels = model.predict(train_data)
    predicted_test_labels = model.predict(test_data)

    return predicted_train_labels, predicted_test_labels


def main():
    train_images = np.loadtxt('data/train_images.txt')
    train_labels = np.loadtxt('data/train_labels.txt', 'int32')

    test_images = np.loadtxt('data/test_images.txt')
    test_labels = np.loadtxt('data/test_labels.txt', 'int32')

    scaled_train_data, scaled_test_data = normalize_data(train_images, test_images, 'standard')
    predicted_train_labels, predicted_test_labels = svm_classifier(scaled_train_data, train_labels, scaled_test_data, 1)

    print(sm.accuracy_score(predicted_train_labels, train_labels))
    print(sm.accuracy_score(predicted_test_labels, test_labels))


if __name__ == '__main__':
    main()
