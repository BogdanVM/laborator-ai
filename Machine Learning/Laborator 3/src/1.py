import matplotlib.pyplot as plt
import numpy as np
from Knn_classifier import Knn_Classifier

def compute_accuracy_l2(classifier, test_images, test_labels, neighbors):
    predicted_labels = classifier.classify_images(test_images, neighbors)
    return classifier.accuracy(predicted_labels, test_labels)

def compute_accuracy_l1(classifier, test_images, test_labels, neighbors):
    predicted_labels = classifier.classify_images(test_images, neighbors, 'l1')
    return classifier.accuracy(predicted_labels, test_labels)

train_images = np.loadtxt('data/train_images.txt') # incarcam imaginile
train_labels = np.loadtxt('data/train_labels.txt', 'int') # incarcam etichetele avand

test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int') 

knn_classifier = Knn_Classifier(train_images, train_labels)



accuracies_l2 = np.loadtxt('acuratete_l2.txt')
accuracies_l1 = np.loadtxt('acuratete_l1.txt')
num_neighbors = np.arange(1, 11, 2)

plt.plot(num_neighbors, accuracies_l2, label = 'L1')
plt.plot(num_neighbors, accuracies_l1, label = 'L2')
plt.show()

#accuracy_1 = compute_accuracy_l1(knn_classifier, test_images, test_labels, 1)
#accuracy_3 = compute_accuracy_l1(knn_classifier, test_images, test_labels, 3)
#accuracy_5 = compute_accuracy_l1(knn_classifier, test_images, test_labels, 5)
#accuracy_7 = compute_accuracy_l1(knn_classifier, test_images, test_labels, 7)
#accuracy_9 = compute_accuracy_l1(knn_classifier, test_images, test_labels, 9)
##
#np.savetxt('acuratete_l1.txt', 
#           [accuracy_1, accuracy_3, accuracy_5, accuracy_7, accuracy_9])
#

#print(knn_classifier.accuracy(predicted_labels, test_labels) * 100, '%')
#
#np.savetxt('predictii_3nn_l2_mnist.txt', predicted_labels)

#predicted_labels = np.loadtxt('predictii_3nn_l2_mnist.txt')
#print(knn_classifier.confusion_matrix(predicted_labels, test_labels))