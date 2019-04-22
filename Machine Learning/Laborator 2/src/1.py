# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:38:32 2019

@author: Student
"""

from skimage import io # pentru afisarea imaginii
import numpy as np

from NaiveBayes import *

train_images = np.loadtxt('data/train_images.txt') # incarcam imaginile
train_labels = np.loadtxt('data/train_labels.txt', 'int') # incarcam etichetele avand
 # tipul de date int
image = train_images[0, :] # prima imagine
image = np.reshape(image, (28, 28))

naive_bayes = NaiveBayes(5, 255)
naive_bayes.fit(train_images, train_labels)

test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int')

predicted_labels = naive_bayes.predict(test_images)
print(naive_bayes.score(predicted_labels, test_labels))
