# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:39:31 2019

@author: Student
"""
import numpy as np

class NaiveBayes:
    def __init__(self, num_bins, max_value):
        self.num_bins = num_bins
        self.bins = np.linspace(start = 0, stop = max_value, num=self.num_bins)
    
    def values_to_bins(self, x):
        x = np.digitize(x, self.bins)
        x -= 1
        return x
    
    def fit(self, train_images, train_labels):
        self.train_images = self.values_to_bins(train_images)
        self.train_labels = train_labels
        self.num_classes = train_labels.max() + 1
        
        prob_classes = []
        for c in range(self.num_classes):
            prob = np.sum(train_labels == c) / len(train_labels)
            prob_classes.append(prob)
        
        print(prob_classes)
        self.prob_classes = prob_classes
        
        self.num_features = self.train_images.shape[1]
        position_bin_class_probs = np.zeros((self.num_features, self.num_bins,
                                             self.num_classes))
        
        for pos in range(self.num_features):
            for idx_bin in range(self.num_bins):
                for class_id in range(self.num_classes):
                    train_images_class = self.train_images[self.train_labels == class_id]
                    
                    position_bin_class_probs[pos, idx_bin, class_id] = np.sum(
                            train_images_class[:, pos] == idx_bin) / len(train_images_class)
        
        self.position_bin_class_probs = position_bin_class_probs + 1e-10
    
    def predict_one(self, test_image):
        probs = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            probs[c] += np.log(self.prob_classes[c])
            
            for pos in range(self.num_features):
                probs[c] += np.log(self.position_bin_class_probs[pos, test_image[pos], c])
        
        return np.argmax(probs)
    
    def predict(self, test_images):
        test_images = self.values_to_bins(test_images)
        predicted_labels = []
        
        for i in range(test_images.shape[0]):
            predicted_labels.append(self.predict_one(test_images[i]))
        
        return predicted_labels
    
    def score(self, predicted_labels, labels):
        return np.mean(predicted_labels == labels)