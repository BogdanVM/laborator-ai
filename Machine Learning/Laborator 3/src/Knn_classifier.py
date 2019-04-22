import numpy as np

class Knn_Classifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    
    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'):
        
        if metric == 'l2':
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis = 1))
            
            sorted_index = np.argsort(distances)
            sorted_index = sorted_index[:num_neighbors]
            nearest_labels = self.train_labels[sorted_index]
            
            h = np.bincount(nearest_labels)
            return np.argmax(h)
        else:
            distances = np.sum(np.abs(self.train_images - test_image), axis = 1)
            
            sorted_index = np.argsort(distances)
            sorted_index = sorted_index[:num_neighbors]
            nearest_labels = self.train_labels[sorted_index]
            
            h = np.bincount(nearest_labels)
            return np.argmax(h)
    
    def classify_images(self, test_images, num_neighbors = 3, metric = 'l2'):
        num_images = test_images.shape[0]
        predicted_labels = np.zeros((num_images))
        
        for i in range(num_images):
            predicted_labels[i] = self.classify_image(test_images[i], 
                            num_neighbors, metric)
        
        return predicted_labels
    
    def accuracy(self, predicted_labels, labels):
        return np.mean(predicted_labels == labels)
    
    def confusion_matrix(self, predicted_labels, labels):
        num_classes = labels.max() + 1
        matrix = np.zeros((num_classes, num_classes))
        
        for i in range(len(predicted_labels)):
            matrix[labels[i], int(predicted_labels[i])] += 1
        
        return matrix