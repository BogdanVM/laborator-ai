from skimage import io # pentru afisarea imaginii
import numpy as np
train_images = np.loadtxt('./data/train_images.txt') # incarcam imaginile
train_labels = np.asarray(np.loadtxt('./data/train_labels.txt'), dtype = np.int) # incarcam etichetele avand
 # tipul de date int
image = train_images[0, :] # prima imagine
image = np.reshape(image, (28, 28))
io.imshow(image.astype(np.uint8))
io.show()