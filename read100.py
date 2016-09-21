import cPickle
import matplotlib.pyplot as plt
import numpy as np
import random

# Code taken from http://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

test_images = unpickle('./cifar-100-python/test')
training = unpickle('./cifar-100-python/train')
# The keys are: ['data', 'batch_label', 'fine_labels', 'coarse_labels', 'filenames']

def format_display(img):
    # Takes the image - and formats it such that we get an array of pixels, 
    # with each pixel having RGB values
    # Retuns the image as a single 1024 list of pixels
    return np.asarray([img[i::32*32] for i in range(len(img)/3)])


classes = ['apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 
			'bee', 'beetle', 'bicycle', 'bottles', 'bowls', 'boy', 'bridge', 
			'bus', 'butterfly', 'camel', 'cans', 'castle', 'caterpillar', 
			'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 
			'couch', 'crab', 'crocodile', 'cups', 
			'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 
			'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn-mower', 
			'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple', 
			'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges', 
			'orchids', 'otter', 'palm', 'pears', 'pickup truck', 'pine', 'plain', 
			'plates', 'poppies', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 
			'road', 'rocket', 'roses', 'sea', 'seal', 'shark', 'shrew', 
			'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 
			'streetcar', 'sunflowers', 'sweet peppers', 'table', 'tank', 
			'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulips', 
			'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman', 'worm']

superclasses = ['aquatic mammals', 'fish', 'flowers', 'food containers', 
				'fruit and vegetables', 'household electrical devices',
				'household furniture', 'insects', 'large carnivores',
				'large man-made outdoor things', 'large natural outdoor scenes',
				'large omnivores and herbivoes', 'medium-sized mammals', 'non-insect invertebrates',
				'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1',
				'vehicles 2']


fig = plt.figure()
for i in range(50):
	ax = fig.add_subplot(5,10,i+1)
	plt.imshow(format_display(training['data'][i]).reshape(32,32,3))
	index = training['coarse_labels'][i]
	plt.title(superclasses[index])

fig = plt.figure()
for i in range(50):
	ax = fig.add_subplot(5,10,i+1)
	plt.imshow(format_display(training['data'][51+i]).reshape(32,32,3))
	index = training['coarse_labels'][51+i]
	plt.title(superclasses[index])

plt.show()









