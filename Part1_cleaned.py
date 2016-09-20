import cPickle
import matplotlib.pyplot as plt
import numpy as np
import random

# Classes
# +------------+---+	 +------------+---+
# | Airplane   | 0 |	 | Dog        | 5 |
# +------------+---+	 +------------+---+
# | Automobile | 1 |	 | Frog       | 6 |
# +------------+---+	 +------------+---+
# | Bird	   | 2 |	 | Horse      | 7 |
# +------------+---+ 	 +------------+---+
# | Cat        | 3 |	 | Ship       | 8 |
# +------------+---+	 +------------+---+
# | Deer       | 4 |	 | Truck	  | 9 |
# +------------+---+	 +------------+---+


# Code taken from http://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def format_display(img):

	# Takes the image - and formats it such that we get an array of pixels, 
	# with each pixel having RGB values
	# Retuns the image as a single 1024 list of pixels
	px = [img[i::32*32] for i in range(len(img)/3)]
	px = np.asarray(px)

	return px

def RGB2GRY(img):
	# Converts a colour image to grayscale
	# Image needs to be in the format as returned by format_display
	# Input: a 1024 list of pixels where each pixel is [R G B]
	# Output: a 1024 list of grayscale values

	# Standard NTSC conversion
	img = [ 0.2989*i[0] + 0.5870*i[1] + 0.1140*i[2] for i in img]
	img = np.asarray(img)
	return img

def find_eigenimages(c, imgs, n_eig, plot):
	# c: class of images from which to extract the eigenimages
	# imgs: the entire data set of images
	# n_eig: number of eigenimages stored
	# plot (boolean): plots the first couple of eigenimages

	index = np.argwhere(np.asarray(images['labels']) == c)

	# list of selected images, where each image is formatted as an array of pixels
	image_px = [RGB2GRY(format_display(images['data'][i[0]])) for i in index]

	# extract eigen imae 
	_, s, w = np.linalg.svd(image_px)

	top_components = w[0:n_eig, :]

	if plot:
		fig = plt.figure()
		for i in range(25):
			ax = fig.add_subplot(5,5,i+1)
			plt.imshow(top_components[i].reshape(32,32), cmap='Greys_r')
			plt.title('Eigenimages of class' + str(c))

	return top_components

def eigen_reconstruct(img, n_rec, components):
	# img: the image to be reconstructed
	# n_rec: number of eigenimages used in the reconstruction
	# components: the eigenimages that can be used in the reconstruction
	# display (boolean): prints out the contribution of each eigenimage

	img_components = img.dot(components[:n_rec, :].T)
	rec_image = img_components.dot(components[:n_rec, :])
	return rec_image

def comparison(img1, img2):
	return np.sqrt(sum(np.square(img1-img2)))

    

def plot_reconstructed(pic, title):
	fig = plt.figure()
	plt.title(title)
	plt.imshow(pic.reshape(32,32), cmap='Greys_r')



images = unpickle('./cifar-10-batches-py/test_batch')
store_images = []

for i in range(10):
	# create eigenimages
	eigenimages = find_eigenimages(i, images, 100, False)

	# 1st picture in class c
	c = 9
	index = images['labels'].index(c)
	picture = RGB2GRY(format_display(images['data'][index]))
	store_images.append(eigen_reconstruct(picture, 25, eigenimages))


#############################
# Plot reconstructed images #
#############################

fig = plt.figure()
for i in range(10):
	ax = fig.add_subplot(2,5, i+1)
	plt.imshow(store_images[i].reshape(32,32), cmap='Greys_r')
	print(comparison(store_images[i], picture))

#############################
# Plot original image 	    #
#############################

fig = plt.figure()

plt.imshow(picture.reshape(32,32), cmap = 'Greys_r')


plt.show()













