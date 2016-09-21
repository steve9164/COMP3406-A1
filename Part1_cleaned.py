from __future__ import print_function

import cPickle
import matplotlib.pyplot as plt
import numpy as np
import random

import code

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
    with open(file, 'rb') as fo:
        return cPickle.load(fo)

def image_data(img):

    # Make an array of pixels as [r,g,b]
    #img = RGB2GRY(format_display(img))
    #px = 
    return (img - img.min())/float(img.max()-img.min())

def format_display(img):
    # Takes the image - and formats it such that we get an array of pixels, 
    # with each pixel having RGB values
    # Retuns the image as a single 1024 list of pixels
    return np.asarray([img[i::32*32] for i in range(len(img)/3)])

def RGB2GRY(img):
    # Converts a colour image to grayscale
    # Image needs to be in the format as returned by format_display
    # Input: a 1024 list of pixels where each pixel is [R G B]
    # Output: a 1024 list of grayscale values

    # Standard NTSC conversion
    img = [ 0.2989*i[0] + 0.5870*i[1] + 0.1140*i[2] for i in img]
    img = np.asarray(img)
    return img

def generate_eigenimages(imgs, n_eig, plot):
    # c: class of images from which to extract the eigenimages
    # imgs: the entire data set of images
    # n_eig: number of eigenimages stored
    # plot (boolean): plots the first couple of eigenimages

    # Generate a list of lists of images (by category and index within category)
    images = [[] for x in range(10)]

    train = len(imgs['data'])
    for i in range(train):
        label = imgs['labels'][i]
        picture = image_data(imgs['data'][i])
        images[label].append(picture)
    
    # Validation matrix
    validation = []
    eigenimages = []
    # extract eigen images for each category
    print('Starting svd')
    for image in images:
        U, s, V  = np.linalg.svd(image, full_matrices=False)
        b = U[:n_eig,:].dot(s)
        validation.append(b)
        eigenimages.append(V[:n_eig,:])
    print('Finished svd')


    if plot:
        for c in range(10):
            fig = plt.figure()
            for i in range(25):
                ax = fig.add_subplot(5,5,i+1)
                plt.imshow(eigenimages[c][i].reshape(32,32), cmap='Greys_r')
                plt.title('Eigenimages of class' + str(c))

    return eigenimages, validation

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


def translate_x(img, dist):
    img = img.reshape((3,32,32))


#def transform_image(image):


def plot_reconstructed(pic, title):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(pic.reshape(32,32), cmap='Greys_r')

#################
# 	Script 		#
#################

classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

transforms = [[
    ('Identity', lambda img: img), 
    ('Reflect horizontally', lambda img: img.copy().reshape((3,32,32))[:,:,::-1].reshape((3*32*32))), 
    ('Reflect vertically', lambda img: img.copy().reshape((3,32,32))[:,::-1,:].reshape((3*32*32)))
    ], [
    ('Identity', lambda img: img), 
    ('Translate 2 pixels +x'), 
    ('Translate 4 pixels +x'), 
    ('Translate 2 pixels -x'), 
    ('Translate 4 pixels -x'), 
    ], [
    ('Identity', lambda img: img), 
    ('Translate 2 pixels +y'), 
    ('Translate 4 pixels +y'), 
    ('Translate 2 pixels -y'), 
    ('Translate 4 pixels -y')
]]

n_eig = 100

test_images = unpickle('./cifar-10-batches-py/test_batch')
training = [unpickle('./cifar-10-batches-py/data_batch_{}'.format(i+1)) for i in range(5)]
print('Loading training data')
training_images = {'data': [img for train in training for img in train['data']], 
                   'labels': [label for train in training for label in train['labels']]}
print('Finished loading training data')
eigenimages, validation = generate_eigenimages(training_images, n_eig, False)
'''
fig = plt.figure()
for i in range(25):
    ax = fig.add_subplot(5,5, i+1)
    plt.imshow(eigenimages[i][:25].reshape(32,32), cmap='Greys_r')
'''
correct = 0

tests = len(test_images['data'])

for i in range(tests):
    label = test_images['labels'][i]
    picture = image_data(test_images['data'][i])

    #transformed_pictures = transform_image(picture)

    
    reconstructed = [eigen_reconstruct(pic, n_eig, eigs) for pic in (t[1](picture) for t in transforms[0]) for eigs in eigenimages]

    # # Plot reconstructed
    fig = plt.figure()
    for i,p in enumerate(reconstructed + [picture]):
        fig.add_subplot(7,5, i+1)
        plt.imshow(format_display(p.copy()).reshape((32,32,3)))
    plt.show()

    # print(len(reconstructed))

    # ---- Plot the images -----
    # fig = plt.figure()
    # for i in range(len(store_images)):
    # 	ax = fig.add_subplot(2,5, i+1)
    # 	plt.imshow(store_images[i].reshape(32,32), cmap='Greys_r')

    err = [comparison(picture, img) for img in reconstructed]
    print(classes[label], ', '.join('{}: {:.2f} ({})'.format(classes[t[0]%10], t[1], transforms[0][t[0]/10][0]) for t in sorted(enumerate(err), key=lambda t: t[1])))
    if np.argmin(err) % 10 == label:
        correct += 1

print(correct, tests)
print('Accuracy: {:.2%}'.format(correct/float(tests)))

# plt.figure()
# for i in range(10):
#     ax = fig.add_subplot(2,5,i+1)
#     plt.imshow(format_display(store_images[i]))

# # ----- plot all the images ----
# # fig = plt.figure()
# # for i in range(tests):
# # 	ax = fig.add_subplot(4,5, i+1)
# # 	plt.imshow(image_data(test_images['data'][i]).reshape(32,32), cmap='Greys_r')

# # plt.show()



# #############################
# # Plot reconstructed images #
# #############################

# # fig = plt.figure()
# # for i in range(10):
# # 	ax = fig.add_subplot(2,5, i+1)
# # 	plt.imshow(store_images[i].reshape(32,32), cmap='Greys_r')
# # 	print(comparison(store_images[i], picture))

# # #############################
# # # Plot original image 	    #
# # #############################

# # fig = plt.figure()

# # plt.imshow(picture.reshape(32,32), cmap = 'Greys_r')


# plt.show()











