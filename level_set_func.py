

from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
import drlse_algo as drlse
import numpy as np
from image_functions import *
from cv2 import Canny, bilateralFilter, medianBlur, resize, dilate, erode


### function which does the level set algorithm from a given seed point on an image ###
# --- INPUTS ---
# img : input image, uint8 numpy array
# seed_point_full : initial seed point for algorithm to start from
# res_filename : filename you want to store all the outputs as, typically as number ('5')
def level_set_func(img, seed_point_full):

	img = maxminscale(img) # need to scale just to be certain
	org_img = 1*img # need this later!

	# initial seed point -> to grow from
	#seed_point_full = (int(366.7933774834437), int(315.6754966887417))
	reg = 30 # window of interest -> tumor must be completely inside!!! (OBS!)
	img = img[seed_point_full[0]-reg:seed_point_full[0]+reg, seed_point_full[1]-reg:seed_point_full[1]+reg]

	seed_point = (reg, reg) # seed point for new smaller image 50x50 -> need to rename later.. np

	# parameters 
	timestep = 1        # time step (2)
	mu = 0.1/timestep   # coefficient of the distance regularization term R(phi)
	iter_inner = 1 # (4)
	iter_outer = 250 # (30) # use larger 250 for larger objects
	lmda = 2            # coefficient of the weighted length term L(phi).   (2)
	alfa = -9           # coefficient of the weighted area term A(phi)  (-4)
	epsilon = 2.0       # parameter that specifies the width of the DiracDelta function  (2.0)

	sigma = 0.8           # scale parameter in Gaussian kernel (0.8) <- have to change this to (2.0) if [-1024, 1024]
	img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
	[Iy, Ix] = np.gradient(img_smooth)
	f = np.square(Ix) + np.square(Iy)
	g = 1 / (1+f)    # edge indicator function.

	# initialize LSF as binary step function
	c0 = 2  # (2)
	initialLSF = c0 * np.ones(img.shape)
	# generate the initial region R0 as two rectangles
	initialLSF[seed_point] = -c0 # seed point -> to grow from
	phi = initialLSF.copy()

	#plt.ion()
	#fig1 = plt.figure(1)


	def show_fig1():
	    ax1 = fig1.add_subplot(111, projection='3d')
	    y, x = phi.shape
	    x = np.arange(0, x, 1)
	    y = np.arange(0, y, 1)
	    X, Y = np.meshgrid(x, y)
	    ax1.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
	    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)

	#show_fig1()
	#fig2 = plt.figure(2)


	def show_fig2():
	    contours = measure.find_contours(phi, 0)
	    ax2 = fig2.add_subplot(111)
	    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
	    for n, contour in enumerate(contours):
	        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color = "red")


	#show_fig2()
	#print('show fig 2 first time')

	potential = 2
	if potential == 1:
	    potentialFunction = 'single-well'  # use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
	elif potential == 2:
	    potentialFunction = 'double-well'  # use double-well potential in Eq. (16), which is good for both edge and region based models
	else:
	    potentialFunction = 'double-well'  # default choice of potential function

	# start level set evolution
	for n in range(iter_outer):
	    phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
	    if np.mod(n, 2) == 0:
	        print('show fig 2 for %i time' % n)
	        # fig2.clf()  # OBS! <- if want to show level set evolution -> but slows down iteration process
	        # show_fig2()
	        # fig1.clf()
	        # show_fig1()
	        # plt.pause(0.01) # (0.3)  OBS! <------- slows down computational time, but shows how the level set alg. grows from seed point

	# refine the zero level contour by further level set evolution with alfa=0
	alfa = 0 # (0)
	iter_refine = 10 # (10)
	phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

	finalLSF = phi.copy()
	# print('show final fig 2')
	# fig2.clf()
	# show_fig2()
	# fig1.clf()
	# show_fig1()

	# plt.show()


	#fig2.savefig('test_lvlset_' + res_filename + '.png')


	# get segmented tumor object (filled region)
	new_blob = 1*finalLSF
	new_blob = maxminscale(new_blob)
	new_blob = new_blob.astype(np.uint8)

	new_blob = bitwise_not(new_blob)
	new_blob[new_blob <= 30] = 0
	new_blob[new_blob > 30] = 255

	#fig_new = plt.figure(3)
	#plt.imshow(new_blob, cmap = "gray")
	#fig_new.savefig('Results/test_lvlset_' + res_filename + 'filled.png')


	# get ROI plotted on original image -> end result! -> Happy days!!!
	tempest = Canny(new_blob, 0, 255) # object -> region (maybe slow?)
	tempest_full = np.zeros(org_img.shape)

	tempest_full[seed_point_full[0]-reg:seed_point_full[0]+reg, seed_point_full[1]-reg:seed_point_full[1]+reg] = tempest
	vals = im2pixels(tempest_full)

	output = object_filler(maxminscale(tempest_full), (0,0))

	return output
	'''
	fig_res, ax = plt.subplots(num = 4, figsize = plt.figaspect(org_img))
	fig_res.subplots_adjust(0,0,1,1)
	ax.imshow(org_img, cmap="gray")
	ax.scatter(vals[:,0], vals[:,1], 0.1, color = 'firebrick')
	ax.axis('off')
	fig_res.savefig(res_filename)
	'''










