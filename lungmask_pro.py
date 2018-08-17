

import numpy as np
from skimage.morphology import disk, binary_erosion, binary_closing, remove_small_holes, remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label
from image_functions import *
from sklearn.cluster import KMeans
from skimage import morphology, measure
from scipy.ndimage.filters import median_filter
from skimage.filters import rank, threshold_otsu
from pytictoc import TicToc
from scipy.ndimage.interpolation import zoom
import os, sys
from mayavi import mlab
from cv2 import Canny, bilateralFilter, medianBlur, resize, dilate, erode, bitwise_not


# Function that segments the lung in a 2D-image
# --- INPUT ---
# img : 2D numpy array
# --- OUTPUT ---
# lungmask : binary numpy array where 1 correspond to lung, 0 else
def lungmask_pro(img, morph = True):

	# to avoid many tears
	img = img.copy()

	# set intensity range to standard (corners have problems) -> doesn't really matter, robust method, but looks better
	img[img <= -1024] = -1024
	img[img >= 1024] = 1024 # also this, because of circular region artifacts (actually also for the one above)

	# scale image to be in the range [0,255] -> because method robust enough to handle it -> smarter thresholds
	img = np.uint8(maxminscale(img))

	# keep scaled original image for masking later
	img_orig = img.copy()

	# blur img to easier being able to segment the lung using kmeans
	img = medianBlur(img, 5)

	# get dimensions of image
	row_size, col_size = img.shape

	# specify window for k-means to work on, such that you get the lung, and not the rest
	middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

	# apply otsu's method to threshold image
	th = threshold_otsu(middle)
	thresh_img = np.where(img<th,1.0,0.0)

	# label each object from the filtering above and only get the lung including juxta-vascular nodules
	labels = label(thresh_img) # Different labels are displayed in different colors
	regions = regionprops(labels)
	good_labels = []
	for prop in regions:
	    B = prop.bbox
	    if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/10 and B[2]<col_size/10*9: # better window!
	        good_labels.append(prop.label)
	mask = np.ndarray([row_size,col_size],dtype=np.int8)
	mask[:] = 0

	# gets objects of interest from criteria defined and used above
	for N in good_labels:
	    mask = mask + np.where(labels==N,1,0)

	## fill regions surrounded by lung regions -> final ROI
	# to fill holes -> without filling unwanted larger region in the middle
	res = binary_fill_holes(mask)

	# need to filter out the main airways easily seperated from lung, because else they will affect the resulting algorithm
	res2 = remove_small_objects(label(res).astype(bool), min_size=800)
	res2[res2 > 0] = 1

	# separate each object in image (i.e. each lung part), and do morphology to include juxta-vascular nodules
	lungmask = np.zeros(res2.shape)
	labels = label(res2)
	for i in range(1,len(np.unique(labels))):
		tmp = np.zeros(labels.shape)
		tmp[labels == i] = 1

		if (morph == True):
			mask = dilate(np.uint8(tmp), disk(17))
			mask = remove_small_objects(label(bitwise_not(maxminscale(mask))).astype(bool), min_size = 500).astype(int)
			mask[mask != 0] = -1
			mask = np.add(mask, np.ones(mask.shape))
			filled_tmp = erode(np.uint8(mask), disk(19))
		else:
			mask = remove_small_objects(label(bitwise_not(maxminscale(tmp))).astype(bool), min_size = 500).astype(int)
			mask[mask != 0] = -1
			filled_tmp = np.add(mask, np.ones(mask.shape))

		lungmask += filled_tmp

	return lungmask



# Function that masks out the lung on a 3D-image stack.
# --- INPUT ---
# data : 3D-numpy array with dimensions (slice, (img x,y))
# --- OUTPUT ---
# res  : binary 3D-numpy array where 1 correspond to lung, 0 else
def lungmask3D(data, morph = True):

	# to avoid many tears
	tmp = data.copy()

	# apply lungmask on each slice
	for i in range(data.shape[0]):
		tmp[i] = lungmask_pro(data[i], morph)

	# remove smaller objects like bladder etc (if 3D object is smaller than 0.5 % of total volume)
	res = remove_small_objects(label(tmp).astype(bool), min_size = int(round(0.005*(data.shape[0]*data.shape[1]*data.shape[2])))).astype(int)
	res[res > 0] = 1

	return res



# --- something something text text ---
def plot_lung_and_tumor(im_st1, pred_st1, gt_st1, thr = 0.5):

	# measure time
	t = TicToc()

	# segment the lung from the 3D-stack
	print('lungmask...', end="\r")
	mask1 = lungmask3D(im_st1)

	# need to resize arrays to make plotting faster
	print('resizing...', end="\r")
	mask = np.zeros((im_st1.shape[0], 256, 256))
	pred_st = mask.copy()
	gt_st = mask.copy()

	for i in range(im_st1.shape[0]):
		mask[i,:,:] = resize(mask1[i,:,:].astype(float), (256,256))
		pred_st[i,:,:] = resize(pred_st1[i,:,:].astype(float), (256,256))
		gt_st[i,:,:] = resize(gt_st1[i,:,:].astype(float), (256,256))

	# get appropriate edge structure around object (surface), for easier & faster plotting
	for i in range(mask.shape[1]):
		mask[:, i, :] = Canny(np.uint8(maxminscale(mask[:, i, :])),50,100)
		#pred_st[:, i, :] = Canny(np.uint8(maxminscale(pred_st[:, i, :])),50,100)

	# apply thresholding to get appropriate plots
	pred_st[pred_st < thr] = 0
	pred_st[pred_st >= thr] = 180
	gt_st[gt_st >= thr] = 180    

	# TODO - make zoom_fac dependent on resolution in z dir (true values)
	zoom_fac = mask.shape[1]/mask.shape[0]

	print('zoom...', end="\r")
	mask = zoom(mask, zoom = [zoom_fac,1,1])
	pred_st = zoom(pred_st, zoom = [zoom_fac,1,1])
	gt_st = zoom(gt_st, zoom = [zoom_fac,1,1])

	t.tic()
	print('drawing...', end="\r")
	mlab.figure(1, size = (1000,800))
	mlab.contour3d(pred_st, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 255)
	mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)
	mlab.title('Prediction')

	mlab.figure(2, size = (1000,800))
	mlab.contour3d(gt_st, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 255)
	mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)
	mlab.title('Ground truth')
	t.toc()

	mlab.show()

















