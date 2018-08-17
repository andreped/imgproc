

from image_functions import imgclean, maxminscale, load_itk, object_filler, cmass, box_minimizer, find_nearest, im2pixels
from level_set_func import level_set_func
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pytictoc import TicToc
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.cluster import KMeans
from skimage import morphology, measure
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, binary_erosion, binary_closing, remove_small_holes, remove_small_objects
from skimage.segmentation import inverse_gaussian_gradient
from skimage.feature import blob_dog, blob_log, blob_doh
import scipy
from skimage.color import rgb2gray
import imageio
from SRM import SRM # (!) <- can't install using pip (!)
from skimage.filters import threshold_otsu, threshold_adaptive
import skfuzzy as fuzz
import operator
#import pymrt.geometry # <- reason for annoying display of module name...
import skimage
import h5py
from scipy.ndimage import gaussian_gradient_magnitude
from tqdm import tqdm
import time
from sklearn.cluster import DBSCAN
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from chanvese3d import chanvese3d
from scipy import ndimage
import sys
from scipy import misc


###################################################################################################################
# Try to segment out lung to use it as mask to filter out everything not of interest
###################################################################################################################

# measure time
t = TicToc()
t.tic()

# choose patient
pat = input("choose patient: ")

## get data
print('reading patient data...')
t.tic()
patient = '/Users/medtek/Documents/Project/Images/patient_data/' + str(pat) + '.hd5'
f = h5py.File(patient, 'r')
Input = f['data']
output = f['label']
data = np.squeeze(Input, axis=3)
t.toc()

print(data.shape)

# display slices with tumor
print('finding slice numbers containing tumors...')
t.tic()
gt_data = output.value[:,:,:,0]
gt_vals = []
for i in range(gt_data.shape[0]):
	if len(np.unique(gt_data[i,:,:])) > 1:
		gt_vals.append(i)
t.toc()

print(gt_vals)

# get resolution data about current patient
print('get resolution info about patient data')
res_path = '/Users/medtek/Documents/Project/Images/res_data/res_pat.hd5'
f = h5py.File(res_path, 'r')
res = np.squeeze(f['res'], axis = 2)
res = res[eval(pat)-1,:]





print('--')
print(data.shape)

# get some image
num = int(input("select slice containing tumor: "))
#num = 89
img = data[num,:,:]

#imgclean(img, Figure=True)

#num = 224 # slice number
#img = data[num,:,:] # 84 is a problem! lung segmentation not robust enough

imgclean(img, Figure=True)

img[img <= -1024] = -1024
img[img > 1024] = 1024


#imgclean(img, Figure=True)

# keep original maxminscaled image -> becuase img is going to be altered
img_orig = img

# blur img
#img = maxminscale(img)
img = cv2.medianBlur(img, 1)

# get dimensions of image
row_size, col_size = img.shape

# specify window for k-means to work on, such that you get the lung, and not the rest
middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] # lung area of interest

kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img<threshold,1.0,0.0)

#imgclean(thresh_img, Figure=True)

labels = measure.label(thresh_img) # Different labels are displayed in different colors
regions = measure.regionprops(labels)
good_labels = []
for prop in regions:
    B = prop.bbox
    if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/10 and B[2]<col_size/10*9: # better window!
        good_labels.append(prop.label)
mask = np.ndarray([row_size,col_size],dtype=np.int8)
mask[:] = 0

for N in good_labels:
    mask = mask + np.where(labels==N,1,0)

## fill regions surrounded by lung regions -> final ROI
# to fill holes -> without filling unwanted larger region in the middle
res = binary_fill_holes(mask).astype(int)

# need to filter out the main airways, because elsewise they will affect the resulting algorithm
res2 = remove_small_objects(label(res), min_size=1100)
res2[res2 > 0] = 1
#res2 = res.copy()

# first close boundaries to include juxta-vascular nodule candidates
mask = morphology.dilation(res2, disk(7)) # 7 (17 for more difficult ones)

# then fill inn nodules to include them, but not the much larger objects, ex. heart etc
mask = remove_small_objects(label(cv2.bitwise_not(maxminscale(mask))), min_size = 300) # can change min_size larger if want to include larger nodule candidates
mask[mask != 0] = -1
mask = np.add(mask, np.ones(mask.shape))

# last erosion to complete the closing+, larger disk size because need to remove some of the lung boundaries
filled_tmp = morphology.erosion(mask, disk(9)) # 9 (19 for more difficult ones)

imgclean(filled_tmp, Figure=True)

#image = filled_tmp*img_orig
image = img_orig.copy()
image[filled_tmp == 0] = np.amin(img_orig)

#image = filled_tmp*img_orig

if (len(np.unique(image)) == 1):
	sys.exit('\n (: no tumor candidates of interest :) \n')

image = np.uint8(maxminscale(image))
image = cv2.medianBlur(image, 5)


## contrast enhancement?
# create a CLAHE object (Arguments are optional). -> more adaptive?
# image = np.uint8(maxminscale(image))
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# image = clahe.apply(image)

# tmp_image = image.copy()

# # Define the window size
# windowsize_r = 64
# windowsize_c = 64

# # Crop out the window and calculate the histogram
# for r in range(0,tmp_image.shape[0] - windowsize_r, windowsize_r):
#     for c in range(0,tmp_image.shape[1] - windowsize_c, windowsize_c):
#         window = tmp_image[r:r+windowsize_r,c:c+windowsize_c]
#         image[r:r+windowsize_r,c:c+windowsize_c] = clahe.apply(window)



imgclean(image, Figure=True)


# ### First: enhance blob-like objects in 2D-slice
# # multi-scale Gaussian gradient applied to image
# blobs_log = blob_log(image, max_sigma=15, num_sigma=15, threshold=40)

# blob_enhancer = np.zeros(image.shape)
# for blob in blobs_log:
# 	y, x, r = blob
# 	r = r-1
# 	blob_enhancer[int(y-r):int(y+r+1), int(x-r):int(x+r+1)] = disk(r)

# imgclean(blob_enhancer, Figure=True)

# image = image# + 50*blob_enhancer



##### NOW: Use 3-class FCM to generate candidates, but also to hopefully seperate between three types of structures in lung ######
# -> assuming data has three clusters from apriori information about the lung. Hopefully this results in good enough seperation

# pixel intensities
I = image.reshape((1, -1))

# init
fpcs = []

# apply FCM
ncenters = 3
m = 2 # fuzziness parameter
error = 0.1
max_iter = 100
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(I, ncenters, m, error=error, maxiter=max_iter, init=None) # choose initialization, to reduce randomness in output
print(fpc)

image_clustered = np.argmax(u, axis = 0).astype(float)

# change shape of array
image_clustered.shape = image.shape


imgclean(image_clustered, Figure=True)




# get segmentation mask for nodule candidates
image_mask = image.copy()
image_mask[image_clustered == np.argmax(cntr)] = 1
image_mask[image_clustered != np.argmax(cntr)] = 0

# segment out nodule candidates from lung-segmented-image
image_cands = np.multiply(image_mask, image)

# use morphology -> eroision to filter out the smallest objects
# closing
image_mask_filtered = image_mask.copy()
#image_mask_filtered = morphology.dilation(image_mask, disk(1)) #(2) <- 5 was a test, 3 is best(?)
#image_mask_filtered = morphology.erosion(image_mask, disk(2))  # (2)
#image_mask_filtered = morphology.dilation(image_mask, disk(2))

## opening
#image_mask_filtered = morphology.erosion(image_mask, disk(1))  # (2)

# label and remove obvious vessels
labels, nlabels = label(image_mask_filtered, return_num = True)

#area = np.asarray([np.sum(s) for s in slices])



print(nlabels)
print('--')


imgclean(labels, Figure=True)

# remove vessel-like objects
#val = []
reg_tmp = regionprops(labels)
ii = 0
val = []
for reg in reg_tmp:
	if ((reg.area > 100) and (reg.area/reg.major_axis_length < 4)):# and (reg.euler_number)): #(reg.area/radii[ii] > 20)):
		labels[labels == reg.label] = 0
	ii += 1

labels[labels > 0] = 1
imgclean(labels, Figure=True)

labeled_new = labels.copy()
labeled_new = morphology.dilation(labeled_new, disk(3))
labeled_new = morphology.erosion(labeled_new, disk(3))
labeled_new = morphology.erosion(labeled_new, disk(1))

imgclean(labeled_new, Figure=True)

# do one last filtering, to remove the larger obviously not tumors after eroision
labeled2 = label(labeled_new)
reg_tmp = regionprops(labeled2)
ii = 0
val = []
for reg in reg_tmp:
	if ((reg.area > 40) and ((reg.perimeter**2/reg.area > 35))): # or ((reg.area/reg.major_axis_length < 5))): # and (reg.euler_number)): #(reg.area/radii[ii] > 20)):
		labeled2[labeled2 == reg.label] = 0
	ii += 1

labeled2[labeled2 > 0] = 1

labeled2 = morphology.erosion(labeled2, disk(2)) # 1.8
labeled2 = morphology.dilation(labeled2, disk(3)) #(2) <- 5 was a test, 3 is best(?)


if (len(np.unique(labeled2)) == 1):
	sys.exit('\n (: no tumor candidates of interest :) \n')


fig, ax = plt.subplots(1,5)
ax[0].imshow(image, cmap = "gray")
ax[1].imshow(image_mask_filtered, cmap = "gray")
ax[2].imshow(labels, cmap = "gray")
ax[3].imshow(labeled_new, cmap = "gray")
ax[4].imshow(labeled2, cmap = "gray")
plt.show()

image_mask_filtered = labeled_new.copy()


#image_mask_filtered = morphology.dilation(image_mask_filtered, disk(2)) #(2) <- 5 was a test, 3 is best(?)

#image_mask_filtered = image_mask

#output_image = image*image_mask_filtered


#exit()

# need to label all 2D-candidates
#label_image = label(image_mask_filtered)
#regions = regionprops(label_image)

#imgclean(label_image, Figure=True, col="inferno")


#label_binary = 1*label_image
#label_binary[label_binary > 0] = 1

# get ground truth for current slice
gt = im2pixels(cv2.Canny(np.uint8(maxminscale(gt_data[num,:,:])), 0, 255))

fig, ax = plt.subplots(3,3)
ax[0,0].imshow(img_orig, cmap = "gray")
ax[0,1].imshow(thresh_img, cmap = "gray")
ax[0,2].imshow(res, cmap = "gray")
ax[1,0].imshow(filled_tmp, cmap = "gray")
ax[1,1].imshow(image, cmap = "gray")
ax[1,2].imshow(image_mask, cmap = "gray")
ax[2,0].imshow(labeled_new, cmap = "gray")
ax[2,1].imshow(labeled2, cmap = "gray")
ax[2,1].scatter(gt[:,0],gt[:,1], color = 'r', marker = '.', s = 0.2)
ax[2,2].imshow(img_orig,extent=[0, 512, 512, 0], cmap = "gray")
ax[2,2].scatter(gt[:,0],gt[:,1], color = 'r', marker = '.', s = 0.2)
plt.show()


exit()

# get area of each object
areas = [r.area for r in regionprops(label_image)]
print(areas)

# get labels
labels2D = [r.label for r in regionprops(label_image)]
print(labels2D)

# get segmented image as coordinates for each 2D object
object2D = [r.coords for r in regionprops(label_image)]


exit()

tmp = np.zeros((512,512))
for i in range(len(object2D[-1][:][:])):
	pos = object2D[-1][:][:][i,:]
	tmp[pos[0], pos[1]] = 1

plt.imshow(tmp, cmap = "gray")
plt.show()















