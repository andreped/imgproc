

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


# get some image
num = int(input("select slice containing tumor: "))
img = data[num,:,:]

#num = 224 # slice number
#img = data[num,:,:] # 84 is a problem! lung segmentation not robust enough

img[img <= -1024] = -1024 # better visual contrast, BUT AFFECTS PERFORMANCE OF FCM!!!
img[img >= 1024] = 1024
img = maxminscale(img) # OBS

#imgclean(img, Figure=True)


# keep original maxminscaled image -> becuase img is going to be altered
img_orig = img.copy()

# blur img
img = np.uint8(img)
img = cv2.medianBlur(img, 5)

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
label_vals = np.unique(labels)
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
print('--')

# need to filter out the main airways, because elsewise they will affect the resulting algorithm
res2 = label(res)
#imgclean(res2,Figure=True)
res2 = remove_small_objects(res2, min_size=800)
#imgclean(res2,Figure=True)
res2[res2 > 0] = 1


# first close boundaries to include juxta-vascular nodule nodules
mask = morphology.dilation(res2, disk(7)) # 7 (17 for more difficult ones)

# then fill inn nodules to include them, but not the much larger objects, ex. heart etc
mask = cv2.bitwise_not(np.uint8(maxminscale(mask)))
mask = label(mask)
mask = remove_small_objects(mask, min_size = 300) # can change min_size larger if want to include larger nodule candidates
tmp_mask = 1*mask
tmp_mask[tmp_mask != 0] = -1
mask = np.add(tmp_mask, np.ones(tmp_mask.shape))

# last erosion to complete the closing+, larger disk size because need to remove some of the lung boundaries
filled_tmp = morphology.erosion(mask, disk(9)) # 9 (19 for more difficult ones)

image = filled_tmp*img_orig

## use the mask -> get a segmented image only containing the lung AND tumor inside/border
# tmp = np.zeros(img_orig.shape)
# image = 1*img_orig
# image[np.where(filled_tmp == 0)] = np.amin(img_orig)
# t.toc()




# multi-scale Gaussian gradient applied to image
#Filter = gaussian_gradient_magnitude(image, sigma = 5) # (4)
#Filter = inverse_gaussian_gradient(image, alpha=100.0, sigma=4.0)
#imgclean(Filter, Figure=True)


# blobs_log = blob_log(image, max_sigma=15, num_sigma=15, threshold=40)

# # Compute radii in the 3rd column.
# blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# blobs_dog = blob_dog(image, min_sigma = 5, max_sigma=15, threshold=50)
# blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

# blobs_doh = blob_doh(image, max_sigma=15, threshold=60)

# blobs_list = [blobs_log, blobs_dog, blobs_doh]
# colors = ['yellow', 'lime', 'red']
# titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
#           'Determinant of Hessian']
# sequence = zip(blobs_list, colors, titles)

# fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
# ax = axes.ravel()

# for idx, (blobs, color, title) in enumerate(sequence):
#     ax[idx].set_title(title)
#     ax[idx].imshow(image, interpolation='nearest')
#     for blob in blobs:
#         y, x, r = blob
#         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#         ax[idx].add_patch(c)
#     ax[idx].set_axis_off()

# plt.tight_layout()
# plt.show()

#exit()
#print(np.unique(Filter))

#imgclean(Filter, Figure=True)

# # Use Laplacian of Gaussian (LoG-filter) to enhance smaller nodule candidates
# # cannot use gaussian gradient filter, because enhances all types of edges too much...
# blobs_log = blob_log(image, max_sigma=25, num_sigma=25, threshold=40)
# #blobs_log = blob_doh(image, max_sigma=15, threshold=60)
# print(blobs_log)
# blob_enhancer = np.zeros(image.shape)
# for blob in blobs_log:
# 	y, x, r = blob
# 	r = r-1
# 	blob_enhancer[int(y-r):int(y+r+1), int(x-r):int(x+r+1)] = disk(r)

# imgclean(image, Figure=True)
# imgclean(blob_enhancer, Figure=True)


#exit()

tmp = image.copy()# + 100*blob_enhancer

# multi-scale Gaussian gradient applied to image
Filter = gaussian_gradient_magnitude(tmp, sigma = 5) # (4)
imgclean(Filter, Figure=True)

GI = image# + Filter  # image sharpening -> highboost filtering idea
GI = maxminscale(GI)
#GI = np.uint8(GI)
#imgclean(GI, Figure=True)


### APPLY TOBOGGAN ALGORITHM ###
mat = [[0,1,2],[3,4,5],[6,7,8]]
footprint = [[0,1,0],[1,0,1],[0,1,0]]


# init
# GI : gradient image <- input
isMin = True
VG = np.ones(image.shape) # label gradient
label_img = -1*np.ones(image.shape) #np.zeros(image.shape)
p = 0.15 # optimal value (0.0, 0.2] range
q = 1-p
tmp_path = []

# for each pixel, skip boundary pixels -> not of interest anyways
# -> always scan for new pixels, but skip already labeled pixels
for i in tqdm(range(1,GI.shape[0]-1), ncols=100):
	for j in range(1,GI.shape[1]-1):
		# skip already labeled pixels or if it is already global minimum (0)
		if (label_img[i,j] != -1 or GI[i,j] == 0):
			continue
		else:
			isMin = False # init toboggan process
			source = GI[i,j] # get current pixel
			Min = source.copy() # current minimum

			# to use for neighbourhood search. They change as we slide towards a local minima, while (i,j) stays the same for original center pixel
			ii = i; jj = j

			# reset path
			del tmp_path[:]  # store locations it has already been, to skip unnecessairy iterations

			# 
			while (isMin == False): # True or False - ?
				# store path in list
				tmp_path.append((ii,jj))
				isMin = True # if minima found <- should slide towards local minima from source pixel
				# for all adjecent pixels with 4-connectivity (neighbours), check if smaller than center
				# should continue to slide towards local minima!
				#t.tic()
				if (GI[ii,jj+1] < Min):
					Min = GI[ii,jj+1]
					isMin = False
					jj = jj+1 # change centre dep on minimum neighbour
				if (GI[ii,jj-1] < Min):
					Min = GI[ii,jj-1]
					isMin = False
					jj = jj-1
				if (GI[ii+1,jj] < Min):
					Min = GI[ii+1,jj]
					isMin = False
					ii = ii+1
				if (GI[ii-1,jj] < Min): # Min or GI(P) - ?
					Min = GI[ii-1,jj] # leftmost pixel chosen if several candidates
					isMin = False
					ii = ii-1

				# if path already been labeled, we know were we should end when we slide down the hill -> don't need to continue
				#if (label_img[ii,jj] != -1):
				#	isMin = True # stop, we know were we should end # OBS <- RESULTS IN NOISY OUTPUT!

				if (isMin == True):
					#t.tic()
					# try to match the min
					isExist = Min in VG #sum(sum(VG == Min)) # OBS: 0 if False, else True
					if (isExist == True): # if True
						for k in range(len(tmp_path)):
							label_img[tmp_path[k]] = Min #p*source + q*Min**2 #Min
						#label_img[i,j] = Min #p*source + q*(Min)**2 #Min <- use Min to get patched output
					else:
						for k in range(len(tmp_path)):
							VG[i,j] = source
							label_img[tmp_path[k]] = source
					#t.toc()
			#time.sleep(0.1)
label_img[label_img < 0] = 0

GI = np.float32(GI)
print(np.amin(GI), np.amax(GI))
print(np.amin(label_img), np.amax(label_img))
label_tmp = label_img.copy()
label_tmp = p*GI + q*label_tmp*2
label_tmp_th = label_tmp
#label_img_th = np.uint8(maxminscale(label_img))

# Otsu's thresholding
th_img = label_tmp_th.copy()
th_img[th_img <= 80] = 0 # 70
th_img[th_img > 80] = 1
#ret2,th_img = cv2.threshold(label_tmp_th,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#imgclean(label_img, Figure=True)
#imgclean(th_img, Figure=True)

# now filter out some nodule candidates, to get hopefully single object -> nodule -> seed point
#th = 80
#label_new = label_img
#label_new[label_new <= th] = 0
#label_new[label_new > th] = 1
label_new = label(th_img)
label_new = remove_small_objects(label_new, min_size=30)
print(len(np.unique(label(label_new)))-1)
label_new[label_new > 0] = 1

gt = im2pixels(cv2.Canny(np.uint8(maxminscale(gt_data[num,:,:])), 0, 255))




fig, ax = plt.subplots(3,3)
ax[0,0].imshow(img_orig, cmap = "gray")
ax[0,1].imshow(thresh_img, cmap = "gray")
ax[0,2].imshow(res, cmap = "gray")
ax[1,0].imshow(filled_tmp, cmap = "gray")
ax[1,1].imshow(GI, cmap = "gray")
ax[1,2].imshow(label_img, cmap = "gray")
ax[2,0].imshow(th_img, cmap = "gray")
ax[2,1].imshow(label_new, cmap = "gray")
ax[2,2].imshow(GI,extent=[0, 512, 512, 0], cmap = "gray")
ax[2,2].scatter(gt[:,0],gt[:,1], color = 'r', marker = '.', s = 0.2)
plt.show()


blobs_log = blob_log(label_img, max_sigma=15, num_sigma=10, threshold=60)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

blobs_dog = blob_dog(label_img, max_sigma=15, threshold=60)
blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

blobs_doh = blob_doh(label_img, max_sigma=15, threshold=60)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(label_img, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()



# label_img = np.uint8(maxminscale(label_img))
# label_img = cv2.medianBlur(label_img, 5)
# ### Test using circular Hough transform to detect nodules
# circles = cv2.HoughCircles(label_img,cv2.HOUGH_GRADIENT,5,250,
#                             param1=3,param2=3,minRadius=0,maxRadius=25)

# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(label_img,(i[0],i[1]),i[2],(127,50,0),1)
#     # draw the center of the circle
#     cv2.circle(label_img,(i[0],i[1]),2,(127,0,50),2)

# cv2.imshow('detected circles',label_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# exit()

### Now, do region growing with multi-constraints to segment 3D-object :) ###

# plot center of mass of all nodule candidates
fig1, ax1 = plt.subplots()
ax1.imshow(label_new, extent=[0, 512, 512, 0], cmap = "gray")
image = label(label_new)
for i in range(len(np.unique(image))-1):
	CM = regionprops(image)[i].centroid
	ax1.plot(CM[1], CM[0], 'ro')
plt.show()


# get initial seed point
seedpoint = CM


exit()



### REGION GROWING TEST ###
# def get8n(x, y, shape):
#     out = []
#     maxx = shape[1]-1
#     maxy = shape[0]-1

#     #top left
#     outx = min(max(x-1,0),maxx)
#     outy = min(max(y-1,0),maxy)
#     out.append((outx,outy))

#     #top center
#     outx = x
#     outy = min(max(y-1,0),maxy)
#     out.append((outx,outy))

#     #top right
#     outx = min(max(x+1,0),maxx)
#     outy = min(max(y-1,0),maxy)
#     out.append((outx,outy))

#     #left
#     outx = min(max(x-1,0),maxx)
#     outy = y
#     out.append((outx,outy))

#     #right
#     outx = min(max(x+1,0),maxx)
#     outy = y
#     out.append((outx,outy))

#     #bottom left
#     outx = min(max(x-1,0),maxx)
#     outy = min(max(y+1,0),maxy)
#     out.append((outx,outy))

#     #bottom center
#     outx = x
#     outy = min(max(y+1,0),maxy)
#     out.append((outx,outy))

#     #bottom right
#     outx = min(max(x+1,0),maxx)
#     outy = min(max(y+1,0),maxy)
#     out.append((outx,outy))

#     return out

# def region_growing(img, seed):
#     list = []
#     outimg = np.zeros_like(img)
#     list.append((seed[0], seed[1]))
#     processed = []
#     while(len(list) > 0):
#         pix = list[0]
#         outimg[pix[0], pix[1]] = 255
#         for coord in get8n(pix[0], pix[1], img.shape):
#             if img[coord[0], coord[1]] != 0:
#                 outimg[coord[0], coord[1]] = 255
#                 if not coord in processed:
#                     list.append(coord)
#                 processed.append(coord)
#         list.pop(0)
#         #cv2.imshow("progress",outimg)
#         #cv2.waitKey(1)
#     return outimg

# def on_mouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print ('Seed: ' + str(x) + ', ' + str(y), img[y,x])
#         clicks.append((y,x))

# GI = np.uint8(maxminscale(GI))

# clicks = []
# #ret, img = cv2.threshold(GI, 20, 255, cv2.THRESH_BINARY)
# cv2.namedWindow('Input')
# cv2.setMouseCallback('Input', on_mouse, 0, )
# cv2.imshow('Input', GI)
# cv2.waitKey()
# seed = clicks[-1]
# out = region_growing(GI, seed)
# cv2.imshow('Region Growing', out)
# cv2.waitKey()
# cv2.destroyAllWindows()


### ACTIVE CONTOUR TEST ###
# img = np.uint8(maxminscale(GI))

# s = np.linspace(0, 2*np.pi, 400)
# x = seedpoint[1] + 40*np.cos(s)
# y = seedpoint[0] + 40*np.sin(s)
# init = np.array([x, y]).T

# imgclean(img, Figure=True)
# imgclean(gaussian(img,2), Figure=True)

# snake = active_contour(gaussian(img,3),
#                        init, alpha=0.015, beta=10, gamma=0.0001)

# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
# ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.show()



# ### 3D-region growing from single seed point!!!
# def grow(img, seed, t):
#     """
#     img: ndarray, ndim=3
#         An image volume.
    
#     seed: tuple, len=3
#         Region growing starts from this point.

#     t: int
#         The image neighborhood radius for the inclusion criteria.
#     """
#     seg = np.zeros(img.shape, dtype=np.bool)
#     checked = np.zeros_like(seg)

#     seg[seed] = True
#     checked[seed] = True
#     needs_check = get_nbhd(seed, checked, img.shape)

#     while len(needs_check) > 0:
#         pt = needs_check.pop()

#         # Its possible that the point was already checked and was
#         # put in the needs_check stack multiple times.
#         if checked[pt]: continue

#         checked[pt] = True

#         # Handle borders.
#         imin = max(pt[0]-t, 0)
#         imax = min(pt[0]+t, img.shape[0]-1)
#         jmin = max(pt[1]-t, 0)
#         jmax = min(pt[1]+t, img.shape[1]-1)
#         kmin = max(pt[2]-t, 0)
#         kmax = min(pt[2]+t, img.shape[2]-1)

#         if img[pt] >= img[imin:imax+1, jmin:jmax+1, kmin:kmax+1].mean():
#             # Include the voxel in the segmentation and
#             # add its neighbors to be checked.
#             seg[pt] = True
#             needs_check += get_nbhd(pt, checked, img.shape)

#     return seg


### Chan-vese level set 3D-segmentation from single point!!! ###

# First need 3D-volume:
img_stack = data

# single 3D-seed point
seed = (num,) + tuple(np.asarray(seedpoint).astype(int))
print(seed)
mask = np.zeros(img_stack.shape)
val = 2
mask[seed[0]-val:seed[0]+val, seed[1]-val:seed[1]+val, seed[2]-val:seed[2]+val] = 1

I = img_stack
init_mask = mask

seg,phi,its = chanvese3d(I,init_mask,max_its=20,alpha=0.2,thresh=0,color='r',display=True)
plt.show()























