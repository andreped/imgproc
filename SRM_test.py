



from image_functions import imgclean, maxminscale, load_itk, object_filler, cmass, box_minimizer
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
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
import scipy
from skimage.color import rgb2gray
import imageio
from SRM import SRM # (!) <- can't install using pip (!)
from skimage.morphology import disk, binary_erosion, binary_closing, remove_small_holes, remove_small_objects
import h5py


###################################################################################################################
# Try to segment out lung to use it as mask to filter out everything not of interest
###################################################################################################################

# measure time
t = TicToc()
t.tic()


## 1) read data and store as numpy array
# choose patient
pat = input("choose patient: ")

## get data
patient = '/Users/medtek/Documents/Project/Images/patient_data/' + str(pat) + '.hd5'
f = h5py.File(patient, 'r')
Input = f['data']
output = f['label']
data = np.squeeze(Input, axis=3)

# get ground truth data and display at which slice there is a nodule as a list
gt_data = output.value[:,:,:,0]
gt_vals = []
for i in range(gt_data.shape[0]):
	if len(np.unique(gt_data[i,:,:])) > 1:
		gt_vals.append(i)
print(gt_vals)


# choose slice
num = input("choose patient: ")
img = data[int(num),:,:]


img = np.asarray(img)
org_img = maxminscale(img)
org_img = np.uint8(org_img)
org_img = cv2.medianBlur(org_img, 11)


## 2) get 8-bit image
#img = maxminscale(img)
#img = img.astype(np.uint8)
img_orig = 1*img
img_orig = maxminscale(img_orig)

row_size= img.shape[0]
col_size = img.shape[1]

mean = np.mean(img)
std = np.std(img)
img = img-mean
img = img/std

middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
mean = np.mean(middle)  
max = np.max(img)
min = np.min(img)

img[img==max]=mean
img[img==min]=mean

kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img<threshold,1.0,0.0)
print(threshold)

kernel_er = disk(1)
kernel_di = disk(10) # (6) -> img 000003 problem
eroded = morphology.erosion(thresh_img, kernel_er)
dilation = morphology.dilation(eroded, kernel_di)


labels = measure.label(dilation) # Different labels are displayed in different colors
label_vals = np.unique(labels)
regions = measure.regionprops(labels)
good_labels = []
for prop in regions:
    B = prop.bbox
    #print(B)
    if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
        good_labels.append(prop.label)
mask = np.ndarray([row_size,col_size],dtype=np.int8)
mask[:] = 0	
#print(regions)

#print(good_labels)

for N in good_labels:
    mask = mask + np.where(labels==N,1,0)
#mask = morphology.dilation(mask,disk(6)) # one last dilation (6)




## fill regions surrounded by lung regions -> final ROI
mask2 = binary_fill_holes(mask).astype(int)
#mask2 = cv2.bitwise_not(mask2)
mask_lung = morphology.erosion(mask2, disk(9)) #9
#mask_lung = morphology.dilation(mask_lung, disk(8)) #3
#mask_lung = 1*mask2



## use the mask -> get a segmented image only containing the lung AND tumor inside/border
tmp = np.zeros(img_orig.shape)
image = 1*img_orig
image[np.where(mask_lung == 0)] = 255
#image = mask_lung*img_orig
t.toc()


## reshape image to only include what's of interest
val = 10
mask = 1*mask_lung
mask_new, pos1, pos2, pos3, pos4 = box_minimizer(mask, val)
image = image[pos1[0]-val:pos2[0]+val, pos3[1]-val:pos4[1]+val]
#mask_new = 1*mask_lung
#image = mask_new*image


#imgclean(image2)
imgclean(img_orig, Figure=True)

imgclean(image, Figure=True)


#imgclean(image)
#imgclean(mask_new)
#plt.show()

#exit()

'''
imgclean(mask_lung*img_orig)
imgclean(mask2*img_orig)
imgclean(mask*img_orig)
imgclean(img_orig)
plt.show()
'''

#exit()

print(image.shape)
#image = image.astype(int)
image = maxminscale(image)
image = np.uint8(image)

# need 3-color input
#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#image = maxminscale(image)
#image = image.astype(np.uint8)
#image =
#print(image.shape)
#print(np.amin(image), np.amax(image))

#image = image.astype(np.float32)


# rough estimate of the number of expected regions
Q = 500

srm = SRM(image, Q)
segmented = srm.run()
gray = segmented/Q
#gray = rgb2gray(segmented/Q)

print(len(np.unique(gray)))


# plt.figure(3)
# classes, map = srm.map()
# print(classes)
# plt.imshow(map)
# plt.show()

print(gray.shape)
# plt.figure(2)
# plt.imshow(gray[:,:,0]) #"PuBu_r"
# plt.show()

#vals = np.linspace(0,1,256)
#np.random.shuffle(vals)
#cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))

imgclean(gray[:,:,0], Figure=True)

#exit()

















