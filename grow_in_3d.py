

import numpy as np
from lungmask_pro import *
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, circle_level_set, inverse_gaussian_gradient, checkerboard_level_set
from skimage.measure import label
from scipy import ndimage
import skimage
from scipy.ndimage.interpolation import zoom

import warnings 
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

from print_fix import *
blockPrint()
from pymrt import geometry
enablePrint()

sys.setrecursionlimit(1000000)


'''
A set of functions relevant for 3D lung nodules segmentation. Two main functions: level_set3D and region_growing3D,
which both can take same CT-image stack, but only one seed-point, since it was of interest at the time. Also includes 
two 3D-kernels, which might come of interest if doing 3D-morphology. I would also recommend using the PYMRT-module, 
since it contains TONS of 2D and 3D-kernels. Also skimage should have a few 3D-kernels, but lacks a spheroid/ellipsoid
kernel, which was of interest at the time.
NB: Handle with caution. Default settings should hopefully rarely be changed, unless you are studying for instance
really tiny nodules, want a more specific fit for a single nodule, or if smoothing is to strong etc...
------------------------------------------------------------------------------------------------------------
Made by: André Pedersen
e-mail : ape107@post.uit.no
'''


# --- INPUT ---
# data   : 3D CT-image stack, with assumed dimensions (z,y,x)
# seed3D : single 3D seed point corresponding to image dimensions
# resol  : list containing resolution info about CT-stack (z,x), assumed same res in x and y
# lambda1 : variance stabilizing coefficient. 
# lambda2 : second variance stabilizing coefficient. lambda1 and lambda2 defines ideal segmentation result
# smoothing : number of smoothing steps for each iteration
# iterations : number of iterations
# rad : radius for initial curve/surface. Defines the initial sphere from seed point
# method : method of level set used
# alpha : how steep the edges for the pre-processing step of GAC should be (isn't used in ACWE, only GAC)
# sigma : the standard deviation of the Gaussian kernel -> smoothing coefficient (isn't used in ACWE, only GAC)
# balloon : defines the balloon force. 1 -> grows, -1 : shrinks (isn't used in ACWE, only GAC)
# --- OUTPUT ---
# new_res : result of 3D segmentation from single seed point

def level_set3D(data, seed3D, resol, lambda1=1, lambda2=4, smoothing = 0, iterations = 100, rad = 3, method = 'ACWE', alpha = 100, sigma = 2, balloon = 1):

	## init
	res_fac = resol[1]/resol[0] # resolution factor to scale chunk to real dimensions
	N = 60 # approximately the chunk size (in mm) around nodule
	num_slices = int(round(N/res_fac)) # number of slices before 
	reg = 30 # window of interest centered around seed point
	s = seed3D[1:] # 2D seed point for (x,y)-plane
	num = seed3D[0] # slice number seed point is selected from

	# apply lungmask on each slice of interest around seed point
	tmp_data = np.zeros((num_slices, data.shape[1],data.shape[2]))
	ii = 0
	for i in range(data.shape[0]):
		if (i >= num - int(round(num_slices/2)) and i <= num + int(round(num_slices/2))-1):
			mask = lungmask_pro(data[i,:,:].copy())
			tmp = data[i,:,:].copy()
			tmp[mask == 0] = np.amin(tmp) # OBS: -1024 maybe not far away enough from lung intensities
			tmp_data[ii,:,:] = tmp.copy()
			ii += 1

	# only apply level set on small volume around seed point -> increases speed and accuracy for fixed number of iterations
	tmp = tmp_data.copy()
	tmp = tmp[:, s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg]

	# transform chunk to true size (in mm) by stretching the slice-axis relevant to a resolution factor
	tmp = zoom(tmp.copy(), zoom = [res_fac, 1, 1], order = 1)

	# apply 3D level set from single seed point from initial 3D blob around current seed point
	inits = circle_level_set(tmp.shape, center = (int(num_slices/2*res_fac), reg, reg), radius = rad)

	# choose between two types of level set methods
	if method == 'ACWE':
		tmp = morphological_chan_vese(tmp.copy(), iterations = iterations, init_level_set = inits, smoothing = smoothing, lambda1 = lambda1, lambda2 = lambda2).astype(int)
	elif method == 'GAC':
		tmp = tmp.astype(np.float32)
		tmp = inverse_gaussian_gradient(tmp, alpha = alpha, sigma = alpha)
		tmp = morphological_geodesic_active_contour(tmp, iterations = iterations, init_level_set=inits, smoothing=smoothing, threshold='auto', balloon=1)
	else:
		print('Please choose a valid method!')
		return None

	# if no nodule was segmented, break
	if (len(np.unique(tmp)) == 1):
		#print('No nodule was segmented. Try changing parameters...')
		return None

	# check if leakage has occured
	#if ((tmp[0,0,0] > 0) or (tmp[0,0,-1] > 0) or (tmp[0,-1,0] > 0) or (tmp[0,-1,-1] > 0) or (tmp[-1,0,0] > 0) or (tmp[-1,-1,0] > 0) or (tmp[-1,0,-1] > 0) or (tmp[-1,-1,-1] > 0)):
	# if ((len(np.unique(tmp[0,:,:])) > 1) or (len(np.unique(tmp[:,0,:])) > 1) or (len(np.unique(tmp[:,:,0])) > 1) or
	#  (len(np.unique(tmp[-1,:,:])) > 1) or (len(np.unique(tmp[:,-1,:])) > 1) or (len(np.unique(tmp[:,:,-1])) > 1)):
	# 	print("Leakage problems? Growing reached boundaries... Discards segmentation")
	# 	return None

	# only keep segments connected to seed point (blood vessels will hopefully not be connected with nodule after level set, if leakage has occured)
	labels_tmp = label(tmp.copy())
	res = np.zeros(tmp.shape)
	if (labels_tmp[int(num_slices/2*res_fac), reg, reg] > 0):
		res[labels_tmp == labels_tmp[int(num_slices/2*res_fac), reg, reg]] = 1

	# need to transform chunk back to original size
	res = zoom(res.copy(), zoom = [1/res_fac, 1, 1], order = 1)

	# # just in case some parts are not connected anymore after interpolation -> remove not connected components
	# labels_tmp = label(res.copy())
	# res = np.zeros(res.shape)
	# if (labels_tmp[int(num_slices/2), reg, reg] > 0):
	# 	res[labels_tmp == labels_tmp[int(num_slices/2), reg, reg]] = 1

	# get the final nodule mask to the original image stack shape
	# but handle cases where seed point is selected at ends of image stack, and window is outside of range
	new_res = np.zeros(data.shape)
	if (num+int(num_slices/2) > new_res.shape[0]):
		new_res[num-int(num_slices/2):num+int(num_slices/2), s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg] = res[:num+int(num_slices/2)-new_res.shape[0]]
	elif (num-int(num_slices/2) < 0):
		new_res[0:num+int(num_slices/2), s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg] = res[:num+int(num_slices/2)]
	else:
		new_res[num-int(np.floor(num_slices/2)):num+int(np.ceil(num_slices/2)), s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg] = res

	return new_res



# 3D-region growing algorithm
# --- INPUT ---
# data     : 3D-data on the form (z,y,x))
# seed     : single initial 3D seed point.
# r_max    : maximum radius allowed for growth from seed point
# min_th   : minimum threshold for growth -> CT-number of lung (air)
# min-grad : minimum negative local gradient allowed
# morph    : whether or not to apply morphology to fix segment (split nodule from vessels)
# --- OUTPUT ---
# res  : Binary 3D-array result of RG. Follow the same structure as data
def region_growing3D(data, s, r_max = 30, min_th = -700, min_grad = -80, morph = True):

	# get slice number
	num = s[0]

	# apply lungmask on data -> better performance
	tmp_data = np.zeros(data.shape)
	for i in range(data.shape[0]):
		if (i >= num - 25 and i <= num + 25):
			mask = lungmask_pro(data[i,:,:].copy())
			tmp = data[i,:,:].copy()
			tmp[mask == 0] = np.amin(tmp)
			tmp_data[i,:,:] = tmp
	data = tmp_data.copy()

	# initialize output -> pixels to be updated if accepted in growing process
	res = np.zeros_like(data)

	# put initial seed points in res
	res[s] = 1

	# to store values in queue to grow from later
	queue = [s]

	# boundary image -> given by maximum radius allowed for region growing (r = 15 => d = 30)
	grow_range = sphere(data.shape, r_max, s).astype(int)

	# current position
	curr = queue[0] # get initial seed point from queue

	# set intensity representative from mean intensity of sphere around initial seed point
	seed_int = data[s]

	# counter for how many pixels have been included in the growing process
	cnt = 1

	# start region growing process
	while (len(queue) > 0):

		# point to grow from
		curr = queue[0]

		# remove the point from queue
		queue.pop(0)

		# for all 8-connected neighbours
		# 3D-growing dependent on volume size, 8-connectivity and minimum allowed intensity
		for i in range(-1,2):
			for j in range(-1,2):
				for k in range(-1,2):
					# should not be allowed to have a negative large gradient, but upwards is OK
					loc_grad = data[curr] - data[curr[0]+k ,curr[1]+i, curr[2]+j]

					if ((curr[0]+i > 0) and (curr[0]+i <= data.shape[0]) and
						(curr[1]+j > 0) and (curr[1]+j <= data.shape[2]) and
						(curr[2]+j > 0) and (curr[2]+j <= data.shape[1]) and
						(i + j + k != 0) and
						(res[curr[0]+k ,curr[1]+i, curr[2]+j] == 0) and
						(grow_range[curr[0]+k ,curr[1]+i, curr[2]+j] != 0) and
						(data[curr[0]+k ,curr[1]+i, curr[2]+j] >= min_th) and
						(loc_grad >= min_grad)):

						# if accepted
						res[curr[0]+k ,curr[1]+i, curr[2]+j] = 1
						queue.append((curr[0]+k ,curr[1]+i, curr[2]+j))
						cnt += 1

					# elif (grow_range[curr[0]+k ,curr[1]+i, curr[2]+j] == 0):
					# 	print("Leakage problems? Growing reached boundaries... Discards segmentation")
					# 	return None

	# post processing -> apply opening to split nodule and vessels/bronchi
	if morph == True:
		kernel = sphere((5,5,5), 1, (2,2,2))
		tmp = ndimage.morphology.grey_closing(res.copy(), structure = kernel)
		kernel = sphere((5,5,5), 1.5, (2,2,2))
		tmp2 = ndimage.morphology.grey_opening(tmp.copy(), structure = kernel)
		labels_tmp2 = label(tmp2)

		res = np.zeros(tmp2.shape)
		if (labels_tmp2[s] > 0):
			res[labels_tmp2 == labels_tmp2[s]] = 1

	return res




# function which makes a (3D) spherical kernel of user defined radius at specified position in the given shape
# --- INPUT ----
# shape    : output size of kernel
# radius   : radius of sphere
# position : centre point of sphere. Where it is placed in the space 'shape'
# --- OUTPUT ---
# arr : 3D-numpy array for a sphere with type bool
def sphere(shape, radius, position):
    # init : assume shape and position are both a 3-tuple of int or float
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]

    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0



# function which makes a (3D) spheroid kernel of user defined semi-axes
# --- INPUT ----
# radxy    : semi-axis in xy-plane (symmetric - assumed same res in x and y)
# radius   : semi-axis along z (slice direction)
# --- OUTPUT ---
# arr : 3D-numpy array for a spheroid with type uint8
def spheroid(radxy, radz, dtype=np.uint8):
	m = 2 * radz + 1
	n = 2 * radxy + 1
	Z, Y, X = np.mgrid[-radz:radz:m * 1j,
                       -radxy:radxy:n * 1j,
                       -radxy:radxy:n * 1j]
	s = (X/radxy) ** 2 + (Y/radxy) ** 2 + (Z/radz) ** 2
	return np.array(s <= 1, dtype=dtype)







