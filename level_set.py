

import sys, os
import matplotlib.pyplot as plt
from mayavi import mlab
from cv2 import Canny, bilateralFilter, medianBlur, resize, dilate, erode
os.system('cls' if os.name == 'nt' else 'clear')

import numpy as np
import h5py
from pytictoc import TicToc
from tqdm import tqdm
from image_functions import *
from level_set_func import level_set_func
from skimage.morphology import disk, binary_erosion, binary_closing, remove_small_holes, remove_small_objects
import time
from tqdm import tqdm
from imgproc import *
from scipy.ndimage.filters import median_filter
from matplotlib.widgets import Cursor, Button
import Quartz
from matplotlib.patches import Circle
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.exposure import equalize_adapthist
from lungmask_pro import *
from chanvese3d import chanvese3d
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, circle_level_set, inverse_gaussian_gradient, checkerboard_level_set, active_contour
from skimage import img_as_float
#from morphsnakes import 
from print_fix import *
from matplotlib.widgets import Slider, Button
sys.setrecursionlimit(1000000)

blockPrint()
from pymrt import geometry
enablePrint()

from grow_in_3d import level_set3D, region_growing3D



##############################
####### Handle events ########
##############################

# makes plot for seed selection
def slice_plot():

	global img, ax, circ, seed, index, fig

	fig, ax = plt.subplots(num = index, figsize = (12,12))#plt.figaspect(img))
	fig.subplots_adjust(0,0,1,1)
	fig.canvas.mpl_connect('button_press_event', onclick)
	fig.canvas.mpl_connect('figure_enter_event', enter_figure)
	fig.canvas.mpl_connect('figure_leave_event', leave_figure)
	fig.canvas.mpl_connect('key_press_event', remove_seed)
	fig.canvas.mpl_connect('scroll_event', up_scroll)
	fig.canvas.mpl_connect('scroll_event', down_scroll)
	fig.canvas.mpl_connect('key_press_event', up_scroll_alt)
	fig.canvas.mpl_connect('key_press_event', down_scroll_alt)
	plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

	ax.imshow(img, cmap = "gray")
	cursor = Cursor(ax, useblit=True, color='red', linewidth=0.5)
	if (len(circ) > 0):
		for c in range(len(circ)):
			ax.add_patch(Circle((seed[c][1],seed[c][0]), 5, fill = False, color = 'orange', linewidth = 1.5))
	ax.axis('off')

	#plt.draw()
	#plt.pause(10)
	#fig.canvas.draw_idle()
	plt.show()


# function to get seed point from a single click. For each click appends position of click in global list 'seed'
def onclick(event):
	global ix, iy, index, fig, circ, seed, seed3D
	ix, iy = int(event.ydata), int(event.xdata)

	if (img[ix,iy] < -700):
		print(img[ix,iy])
		print('Unusual seed point selected, typically not nodule like. Please select new seed point: ')
	else:
		seed.append((ix, iy))
		seed3D.append((index, ix, iy))

		circ.append(ax.add_patch(Circle((iy,ix), 5, fill = False, color = 'orange', linewidth = 1.5)))
		Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
		plt.close()
		slice_plot()
		#plt.close()


# when entering figure, cursor is invisible
def enter_figure(event):
	global fig, ax
	Quartz.CGDisplayHideCursor(Quartz.CGMainDisplayID())
	#event.canvas.figure


# when leaving figure, cursor is visible
def leave_figure(event):
	global fig, ax
	Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
	#event.canvas.figure


# have the possiblity to remove last seed point if misclicked
def remove_seed(event):
	global seed, fig, index
	if event.key == 'r':
		del seed[-1], circ[-1], seed3D[-1]

		Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
		event.canvas.figure
		plt.close()
		slice_plot()


def up_scroll(event):
	global index, img, fig
	if event.button == 'up':
		if (index + 1 > data.shape[0]):
			print("Whoops, end of stack", print(index))
		else:
			index += 1
			img = data[index]
			Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
			event.canvas.figure
			plt.close()
			slice_plot()


def down_scroll(event):
	global index, img, fig
	if event.button == 'down':
		if (index - 1 < 0):
			print("Whoops, end of stack", print(index))
		else:
			index -= 1
			img = data[index]
			Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
			event.canvas.figure
			plt.close()
			slice_plot()


def up_scroll_alt(event):
	global index, img, fig
	if event.key == "up":
		if (index + 1 > data.shape[0]):
			print("Whoops, end of stack", print(index))
		else:
			index += 1
			img = data[index]
			Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
			event.canvas.figure
			plt.close()
			slice_plot()

def down_scroll_alt(event):
	global index, img, fig
	if event.key == "down":
		if (index - 1 < 0):
			print("Whoops, end of stack", print(index))
		else:
			index -= 1
			img = data[index]
			Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
			event.canvas.figure
			plt.close()
			slice_plot()

# when seed points selected, can begin growing by pressing specific key(s)
def quit_figure(event):
	global fig
	if event.key == ' ' or event.key == 'q':
		Quartz.CGDisplayShowCursor(Quartz.CGMainDisplayID())
		event.canvas.figure
		plt.close()


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


##############################
########### MAIN #############
##############################

if __name__ == "__main__":

	# measure time
	t = TicToc()

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

	# get resolution information about current patient
	f = h5py.File('/Users/medtek/Documents/Project/Images/res_data/res_pat.hd5', 'r')
	resol = list(f['res'])[int(pat)-1]

	# make original data more visually pleasant
	for i in range(data.shape[0]):
		tmp = data[i,:,:].copy()
		tmp[tmp <= -1024] = -1024
		tmp[tmp >= 1024] = 1024
		data[i,:,:] = tmp.copy()


	# choose image for seed selection
	num = int(input("select slice containing tumor: "))
	img = data[num,:,:]


	# select seed point
	print('Please select a seed point: ')

	# init
	seed = []; circ = []; seed3D = []; index = int(num); seg = np.zeros(data.shape)

	# make interactive 2D-viewer for user to select seed point(s)
	slice_plot()

	print(resol)

	# grow in 3D for each seed point selected using 3D chan vese
	for s in tqdm(seed3D):
		new_res = level_set3D(data, s, resol)
		#new_res = level_set3D(data, s)
		#new_res = region_growing3D(data, s)
		seg[new_res == 1] = 1


	# display in which slices there appear to be nodules
	nums = []
	for i in range(seg.shape[0]):
		if (len(np.unique(seg[i,:,:]))-1 > 0):
			nums.append(i)
	print(nums)

	if (nums == []):
		sys.exit('\n No nodules were segmented \n')

	# get ground truth 
	vals = []
	for g in gt_vals:
		if (len(np.unique(gt_data[g,:,:])) > 1) and (len(np.unique(seg[g,:,:])) > 1):
			vals.append(g)

	# display 3D-segmented result for each 2D-slice
	for i in vals:
		fig_res, ax = plt.subplots(num = i, figsize = (12,12))#plt.figaspect(img))
		fig_res.subplots_adjust(0,0,1,1)
		if (len(vals) > 0):
			tmp_vals = im2pixels(Canny(np.uint8(maxminscale(gt_data[i,:,:])), 0, 255))
			if (tmp_vals == []):
				tmp_vals = im2pixels(gt_data[i,:,:])
			ax.imshow(seg[i,:,:], cmap="inferno")
			pred_vals = im2pixels(Canny(np.uint8(maxminscale(seg[i,:,:])), 0, 255))
			ax.imshow(data[i,:,:], cmap="gray", alpha = 0.7)
			#ax.scatter(pred_vals[:,0], pred_vals[:,1], 1, color = 'orange', marker = '.')
			ax.scatter(tmp_vals[:,0], tmp_vals[:,1], 1, color = 'red', marker = '.')
			ax.axis('off')
		plt.show()


	# 3D-visualization of segmented output
	plot_lung_and_tumor(data, seg, gt_data, thr = 0.5)




'''


if __name__ == "__main__":

	# measure time
	t = TicToc()

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

	# make original data more visually pleasant
	for i in range(data.shape[0]):
		tmp = data[i,:,:].copy()
		tmp[tmp <= -1024] = -1024
		tmp[tmp >= 1024] = 1024
		data[i,:,:] = tmp.copy()


	#data = np.uint8(maxminscale(data))

	# choose image for seed selection
	num = int(input("select slice containing tumor: "))
	img = data[num,:,:]


	# select seed point
	print('Please select a seed point: ')

	# init
	seed = []; circ = []; seed3D = []; index = int(num)

	# select seed points from slice
	#fig, ax = plt.subplots(num = index, figsize = (12,12))#plt.figaspect(img))

	slice_plot()
	#plt.show()

	new_res = np.zeros(data.shape)

	# init
	num_slices = 30 # number of slices (chunk) sliced oriented around seed point
	reg = 30 # window of interest centered around seed point
	jj = 0 # to traverse 3D seed point list
	# for each seed point:
	for s in tqdm(seed):
		num = seed3D[jj][0]
		jj += 1

		# apply lungmask on each slice of interest around seed point
		tmp_data = np.zeros((num_slices, data.shape[1],data.shape[2]))
		ii = 0
		for i in range(data.shape[0]):
			if (i >= num - num_slices/2 and i <= num + num_slices/2-1):
				mask = lungmask(data[i,:,:].copy())
				tmp = data[i,:,:].copy()
				tmp[mask == 0] = np.amin(tmp)
				#tmp = median_filter(tmp.copy(), 5)
				#tmp = bilateralFilter(tmp.astype(np.float32), 3, 25, 25)
				#tmp = anisodiff(tmp.copy(),niter=10,kappa=50,gamma=0.2,step=(1.,1.))
				tmp_data[ii,:,:] = tmp
				ii += 1

		# only get volume of interest relevant for the level set -> increases speec and accuracy
		tmp = tmp_data.copy()
		tmp = tmp[:, s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg]

		# new 3D seed point:
		#seed3D.append((int(num_slices/2), reg, reg))

		#init_mask = np.zeros(tmp.shape)
		#val = 5
		#init_mask[int(num_slices/2)-5:int(num_slices/2)+5+1, 30-val:30+val+1, 30-val:30+val+1] = 1
		#init_mask[sphere((num_slices,60,60), val, (25,30,30))] = 1

		# apply anisotropic diffusion (smoothing) on current stack
		#print(np.amax(tmp))
		# niter = 10 # number of iterations -> higher the number, higher the smoothing
		# kappa = 25 # conductance coefficent (typically 20 - 100)
		# gamma = 0.2 # max value of 0.25 for stability
		# step = (1.,1.,2.) # distance between adjacent pixels (resolution important for 3D smoothing!)
		# tmp = anisodiff3(tmp.copy(), niter=5, kappa=9, gamma=0.1, step=(1.,1.,1.))
		# # print(np.amax(tmp2))
		# tmp = np.uint8(maxminscale(tmp))

		# for i in range(tmp.shape[0]):
		# 	fig_res, ax = plt.subplots(1,2)#plt.figaspect(img))
		# 	ax[0].imshow(tmp[i], cmap="gray")
		# 	ax[1].imshow(tmp2[i], cmap = "gray")
		# 	#ax.axis('off')
		# 	plt.show()

		# tmp = tmp2.copy()

		# for i in range(tmp.shape[0]):
		# 	tmp[i] = inverse_gaussian_gradient(tmp[i], alpha = 100, sigma = 2)

		#tmp = maxminscale(tmp)
		tmp = img_as_float(tmp)
		tmp = inverse_gaussian_gradient(tmp.copy(), alpha = 80, sigma = 0.8) # (0.8) if threshold = 'auto'

		#tmp = inverse_gaussian_gradient(tmp.copy())

		# for i in range(tmp.shape[0]):
		# 	plt.figure(num = num - int(num_slices/2)+i)
		# 	plt.imshow(tmp[i], cmap = "gray")
		# 	plt.show()
		# print(np.amin(tmp), np.amax(tmp))

		### apply chan-vese's level set in 3D -> Use M-ACWE instead of M-GAC because easier to use and tune ###
		#inits = checkerboard_level_set(tmp.shape, square_size=4)
		#tmp = morphological_chan_vese(tmp.copy(), iterations = 1000, init_level_set = inits, smoothing = 1, lambda1 = 1, lambda2 = 1)
		#tmp = morphological_chan_vese(tmp.copy(), iterations = 1000, init_level_set = 'circle', smoothing = 0, lambda1 = 1, lambda2 = 1)

		inits = circle_level_set(tmp.shape, center = (int(num_slices/2), reg, reg), radius = 2)
		#inits = checkerboard_level_set(tmp.shape, 10)

		#inits = np.zeros(tmp.shape, dtype=np.int8)
		#inits[15:-10, 15:-10] = 1

		tmp = morphological_geodesic_active_contour(tmp.copy(), iterations = 400, init_level_set=inits, smoothing=0, threshold='auto', balloon=1)#, iter_callback = callback)
		#tmp = morphological_chan_vese(tmp.copy(), iterations = 500, init_level_set = inits, smoothing = 0, lambda1 = 1, lambda2 = 1)
		#tmp = morphological_chan_vese(tmp.copy(), iterations = 2000, init_level_set = 'circle', smoothing = 0, lambda1 = 2, lambda2 = 1)
		#tmp, phi, its = chanvese3d(tmp.copy(),init_mask,max_its=100,alpha=0.01,thresh=0,color='r',display=False)
		tmp = tmp.astype(int)

		# if not removing smaller voxels not connected to original seed point (8-connected)
		res = tmp.copy()

		# only keep object 8-connected to the seed point (remove extra smaller segments)
		# labels = label(tmp)
		# res = np.zeros(tmp.shape)
		# if (labels[seed3D[jj]] > 0):
		# 	res[labels == labels[seed3D[jj]]] = 1


		# apply opening to separate blood vessels from nodules
		kernel = sphere((5,5,5), 1, (2,2,2))
		kernel = geometry.sphere(shape = (2,2,2), radius = 1, position = 0.5)
		# kernel = geometry.ellipsoid(shape = (2,2,2), semiaxes = (1,1,1), position = 0.5)
		# # # kernel = geometry.ellipsoid(shape = (2,4,4), semiaxes = (1,2,2), position = 0.5)

		# post processing -> do some smoothing
		kernel = sphere((5,5,5), 2.5, (2,2,2))
		#tmp = ndimage.morphology.grey_closing(tmp.copy(), structure = kernel)
		#kernel = sphere((5,5,5), 1.5, (2,2,2))
		#tmp2 = ndimage.morphology.grey_opening(tmp.copy(), structure = kernel)
		#labels_tmp2 = label(tmp)

		# tmp2 = ndimage.morphology.grey_opening(res.copy(), structure = kernel)
		# labels_tmp2 = label(tmp2.copy())
		# res = np.zeros(tmp2.shape)
		# if (labels_tmp2[int(num_slices/2), reg, reg] > 0):
		# 	res[labels_tmp2 == labels_tmp2[int(num_slices/2), reg, reg]] = 1

		# get the final nodule mask to the original image stack shape
		# but handle cases where seed point is selected at ends of stack and sliced chunk is less than given
		if (num+int(num_slices/2) > new_res.shape[0]):
			new_res[num-int(num_slices/2):num+int(num_slices/2), s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg] = res[:num+int(num_slices/2)-new_res.shape[0]]
		elif (num-int(num_slices/2) < 0):
			new_res[0:num+int(num_slices/2), s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg] = res[:num+int(num_slices/2)]
		else:
			new_res[num-int(num_slices/2):num+int(num_slices/2), s[0]-reg:s[0]+reg, s[1]-reg:s[1]+reg] = res


	# display in which slices there appear to be nodules
	nums = []
	for i in range(new_res.shape[0]):
		if (len(np.unique(new_res[i,:,:]))-1 > 0):
			nums.append(i)
	print(nums)

	if (nums == []):
		sys.exit('\n No nodules were segmented \n')

	# get ground truth 
	vals = []
	for g in gt_vals:
		if (len(np.unique(gt_data[g,:,:])) > 1) and (len(np.unique(new_res[g,:,:])) > 1):
			vals.append(g)

	# display 3D-segmented result for each 2D-slice
	for i in vals:
		fig_res, ax = plt.subplots(num = i, figsize = (12,12))#plt.figaspect(img))
		fig_res.subplots_adjust(0,0,1,1)
		if (len(vals) > 0):
			tmp_vals = im2pixels(Canny(np.uint8(maxminscale(gt_data[i,:,:])), 0, 255))
			ax.imshow(new_res[i,:,:], cmap="inferno")
			ax.imshow(data[i,:,:], cmap="gray", alpha = 0.7)
			ax.scatter(tmp_vals[:,0], tmp_vals[:,1], 1, color = 'red', marker = '.')
			ax.axis('off')
		plt.show()


	# 3D-visualization of segmented output
	plot_lung_and_tumor(data, new_res, gt_data, thr = 0.5)


'''





