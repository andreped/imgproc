

import numpy as np
from skimage.morphology import disk
#import cv2
from cv2 import Canny, bilateralFilter, medianBlur, resize, dilate, erode



# Fast binary region growing (floodfill idea). Doesn't work on anything else than binary images
# --- INPUT ---
# img    : 2D-numpy array, for instance an image
# seed   : initial seed point to grow from
# --- OUTPUT ---
# outimg : Binary 2D-array result of RG
def binary_region_growing(img, seed):
	queue = []
	outimg = np.zeros_like(img)
	queue.append((seed[0], seed[1]))
	processed = []
	while(len(queue) > 0):
		pix = queue[0]
		outimg[pix[0], pix[1]] = 255
		for coord in get8n(pix[0], pix[1], img.shape):
			if img[coord[0], coord[1]] != 0:
				outimg[coord[0], coord[1]] = 255
				if not coord in processed:
					queue.append(coord)
				processed.append(coord)
		queue.pop(0)
	return outimg




# 2D-region growing algorithm
# --- INPUT ---
# img  : 2D-numpy array, for instance an image
# seed : initial seed point. Choice of point is clinical for the algorithm as growing depends on it
# --- OUTPUT ---
# res  : Binary 2D-array result of RG. Follow the same structure as img
def region_growing2D(img, seed):
	# maximum radius allowed (actually d = 2*r+1, because centre value discarded)
	r_max = 30

	# allowed deviation from threshold (measure of how much the intensity can change away from the initial threshold)
	#seed_dev = 300 #550 #300 # (20 if maxminscaled)

	# minimum threshold for growth -> CT-number of lung (air)
	min_th = -600

	# initialize output -> pixels to be updated if accepted in growing process
	res = np.zeros_like(img)

	# possible to grow from several seed points :)
	for s in seed:

		# put initial seed points in res
		res[s] = 1

		# to store values in queue to grow from later
		queue = [s]

		# boundary image -> given by maximum radius allowed for region growing (r = 15 => d = 30)
		grow_range = np.zeros_like(img)
		grow_range[s[0]-r_max:s[0]+r_max+1, s[1]-r_max:s[1]+r_max+1] = disk(r_max)

		# current position
		curr = queue[0] # get initial seed point from queue

		# intensity of initial seed point
		seed_int = img[s]

		# counter for how many pixels have been included in the growing process
		cnt = 1

		# start region growing process
		while (len(queue) > 0):

			# point to grow from
			curr = queue[0]

			# remove the point from queue
			queue.pop(0)

			# for all 8-connected neighbours
			for i in range(-1,2):
				for j in range(-1,2):
					if ((curr[0]+i > 0) and (curr[0]+i <= img.shape[1]) and
						(curr[1]+j > 0) and (curr[1]+j <= img.shape[0]) and
						(i + j != 0) and
						(res[curr[0]+i, curr[1]+j] == 0) and
						(grow_range[curr[0]+i, curr[1]+j] != 0) and
						(img[curr[0]+i, curr[1]+j] >= min_th)):
						#(img[curr[0]+i, curr[1]+j] <= seed_int + seed_dev) and 
						#(img[curr[0]+i, curr[1]+j] >= seed_int - seed_dev)):

						# if accepted
						res[curr[0]+i, curr[1]+j] = 1
						queue.append((curr[0]+i, curr[1]+j))
						cnt += 1

		#print(cnt)

	return res



 

# 3D-region growing algorithm
# --- INPUT ---
# data : 3D-data on the form (num slices, (image shape x,y))
# seed : initial seed point. Choice of point is clinical for the algorithm as growing depends on it
# --- OUTPUT ---
# res  : Binary 3D-array result of RG. Follow the same structure as data
def region_growing3D(data, seed):
	# maximum radius allowed (actually d = 2*r+1, because centre value discarded)
	r_max = 30

	# allowed deviation from threshold (measure of how much the intensity can change away from the initial threshold)
	#seed_dev = 600
	#seed_dev = 200 #550 # 550 better for single slice in 3D, but worse for some pat (ex. pat=2)

	# minimum threshold for growth -> CT-number of lung (air)
	min_th = -750

	# initialize output -> pixels to be updated if accepted in growing process
	res = np.zeros_like(data)

	# possible to grow several from several seed points :)
	for s in seed:

		# put initial seed points in res
		res[s] = 1

		# to store values in queue to grow from later
		queue = [s]

		# boundary image -> given by maximum radius allowed for region growing (r = 15 => d = 30)
		grow_range = sphere(data.shape,r_max,s).astype(int)

		# current position
		curr = queue[0] # get initial seed point from queue

		# set intensity representative from mean intensity of sphere around initial seed point
		#tmp_rad = 1
		#seed_int = np.mean(data[s[0]-tmp_rad:s[0]+tmp_rad+1, s[1]-tmp_rad:s[1]+tmp_rad+1, s[2]-tmp_rad:s[2]+tmp_rad+1])
		seed_int = data[s]

		# get gradient of volume
		# grad = np.array(np.gradient(data))
		# print(np.amin(grad), np.amax(grad), np.mean(grad))
		# print(grad.shape)

		# counter for how many pixels have been included in the growing process
		cnt = 1

		# start region growing process
		while (len(queue) > 0):

			# point to grow from
			curr = queue[0]

			# remove the point from queue
			queue.pop(0)

			#print(curr)

			# for all 8-connected neighbours
			# 3D-growing dependent on volume size, 8-connectivity and minimum allowed intensity
			for i in range(-1,2):
				for j in range(-1,2):
					for k in range(-1,2):
						# should not be allowed to have a negative large gradient, but upwards is OK
						loc_grad = data[curr] - data[curr[0]+k ,curr[1]+i, curr[2]+j] #np.abs(data[curr] - data[curr[0]+k ,curr[1]+i, curr[2]+j])
						#print(loc_grad)
						if ((curr[0]+i > 0) and (curr[0]+i <= data.shape[0]) and
							(curr[1]+j > 0) and (curr[1]+j <= data.shape[2]) and
							(curr[2]+j > 0) and (curr[2]+j <= data.shape[1]) and
							(i + j + k != 0) and
							(res[curr[0]+k ,curr[1]+i, curr[2]+j] == 0) and
							(grow_range[curr[0]+k ,curr[1]+i, curr[2]+j] != 0) and
							(data[curr[0]+k ,curr[1]+i, curr[2]+j] >= min_th) and
							(loc_grad >= -50)):
							#(data[curr[0]+k ,curr[1]+i, curr[2]+j] <= seed_int + seed_dev) and 
							#(data[curr[0]+k ,curr[1]+i, curr[2]+j] >= seed_int - seed_dev)):

							# if accepted
							res[curr[0]+k ,curr[1]+i, curr[2]+j] = 1
							queue.append((curr[0]+k ,curr[1]+i, curr[2]+j))
							cnt += 1

	return res


# region_growing3D 



# get slope between two points, but only dep. on positions, NOT intensity
def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m



# get 8-neighbours for 2D-point -> useful for watersheds, region growing, edge-linkage etc...
# -> stores all pixel positions around centre in a list
# --- INPUT ---
# (x,y) : location of centre pixel
# shape : how large the mask should be
# --- OUTPUT ---
# out   : list containing position of all neighbours of 8-connectivity
def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out



# function which makes a (3D) spherical kernel of user defined radius at specified position in the given shape
# --- INPUT ----
# shape    : output size of kernel
# radius   : radius of sphere
# position : centre point of sphere. Where it is placed in the space 'shape'
# --- OUTPUT ---
# arr : 3D-numpy array for a sphere with type bool
def sphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
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



# Function that does anisotropic diffusion (edge-preserving smoothing method) on single (2D) image
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
        """
        Anisotropic diffusion.
 
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
 
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration
 
        Returns:
                imgout   - diffused image.
 
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
 
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
 
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
 
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
 
        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
 
        Original MATLAB code by Peter Kovesi  
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>
 
        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>
 
        June 2000  original version.      
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
 
        # ...you could always diffuse each color channel independently if you
        # really want
        if img.ndim == 3:
                warnings.warn("Only grayscale images allowed, converting to 2D matrix")
                img = img.mean(2)
 
        # initialize output array
        img = img.astype('float32')
        imgout = img.copy()
 
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
 
        # create the plot figure, if requested
        if ploton:
                import pylab as pl
                from time import sleep
 
                fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
                ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
                ax1.imshow(img,interpolation='nearest')
                ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
                ax1.set_title("Original image")
                ax2.set_title("Iteration 0")
 
                fig.canvas.draw()
 
        for ii in range(niter): # OBS! Had to change xrange -> range
 
                # calculate the diffs
                deltaS[:-1,: ] = np.diff(imgout,axis=0)
                deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
                # conduction gradients (only need to compute one per dim!)
                if option == 1:
                        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                        gE = np.exp(-(deltaE/kappa)**2.)/step[1]
                elif option == 2:
                        gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                        gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
                # update matrices
                E = gE*deltaE
                S = gS*deltaS
 
                # subtract a copy that has been shifted 'North/West' by one
                # pixel. don't as questions. just do it. trust me.
                NS[:] = S
                EW[:] = E
                NS[1:,:] -= S[:-1,:]
                EW[:,1:] -= E[:,:-1]
 
                # update the image
                imgout += gamma*(NS+EW)
 
                if ploton:
                        iterstring = "Iteration %i" %(ii+1)
                        ih.set_data(imgout)
                        ax2.set_title(iterstring)
                        fig.canvas.draw()
                        # sleep(0.01)
 
        return imgout



# Function that does anisotropic diffusion (edge-preserving smoothing method) on 3D image stack
def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):
	"""
	3D Anisotropic diffusion.

	Usage:
	stackout = anisodiff(stack, niter, kappa, gamma, option)

	Arguments:
	        stack  - input stack
	        niter  - number of iterations
	        kappa  - conduction coefficient 20-100 ?
	        gamma  - max value of .25 for stability
	        step   - tuple, the distance between adjacent pixels in (z,y,x)
	        option - 1 Perona Malik diffusion equation No 1
	                 2 Perona Malik diffusion equation No 2
	        ploton - if True, the middle z-plane will be plotted on every 
	        	 iteration

	Returns:
	        stackout   - diffused stack.

	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.

	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)

	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x,y and/or z axes

	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.

	Reference: 
	P. Perona and J. Malik. 
	Scale-space and edge detection using ansotropic diffusion.
	IEEE Transactions on Pattern Analysis and Machine Intelligence, 
	12(7):629-639, July 1990.

	Original MATLAB code by Peter Kovesi  
	School of Computer Science & Software Engineering
	The University of Western Australia
	pk @ csse uwa edu au
	<http://www.csse.uwa.edu.au>

	Translated to Python and optimised by Alistair Muldal
	Department of Pharmacology
	University of Oxford
	<alistair.muldal@pharm.ox.ac.uk>

	June 2000  original version.       
	March 2002 corrected diffusion eqn No 2.
	July 2012 translated to Python
	"""

	# ...you could always diffuse each color channel independently if you
	# really want
	if stack.ndim == 4:
		warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
		stack = stack.mean(3)

	# initialize output array
	stack = stack.astype('float32')
	stackout = stack.copy()

	# initialize some internal variables
	deltaS = np.zeros_like(stackout)
	deltaE = deltaS.copy()
	deltaD = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	UD = deltaS.copy()
	gS = np.ones_like(stackout)
	gE = gS.copy()
	gD = gS.copy()

	# create the plot figure, if requested
	if ploton:
		import pylab as pl
		from time import sleep

		showplane = stack.shape[0]//2

		fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
		ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

		ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
		ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
		ax1.set_title("Original stack (Z = %i)" %showplane)
		ax2.set_title("Iteration 0")

		fig.canvas.draw()

	for ii in np.arange(1,niter):

		# calculate the diffs
		deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
		deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
		deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

		# conduction gradients (only need to compute one per dim!)
		if option == 1:
			gD = np.exp(-(deltaD/kappa)**2.)/step[0]
			gS = np.exp(-(deltaS/kappa)**2.)/step[1]
			gE = np.exp(-(deltaE/kappa)**2.)/step[2]
		elif option == 2:
			gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
			gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
			gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

		# update matrices
		D = gD*deltaD
		E = gE*deltaE
		S = gS*deltaS

		# subtract a copy that has been shifted 'Up/North/West' by one
		# pixel. don't as questions. just do it. trust me.
		UD[:] = D
		NS[:] = S
		EW[:] = E
		UD[1:,: ,: ] -= D[:-1,:  ,:  ]
		NS[: ,1:,: ] -= S[:  ,:-1,:  ]
		EW[: ,: ,1:] -= E[:  ,:  ,:-1]

		# update the image
		stackout += gamma*(UD+NS+EW)

		if ploton:
			iterstring = "Iteration %i" %(ii+1)
			ih.set_data(stackout[showplane,...].squeeze())
			ax2.set_title(iterstring)
			fig.canvas.draw()
			# sleep(0.01)

	return stackout










