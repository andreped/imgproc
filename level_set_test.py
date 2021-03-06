

from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
import drlse_algo as drlse
import numpy as np
from image_functions import maxminscale, imgclean, img2pixels
import cv2


#img = plt.imread('test_end.png')
img = plt.imread('Figure_1_org.png')

img = img[:,:,0]
img = maxminscale(img)
org_img = 1*img # need this later!

# initial seed point -> to grow from
seed_point_full = (int(366.7933774834437), int(315.6754966887417))
reg = 25
img = img[seed_point_full[0]-reg:seed_point_full[0]+reg, seed_point_full[1]-reg:seed_point_full[1]+reg]

seed_point = (reg, reg) # seed point for new smaller image 50x50 -> need to rename later.. np



# parameters 
timestep = 1        # time step (1)
mu = 0.2/timestep   # coefficient of the distance regularization term R(phi)
iter_inner = 4 # (4)
iter_outer = 30 # (30)
lmda = 4            # coefficient of the weighted length term L(phi).   (2)
alfa = -9           # coefficient of the weighted area term A(phi)  (-9)
epsilon = 2.0       # parameter that specifies the width of the DiracDelta function

sigma = 0.8           # scale parameter in Gaussian kernel (0.8)
img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
[Iy, Ix] = np.gradient(img_smooth)
f = np.square(Ix) + np.square(Iy)
g = 1 / (1+f)    # edge indicator function.

# initialize LSF as binary step function
c0 = 2
print(np.amin(img), np.amax(img), img.shape)
initialLSF = c0 * np.ones(img.shape)
# generate the initial region R0 as two rectangles
# initialLSF[24:35, 19:25] = -c0
print(initialLSF.shape)
#initialLSF[24:35, 20:26] = -c0
initialLSF[seed_point] = -c0 # seed point -> to grow from
phi = initialLSF.copy()

plt.ion()
fig1 = plt.figure(1)

def show_fig1():
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)

show_fig1()
fig2 = plt.figure(2)


def show_fig2():
    contours = measure.find_contours(phi, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color = "red")


show_fig2()
print('show fig 2 first time')

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
        fig2.clf()
        show_fig2()
        fig1.clf()
        show_fig1()
        plt.pause(0.3) # 0.3

# refine the zero level contour by further level set evolution with alfa=0
alfa = 0
iter_refine = 10
phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

finalLSF = phi.copy()
print('show final fig 2')
fig2.clf()
show_fig2()
fig1.clf()
show_fig1()

res_filename = '5'
fig2.savefig('test_lvlset_' + res_filename + '.png')
#fig1.savefig('test_lvlset_' + res_filename + 'graph.png')




'''
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
y, x = finalLSF.shape
x = np.arange(0, x, 1)
y = np.arange(0, y, 1)
X, Y = np.meshgrid(x, y)
ax3.plot_surface(X, Y, -finalLSF, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
ax3.contour(X, Y, finalLSF, 0, colors='g', linewidths=2)
'''

#plt.pause(10) #10

#print('yes')

new_blob = 1*finalLSF
new_blob = maxminscale(new_blob)
new_blob = new_blob.astype(np.uint8)

new_blob = cv2.bitwise_not(new_blob)
#print(np.unique(new_blob))
new_blob[new_blob <= 30] = 0
new_blob[new_blob > 30] = 255

#print(np.unique(new_blob))

fig_new = plt.figure(3)
plt.imshow(new_blob, cmap = "gray")
fig_new.savefig('test_lvlset_' + res_filename + 'filled.png')

plt.show()


# now get segmented result!!! :D
#reg_vals = img2pixels(new_blob)
#res_img[seed_point[0]-reg:seed_point[0]+reg, seed_point[1]-reg:seed_point[1]+reg] = new_blob

tempest = cv2.Canny(new_blob, 0, 255) # object -> region!
tempest_full = np.zeros(org_img.shape)

tempest_full[seed_point_full[0]-reg:seed_point_full[0]+reg, seed_point_full[1]-reg:seed_point_full[1]+reg] = tempest
vals = img2pixels(tempest_full)


fig_res, ax = plt.subplots(num = 4, figsize = plt.figaspect(org_img))
fig_res.subplots_adjust(0,0,1,1)
ax.imshow(org_img, cmap="gray")
ax.scatter(vals[:,0], vals[:,1], 0.1, color = 'firebrick')
ax.axis('off')
fig_res.savefig('test_lvlset_' + res_filename + 'segres.png')























