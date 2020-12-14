import numpy as np


# Input-----------------------------------------------------------------------
# camera parameters
c = 15.     # [mm]
fw = 17.3   # [mm]
fh = 13.    # [mm]
npx = 5280  # [px]
npy = 3956  # [px]

# size of the surveyed area
X = 150.    # [m]
Y = 350.    # [m]

# flight design parameters
h = 40.     # [m]
Rl = .7     # [m]
Rt = .7     # [m]

# accuracy in image collimation
scoll = 1   # [px]

# density of the simulation ground grid
delta = 1.  # resolution of the ground point grid [m]


# compute parameters ---------------------------------------------------------
W = fw*h / c     # footprint width  [m]
H = fh*h / c     # footprint height [m]
GSDw = W / npx   # GSD              [m]
GSDh = H / npy   # GSD (check)      [m]

b = (1-Rl) * H   # baseline
i = (1-Rt) * W   # interaxie

nstrip_y = np.ceil(Y/b)   # number of strips in y
nstrip_x = np.ceil(X/i)   # number of strips in x

# adapt the estimated parameters to uniformly cover the area
b_real = Y / nstrip_y    # real baseline
i_real = X / nstrip_x    # real interaxie
Rl_real = 1 - b_real/H   # real longitudinal overlapping
Rt_real = 1 - i_real/W   # real transversal overlapping

# determine the position of camera acquisition
xo = np.arange(i_real/2, X, i_real)         #                       [m]
yo = np.arange(b_real/2, Y, b_real)         #                       [m]
Xo, Yo = np.meshgrid(xo, yo)                # grid of coordinates   [m]
Zo = h                                      # height of acquisition [m]

# compute the flight path (sort the couple x and y according to the drone path
nstrip_x = nstrip_x.astype(np.int64)
nstrip_y = nstrip_y.astype(np.int64)
flpath = np.zeros((nstrip_x*nstrip_y, 2))  # [Xo, Yo]
for i in range(nstrip_x):
    flpath[i*nstrip_y:(i+1)*nstrip_y, 0] = xo[i]    # [Xo]
    if i % 2 == 1:                                  # [Yo]
        flpath[i*nstrip_y:(i+1)*nstrip_y, 1] = np.flipud(yo)
    else:
        flpath[i*nstrip_y:(i+1)*nstrip_y, 1] = yo

# define the grid of ground points -------------------------------------------
xGrid = np.arange(delta/2, X, delta)                   # vector of x
yGrid = np.arange(delta/2, Y, delta)                   # vector of y coordinates
XGrid, YGrid = np.meshgrid(xGrid, yGrid)               # matrices of all coordinates (couples x and y)
ptName = np.arange(len(XGrid[0])*len(YGrid)).reshape(len(XGrid[0]), len(YGrid))     #index ("name") of each point (to be used in the simulation)

# overlapping map ------------------------------------------------------------
# map of the number of images from which each point is seen
#overlapping in y direction (vector) - longitudinal
over_y = np.zeros(len(yGrid))
for i in range(nstrip_y):
    for j, val in enumerate(yGrid):
        if val >= yo[i]-H/2 and val <= yo[i]+H/2:
            over_y[j] += 1

#overlapping in x direction (vector) - transversal
over_x = np.zeros(len(xGrid))
for i in range(nstrip_x):
    for j, val in enumerate(xGrid):
        if val >= xo[i]-W/2 and val <= xo[i]+W/2:
            over_x[j] += 1

# determine the overlapping maps
Over_x, Over_y = np.meshgrid(over_x, over_y)

# error map (derived from overlapping map) -----------------------------------
s2px = 2 * (scoll*fw/npx)**2                #  sigma2 of the parallax (propagating the sigma of collimation) [mm^2]
s2zy = (h**2 / (c*1e-3 * i_real))**2 * s2px #  sigma2 of the computed z considering longitudinal overlapping [mm^2]
s2zx = (h**2 / (c*1e-3 * b_real))**2 * s2px #  sigma2 of the computed z considering transversal overlapping  [mm^2]

# propagate s2zx and s2zy
sz1 = (((s2zy * (Over_y - 1) + s2zx * (Over_x - 1))/ ((Over_y - 1) + (Over_x - 1))**2))**(.5)


