import numpy as np
from numpy.random import randn
from scipy.sparse import csr_matrix


def normal_case(drone, sensor, X, Y, h, Rl, Rt, scoll):
    # Input-----------------------------------------------------------------------
    # camera parameters
    shooting_int = sensor['ShootInterval']
    UAS_v = drone['MaxSpeed']
    c = sensor['FocalLength']  # [mm]
    fw = sensor['SizeX']       # [mm]
    fh = sensor['SizeY']       # [mm]
    npx = sensor['ImgSizeX']   # [px]
    npy = sensor['ImgSizeY']   # [px]

    # density of the simulation ground grid
    delta = 1.  # resolution of the ground point grid [m]

    # compute parameters ---------------------------------------------------------
    W = fw*h / c     # footprint width  [m]
    H = fh*h / c     # footprint height [m]
    GSDw = W / npx   # GSD              [m]
    GSDh = H / npy   # GSD (check)      [m]

    b = (1-Rl) * H   # baseline
    interaxie = (1-Rt) * W   # interaxie

    nstrip_y = np.ceil(Y/b)   # number of strips in y
    nstrip_x = np.ceil(X/interaxie)   # number of strips in x

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
    ptName = np.arange(len(XGrid[0])*len(YGrid)).reshape((len(YGrid), len(XGrid[0])), order='F')     #index ("name") of each point (to be used in the simulation)
    #ptName = np.arange(len(XGrid[0])*len(YGrid)).reshape(len(XGrid[0]), len(YGrid))
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

    num_images = nstrip_y * nstrip_x

    pixel_size = fw / npx
    UAS_v_min = b / shooting_int
    max_distance = UAS_v * 60 * UAS_v_min
    max_distance_proj = nstrip_x * b * nstrip_y + b * i
    # propagate s2zx and s2zy
    #sz1 = (((s2zy * (Over_y - 1) + s2zx * (Over_x - 1))/ ((Over_y - 1) + (Over_x - 1))**2))**(.5)
    sigma_csi = (2 ** 1/2) * pixel_size * scoll
    sigma_crossY = H / (1 - Rt_real) / fw * sigma_csi
    sigma_alongY = H / (1 - Rl) / fh * sigma_csi

    outputs_all = {
        'GSDw': GSDw,
        'GSDh': GSDh,
        'num_images': num_images,
        'pixel_size': pixel_size,
        'min_sigma_z': 0, # TODO cambiarla
        'W': W,
        'H': H,
        'UAS_min_speed': UAS_v_min,
        'max_distance': max_distance,
        'max_distance_proj': max_distance_proj,
        'i': interaxie,
        'b': b
    }
    outputs_along_track = {
        'number_stripes': nstrip_y,
        'images_per_stripe': nstrip_x,
        'i_real': i_real,
        'b_real': b_real,
        'sigma_crossY': sigma_crossY,
        'sigma_alongY': sigma_alongY
    }

    return outputs_all, outputs_along_track

    # GSDw, GSDh, num_images, pixel_size, ##sigma_z, W, H, UAS_v_min, max_distance, i(along), b(across),
    # max_distance_project
    # nstrip_y, nstrip_x, i_real, b_real, sigma_crossY, sigma_alongY

drone = {
    "DroneName": "Pioneer",
    "MaxAltitude": 40,
    "MaxSpeed": 30,
    "Battery": "50"
}
sensor = {
    "SensorName": "Sens1",
    "FocalLength": 10,
    "ShootInterval": 2,
    "SizeX": 10,
    "SizeY": 20,
    "ImgSizeX": 10,
    "ImgSizeY": 20
}

#normal_case(drone, sensor, 150, 350, 40, .13, .17, 1)

'''
def simulation():
    #########################simulation part ########################

    #  intitalize empty vectors
    pt_obs, im_obs, xsi_obs, eta_obs = ([] for i in range(4))
    #or
    #pt_obs = np.arange(...)

    # simulate observations
    Xo = np.transpose(Xo)
    Yo = np.transpose(Yo)
    ptName = np.transpose(ptName)
    ptName += 1
    XGrid = np.transpose(XGrid)
    YGrid = np.transpose(YGrid)
    for i in range(nstrip_x):
        for j in range(nstrip_y):
            # compute image coordinates for all the points
            xsiGrid = (XGrid - Xo[i][j]) / h * c   # [mm]
            etaGrid = (YGrid - Yo[i][j]) / h * c   # [mm]
            # filter point inside the image plane
            mask = np.abs(xsiGrid) <= fw/2
            mask2 = abs(etaGrid) <= fh/2
            mask = mask.astype(int) * mask2.astype(int)
            # store the observations
            pt_obs.append(ptName[mask[:]==1])
            im_obs.append(i*mask[mask[:]==1])
            xsi_obs.append(xsiGrid[mask[:]==1])
            eta_obs.append(etaGrid[mask[:]==1])

    im_obs = np.concatenate(im_obs)
    pt_obs = np.concatenate(pt_obs)
    xsi_obs = np.concatenate(xsi_obs)
    eta_obs = np.concatenate(eta_obs)

    n_obs = len(pt_obs)         # number of couples of observations
    n_pt  = np.amax(ptName[:])      # number of ground points
    n_im  = nstrip_y*nstrip_x   # number of images of the block

    noise1 = randn(len(xsi_obs))
    noise2 = randn(len(eta_obs))
    # simulate with noise
    xsi_obs = xsi_obs + (scoll * fw / npx) * noise1
    eta_obs = eta_obs + (scoll*fw/ npx) * noise2
    print(len(xsi_obs))
    print(len(eta_obs))

    #xsi
    ciao = []
    ciao.append(np.arange(n_obs))
    ciao.append(pt_obs)
    ciao.append(c*np.ones(n_obs))
    ciao.append(np.array([n_obs * 2, n_pt * 3]))
    ciao = np.concatenate(ciao)
    print(len(ciao))

    #A = csr_matrix()

    #eta
'''
