import numpy as np
import scipy
from numpy.random import randn
from scipy.sparse import csr_matrix
import scipy.sparse.linalg
from tqdm import tqdm
import matplotlib.pyplot as plt


class PredictionMethod:
    def __init__(self, drone, sensor, X, Y, h, Rl, Rt, scoll, delta):
        # Input-----------------------------------------------------------------------
        # camera parameters
        self.X = X
        self.Y = Y
        self.h = h
        self.Rl = Rl / 100.
        self.Rt = Rt / 100.
        self.scoll = scoll

        self.shooting_int = sensor['ShootInterval']
        self.UAS_v = drone['MaxSpeed']
        self.c = sensor['FocalLength']  # [mm]
        self.fw = sensor['SizeX']       # [mm]
        self.fh = sensor['SizeY']       # [mm]
        self.npx = sensor['ImgSizeX']   # [px]
        self.npy = sensor['ImgSizeY']   # [px]

        # density of the simulation ground grid
        #will be a function based on points density chosen by the user
        self.delta = delta  # resolution of the ground point grid [m]

        # compute parameters ---------------------------------------------------------
        self.W = self.fw*h / self.c     # footprint width  [m]
        self.H = self.fh*h / self.c     # footprint height [m]
        self.GSDw = self.W / self.npx   # GSD              [m]
        self.GSDh = self.H / self.npy   # GSD (check)      [m]

        self.b = (1-Rl) * self.H   # baseline
        self.interaxie = (1-Rt) * self.W   # interaxie

        self.nstrip_y = np.ceil(Y/self.b)   # number of strips in y
        self.nstrip_x = np.ceil(X/self.interaxie)   # number of strips in x

        self.num_images = self.nstrip_y * self.nstrip_x  # total expected number of images (in both directions)

        self.pixel_size = self.fw / self.npx    # pixel size [mm]

        self.UAS_v_min = self.b / self.shooting_int # UAS min speed compliant with shooting interval

        self.max_distance = self.UAS_v * 60 * self.UAS_v_min    # max distance covered [m]
        self.max_distance_proj = self.nstrip_x * self.b * self.nstrip_y + self.b * self.interaxie   # max distance in project [m]

        # adapt the estimated parameters to uniformly cover the area
        self.b_real = Y / self.nstrip_y    # real baseline
        self.i_real = X / self.nstrip_x    # real interaxie
        self.Rl_real = 1 - self.b_real/self.H   # real longitudinal overlapping
        self.Rt_real = 1 - self.i_real/self.W   # real transversal overlapping

        # determine the position of camera acquisition
        self.xo = np.arange(self.i_real/2, X, self.i_real)         # [m]
        self.yo = np.arange(self.b_real/2, Y, self.b_real)         # [m]
        self.Xo, self.Yo = np.meshgrid(self.xo, self.yo)           # grid of coordinates   [m]
        self.Zo = h                                                # height of acquisition [m]

        # compute the flight path (sort the couple x and y according to the drone path
        self.nstrip_x = self.nstrip_x.astype(np.int64)
        self.nstrip_y = self.nstrip_y.astype(np.int64)
        self.flpath = np.zeros((self.nstrip_x*self.nstrip_y, 2))  # [Xo, Yo]
        for i in range(self.nstrip_x):
            self.flpath[i*self.nstrip_y:(i+1)*self.nstrip_y, 0] = self.xo[i]    # [Xo]
            if i % 2 == 1:                                  # [Yo]
                self.flpath[i*self.nstrip_y:(i+1)*self.nstrip_y, 1] = np.flipud(self.yo)
            else:
                self.flpath[i*self.nstrip_y:(i+1)*self.nstrip_y, 1] = self.yo

        # define the grid of ground points -------------------------------------------
        self.xGrid = np.arange(self.delta/2, X, self.delta)                   # vector of x
        self.yGrid = np.arange(self.delta/2, Y, self.delta)                   # vector of y coordinates
        self.XGrid, self.YGrid = np.meshgrid(self.xGrid, self.yGrid)               # matrices of all coordinates (couples x and y)
        self.ptName = np.arange(len(self.XGrid[0])*len(self.YGrid)).reshape((len(self.YGrid), len(self.XGrid[0])), order='F')     #index ("name") of each point (to be used in the simulation)

        # overlapping map ------------------------------------------------------------
        # map of the number of images from which each point is seen
        # overlapping in y direction (vector) - longitudinal
        over_y = np.zeros(len(self.yGrid))
        for i in range(self.nstrip_y):
            for j, val in enumerate(self.yGrid):
                if val >= self.yo[i]-self.H/2 and val <= self.yo[i]+self.H/2:
                    over_y[j] += 1

        #overlapping in x direction (vector) - transversal
        over_x = np.zeros(len(self.xGrid))
        for i in range(self.nstrip_x):
            for j, val in enumerate(self.xGrid):
                if val >= self.xo[i]-self.W/2 and val <= self.xo[i]+self.W/2:
                    over_x[j] += 1

        # determine the overlapping maps
        self.Over_x, self.Over_y = np.meshgrid(over_x, over_y)

    def plot_overlapping(self):
        fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2)
        x_rect = np.array([0, self.X, self.X, 0, 0])
        y_rect = np.array([0, 0, self.Y, self.Y, 0])

        # Longitudinal Overlapping plot
        ax11.plot(x_rect, y_rect, 'r')
        ax11.axis('equal')
        im = ax11.imshow(self.Over_y, extent=[0, self.X, 0, self.Y])
        ax11.plot(self.Xo, self.Yo, '.k')
        ax11.plot(self.flpath[:, 0], self.flpath[:, 1], 'r')
        fig.colorbar(im, ax=ax11)
        ax11.set_title("Longitudinal Overlapping")

        # Transversal Overlapping plot
        ax12.plot(x_rect, y_rect, 'r')
        ax12.axis('equal')
        im = ax12.imshow(self.Over_x, extent=[0, self.X, 0, self.Y])
        ax12.plot(self.Xo, self.Yo, '.k')
        ax12.plot(self.flpath[:, 0], self.flpath[:, 1], 'r')
        fig.colorbar(im, ax=ax12)
        ax12.set_title("Transversal Overlapping")

        # Overlapping map
        ax21.plot(x_rect, y_rect, 'r')
        ax21.axis('equal')
        im = ax21.imshow(self.Over_x + self.Over_y, extent=[0, self.X, 0, self.Y])
        ax21.plot(self.Xo, self.Yo, '.k')
        ax21.plot(self.flpath[:, 0], self.flpath[:, 1], 'r')
        fig.colorbar(im, ax=ax21)
        ax21.set_title("Overlapping map")

        fig.delaxes(ax22)
        plt.show()


class NormalCaseMethod(PredictionMethod):

    def __init__(self, drone, sensor, X, Y, h, Rl, Rt, scoll, delta):
        super().__init__(drone, sensor, X, Y, h, Rl, Rt, scoll, delta)

    def plot_error(self):
        pass

    def algorithm(self):

        # error map (derived from overlapping map) -----------------------------------
        self.s2px = 2 * (self.scoll*self.fw/self.npx)**2                      #  sigma2 of the parallax (propagating the sigma of collimation) [mm^2]
        self.s2zy = (self.h**2 / (self.c*1e-3 * self.i_real))**2 * self.s2px  #  sigma2 of the computed z considering longitudinal overlapping [mm^2]
        self.s2zx = (self.h**2 / (self.c*1e-3 * self.b_real))**2 * self.s2px  #  sigma2 of the computed z considering transversal overlapping  [mm^2]

        # propagate s2zx and s2zy
        self.sz1 = np.sqrt(
            ((self.s2zy * (self.Over_y - 1) + self.s2zx * (self.Over_x - 1)) / ((self.Over_y - 1) + (self.Over_x - 1))**2)
        )


class SimulationMethod(PredictionMethod):

    def __init__(self, drone, sensor, X, Y, h, Rl, Rt, scoll, delta):
        super(SimulationMethod, self).__init__(drone, sensor, X, Y, h, Rl, Rt, scoll, delta)

    def plot_error(self):
        pass

    def algorithm(self):
        #########################simulation part ########################

        #  intitalize empty vectors
        pt_obs, im_obs, xsi_obs, eta_obs = ([] for i in range(4))

        # simulate observations
        self.Xo = np.transpose(self.Xo)
        self.Yo = np.transpose(self.Yo)

        for i in range(self.nstrip_x):
            for j in range(self.nstrip_y):
                # compute image coordinates for all the points
                xsiGrid = (self.XGrid - self.Xo[i][j]) / self.h * self.c   # [mm]
                etaGrid = (self.YGrid - self.Yo[i][j]) / self.h * self.c   # [mm]
                # filter point inside the image plane
                mask1 = np.abs(xsiGrid) <= np.float(self.fw)/2
                mask2 = np.abs(etaGrid) <= np.float(self.fh)/2
                mask = mask1.astype(int) * mask2.astype(int)
                cont = 0

                # store the observations
                pt_obs.append(self.ptName[mask[:]==1])
                im_obs.append((i*self.nstrip_y+j)*mask[mask[:]==1])
                xsi_obs.append(xsiGrid[mask[:]==1])
                eta_obs.append(etaGrid[mask[:]==1])

        im_obs = np.concatenate(im_obs)
        pt_obs = np.concatenate(pt_obs)
        xsi_obs = np.concatenate(xsi_obs)
        eta_obs = np.concatenate(eta_obs)

        n_obs = len(pt_obs)                    # number of couples of observations
        n_pt  = np.amax(self.ptName[:]) + 1    # number of ground points
        n_im  = self.nstrip_y*self.nstrip_x    # number of images of the block

        noise1 = randn(len(xsi_obs))
        noise2 = randn(len(eta_obs))
        # simulate with noise
        xsi_obs = xsi_obs + (self.scoll * self.fw / self.npx) * noise1
        eta_obs = eta_obs + (self.scoll * self.fw / self.npx) * noise2

        #csi

        A = csr_matrix(
            (self.c * np.ones(n_obs), (np.arange(n_obs), pt_obs)),
            shape=(
                int(n_obs)*2,
                int(n_pt)*3
            )
        )
        A = A + csr_matrix(
            (xsi_obs, (np.arange(n_obs), pt_obs + 2*n_pt)),
            shape=(
                int(n_obs)*2,
                int(n_pt)*3
            )
        )

        #eta

        A = A + csr_matrix(
            (self.c * np.ones(n_obs), (np.arange(n_obs, 2*n_obs), pt_obs+n_pt)),
            shape=(
                int(n_obs)*2,
                int(n_pt)*3
            )
        )
        A = A + csr_matrix(
            (eta_obs, (np.arange(n_obs, 2*n_obs), pt_obs+2*n_pt)),
            shape=(
                int(n_obs)*2,
                int(n_pt)*3
            )
        )

        print("matrix A created")


        im_obs_x = im_obs // self.nstrip_y
        im_obs_y = im_obs % self.nstrip_y
        my_Xo = np.zeros(len(im_obs))
        my_Yo = np.zeros(len(im_obs))
        for i in range(len(im_obs)):
            my_Xo[i] = self.Xo[im_obs_x[i]][im_obs_y[i]]
            my_Yo[i] = self.Yo[im_obs_x[i]][im_obs_y[i]]

        xsi = xsi_obs * self.Zo + self.c * my_Xo
        eta = eta_obs * self.Zo + self.c * my_Yo
        yo = np.array([xsi, eta])

        A_transpose = A.transpose()
        N = A_transpose @ A
        # using LU decomposition and iteration
        # this method is less compact, but we use tqdm's bar to check our progress
        lu = scipy.sparse.linalg.splu(N)

        bar = tqdm(range(N.shape[0]))
        x_coo = []   # x coordinate of non-zero values
        y_coo = []   # y coordinate of non-zero values
        values = []  # values different from zero in N matrix

        for i in bar:
            b = np.zeros((N.shape[0],))
            b[i] = 1
            current_row = lu.solve(b)

            tmp_x_coo = np.nonzero(current_row)[0]
            tmp_y_coo = np.zeros_like(tmp_x_coo)
            tmp_values = current_row[tmp_x_coo]

            x_coo.append(tmp_x_coo)
            y_coo.append(tmp_y_coo + i)
            values.append(tmp_values)

        x_coo = np.concatenate(x_coo)
        y_coo = np.concatenate(y_coo)
        values = np.concatenate(values)

        # here we create the sparse matrix from the positions of non zero values and their
        # actual value
        invN = csr_matrix((values, (x_coo, y_coo)), shape=(N.shape[0], N.shape[1]))

        yo = yo.reshape(-1)  # this command create a vector from a matrix
        nt = A_transpose @ yo  # normal known term
        x = (invN @ nt)

        # LS residuals
        v_est = yo - A @ x
        s02 = (v_est.transpose() @ v_est) / (A.shape[0] - A.shape[1])  # a-posteriori variance

        # vector of standard deviation (diagolanal of the parameter covariance
        # matrix, rescaled by the a-priori collimation std)
        invN_diag = invN.diagonal()

        s2 = np.sqrt(s02 * invN_diag)                                      # [m] from "empirical" s02
        s3 = (self.scoll*self.fw/self.npx * self.Zo) * np.sqrt(invN_diag)  # [m] from "theoretical" s02

        # extract the error map in the three components("empirical")
        sx2 = np.reshape(s2[0: n_pt], self.XGrid.shape) * 1e3             # [mm]
        sy2 = np.reshape(s2[n_pt: 2 * n_pt], self.XGrid.shape) * 1e3      # [mm]
        sz2 = np.reshape(s2[2 * n_pt: 3 * n_pt], self.XGrid.shape) * 1e3  # [mm]

        # extract the error map in the three components("theoretical")
        sx3 = np.reshape(s3[0: n_pt], self.XGrid.shape) * 1e3             # [mm]
        sy3 = np.reshape(s3[n_pt: 2 * n_pt], self.XGrid.shape) * 1e3      # [mm]
        sz3 = np.reshape(s3[2 * n_pt: 3 * n_pt], self.XGrid.shape) * 1e3  # [mm]

        #invN = scipy.sparse.linalg.inv(N)
        #L = csr_matrix(np.linalg.cholesky(N.todense()))
        #L = scipy.linalg.cholesky(N.todense(),lower=True)
        #invN = np.linalg.pinv(N.toarray())
        #invL = np.linalg.inv(L)
        #invN = np.transpose(invL) @ invL
        #lu_obj = scipy.sparse.linalg.splu(N)
        #lu_obj.solve(np.eye(157500))


#method = SimulationMethod(drone, sensor, 150, 350, 40, 0.7, 0.7, 1, 4)
#method.algorithm()

