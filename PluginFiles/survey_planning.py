import numpy as np
import scipy
from numpy.random import randn
from scipy.sparse import csr_matrix
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import gdal
from shapely.geometry import Polygon
import os


def newpath(org_path, add):
    '''
        This function is to generate an automatic path for the new layers in the Simulation using DTM method
    '''

    path = os.path.normpath(org_path)
    file_name = path.split(os.sep)[-1]
    path = os.path.normpath(org_path)[:-(len(file_name) + 1)]
    new_file_name = file_name[:-4] + "_" + add + file_name[-4:]
    n_path = os.path.join(path, new_file_name)
    return n_path


class UnfilledError(AttributeError):
    pass


class SensorError(AttributeError):
    pass


class DroneBatteryError(ValueError):
    pass


class DroneMaxSpeedError(ValueError):
    pass


class DroneAltitudeError(ValueError):
    pass


class PredictionMethod:
    def __init__(self, drone, sensor, X, Y, h, Rl_list, Rt_list, scoll, delta, target_sz):
        # Inputs-----------------------------------------------------------------------
        # size of the surveyed area
        self.X = X
        self.Y = Y
        # flight design parameters
        self.h = h
        self.Rl_list = Rl_list
        self.Rt_list = Rt_list
        # accuracy in image collimation
        self.scoll = scoll
        # drone and camera parameters
        self.drone_name = drone["DroneName"]
        self.UAS_maxAltitude = drone['MaxAltitude']  # [m]
        self.UAS_v = drone['MaxSpeed']  # [km/h]
        self.battery = drone['Battery'] * 60  # [s]
        self.sensor_name = sensor["SensorName"]
        self.c = sensor['FocalLength']  # [mm]
        self.shooting_int = sensor['ShootInterval']  # [s]
        self.fw = sensor['SizeX']       # [mm]
        self.fh = sensor['SizeY']       # [mm]
        self.npx = sensor['ImgSizeX']   # [px]
        self.npy = sensor['ImgSizeY']   # [px]

        self.check_sensor()
        self.check_inputs()

        # compute parameters ---------------------------------------------------------
        self.W = self.fw*self.h / self.c  # footprint width  [m]
        self.H = self.fh*self.h / self.c  # footprint height [m]

        self.GSDw = self.W / self.npx  # GSD width   [m]
        self.GSDh = self.H / self.npy  # GSD height  [m]

        self.pixel_size = self.fw / self.npx  # pixel size [mm]

        # density of the simulation ground grid
        # will be a function based on points density chosen by the user
        self.delta = delta  # resolution of the ground point grid [m]

        # loop to find the optimal value of the overlapping using the target sigma z and GSD
        for Rl in Rl_list:
            for Rt in Rt_list:
                self.setup_values(Rl=Rl, Rt=Rt)
                self.setup_simulation()
                curr_sz = self.algorithm()  # curr_sz is the biggest sigma z of that algorithm
                if curr_sz < target_sz:
                    break
            if curr_sz < target_sz:
                break

        self.Rt = Rt
        self.Rl = Rl

    def check_sensor(self):
        to_check_sensor = [self.c, self.shooting_int, self.fw, self.fh, self.npx, self.npy]
        for i in to_check_sensor:
            if i == 0:
                raise SensorError

    def check_inputs(self):
        to_check_in = [self.X, self.Y, self.h, self.Rl_list[0], self.Rt_list[0]]
        for i in to_check_in:
            if i == 0:
                raise UnfilledError


    def setup_values(self, Rl, Rt):
        '''
            using the current Rl, Rt (that may vary in case of automatic generation), we compute the
            interaxie and the baseline
        '''
        self.current_Rl = Rl / 100.
        self.current_Rt = Rt / 100.

        self.b = (1 - self.current_Rl) * self.H          # baseline
        self.interaxie = (1 - self.current_Rt) * self.W  # interaxie

    def setup_simulation(self):
        '''
            setup values to run the simulation algorithm.
            Here we obtain the flight path, the real values for some parameters and the grid of ground points.
            Since these parameters will change in simulation using DTM, this function will be overwritten in the DTM
            subclass.
        '''
        self.nstrip_y = np.ceil(self.Y / self.b)          # number of strips in y
        self.nstrip_x = np.ceil(self.X / self.interaxie)  # number of strips in x

        self.num_images = self.nstrip_y * self.nstrip_x  # total expected number of images (in both directions)

        self.generate_real_parameters()

        self.check_drone_error()

        # determine the position of camera acquisition
        self.xo = np.arange(self.i_real/2, self.X, self.i_real)  # [m]
        self.yo = np.arange(self.b_real/2, self.Y, self.b_real)  # [m]
        self.Xo, self.Yo = np.meshgrid(self.xo, self.yo)         # grid of coordinates   [m]

        self.Zo = self.h  # height of acquisition [m]

        self.generate_flight_path()

        # define the grid of ground points -------------------------------------------
        self.xGrid = np.arange(self.delta/2, self.X, self.delta)  # vector of x
        self.yGrid = np.arange(self.delta/2, self.Y, self.delta)  # vector of y coordinates
        self.XGrid, self.YGrid = np.meshgrid(self.xGrid, self.yGrid)  # matrices of all coordinates (couples x and y)
        # index (name) of each point (to be used in the simulation)
        self.ptName = np.arange(len(self.XGrid[0])*len(self.YGrid)).reshape((len(self.YGrid), len(self.XGrid[0])), order='F')

    def check_drone_error(self):
        '''
            This function checks if the chosen drone is suitable for the required task
        '''

        if self.h > self.UAS_maxAltitude:
            # if the selected flight height is too big for that drone an error is raised
            raise DroneAltitudeError

        if np.floor(self.b / self.shooting_int) < float(self.UAS_v) / 3.6:  # [m/s]
            if self.max_distance_proj / self.battery < float(self.UAS_v) / 3.6:
                # UAS min speed compliant with shooting interval
                self.UAS_v_min = max(self.b / self.shooting_int, self.max_distance_proj / self.battery)
            else:
                # if the length is too big for the battery duration an error is raised
                raise DroneBatteryError
        else:
            # if the drone max speed is not enough to cover the baseline according to the shooting interval
            # then, an error is raised
            raise DroneMaxSpeedError

    def generate_real_parameters(self):
        '''
            adapt the estimated parameters to uniformly cover the area
        '''
        self.max_distance = (self.UAS_v / 3600 ) * self.battery  # max distance covered [km]
        self.max_distance_proj = (self.X + self.Y * self.nstrip_x) / 1000  # length of the flight path [km]

        self.b_real = self.Y / self.nstrip_y  # real baseline
        self.i_real = self.X / self.nstrip_x  # real interaxie
        self.Rl_real = 1 - self.b_real / self.H  # real longitudinal overlapping
        self.Rt_real = 1 - self.i_real / self.W  # real transversal overlapping

    def generate_flight_path(self):
        # compute the flight path (sort the couple x and y according to the drone path)
        self.nstrip_x = self.nstrip_x.astype(np.int64)
        self.nstrip_y = self.nstrip_y.astype(np.int64)
        self.flpath = np.zeros((self.nstrip_x * self.nstrip_y, 2))  # [Xo, Yo]

        for i in range(self.nstrip_x):
            self.flpath[i*self.nstrip_y: (i+1)*self.nstrip_y, 0] = self.xo[i]  # [Xo]
            if i % 2 == 1:  # [Yo]
                self.flpath[i*self.nstrip_y: (i+1)*self.nstrip_y, 1] = np.flipud(self.yo)
            else:
                self.flpath[i*self.nstrip_y: (i+1)*self.nstrip_y, 1] = self.yo

    def generate_overlapping_map(self):
        '''
            this function will generate the overlapping map, needed for the plotting
        '''
        # map of the number of images from which each point is seen
        # overlapping in y direction (vector) - longitudinal
        over_y = np.zeros(len(self.yGrid))
        for i in range(self.nstrip_y):
            for j, val in enumerate(self.yGrid):
                if val >= self.yo[i]-self.H/2 and val <= self.yo[i]+self.H/2:
                    over_y[j] += 1

        # overlapping in x direction (vector) - transversal
        over_x = np.zeros(len(self.xGrid))
        for i in range(self.nstrip_x):
            for j, val in enumerate(self.xGrid):
                if val >= self.xo[i]-self.W/2 and val <= self.xo[i]+self.W/2:
                    over_x[j] += 1

        # determine the overlapping maps
        self.Over_x, self.Over_y = np.meshgrid(over_x, over_y)

    def plot_overlapping(self):
        '''
            This function is used to generate and show the overlapping graphs.
            Since the overlapping is independent from the algorithm, this function will be the same for each of them.
        '''
        self.generate_overlapping_map()
        fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2)
        x_rect = np.array([0, self.X, self.X, 0, 0])
        y_rect = np.array([0, 0, self.Y, self.Y, 0])

        # Longitudinal Overlapping plot
        ax11.plot(x_rect, y_rect, 'r')
        ax11.axis('equal')
        im = ax11.imshow(self.Over_y, extent=[0, self.X, 0, self.Y])
        ax11.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax11.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig.colorbar(im, ax=ax11)
        ax11.set_title("Longitudinal Overlapping")

        # Transversal Overlapping plot
        ax12.plot(x_rect, y_rect, 'r')
        ax12.axis('equal')
        im = ax12.imshow(self.Over_x, extent=[0, self.X, 0, self.Y])
        ax12.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax12.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig.colorbar(im, ax=ax12)
        ax12.set_title("Transversal Overlapping")

        # Overlapping map
        ax21.plot(x_rect, y_rect, 'r')
        ax21.axis('equal')
        im = ax21.imshow(self.Over_x + self.Over_y, extent=[0, self.X, 0, self.Y])
        ax21.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax21.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig.colorbar(im, ax=ax21)
        ax21.set_title("Overlapping map")

        fig.delaxes(ax22)
        fig.tight_layout(pad=2.0)
        plt.show()


class NormalCaseMethod(PredictionMethod):

    def __init__(self, drone, sensor, X, Y, h, Rl, Rt, scoll, delta, target_sz):
        super().__init__(drone, sensor, X, Y, h, Rl, Rt, scoll, delta, target_sz=target_sz)
        self.method_name = "Normal Case"

    def plot_error(self):
        '''
            This function plots the error map
        '''
        x_rect = np.array([0, self.X, self.X, 0, 0])
        y_rect = np.array([0, 0, self.Y, self.Y, 0])

        plt.plot(x_rect, y_rect, 'r')
        plt.axis('equal')
        im = plt.imshow(self.sz1, extent=[0, self.X, 0, self.Y])
        plt.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        plt.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        plt.colorbar(im)
        plt.title("Estimated error [mm] - NC - sigma_z")
        plt.show()

    def algorithm(self):
        '''
            Normal case implementation.
            This function will return the maximum sigma z in the error map. This value will be used in the automatic
            generation procedure to select the best values of overlapping
        '''
        self.generate_overlapping_map()
        # error map (derived from overlapping map) -----------------------------------
        self.s2px = 2 * (self.scoll*self.fw/self.npx)**2  #  sigma2 of the parallax (propagating the sigma of collimation) [mm^2]
        self.s2zy = (self.h**2 / (self.c*1e-3 * self.i_real))**2 * self.s2px  #  sigma2 of the computed z considering longitudinal overlapping [mm^2]
        self.s2zx = (self.h**2 / (self.c*1e-3 * self.b_real))**2 * self.s2px  #  sigma2 of the computed z considering transversal overlapping  [mm^2]

        # propagate s2zx and s2zy
        self.sz1 = np.sqrt(
            ((self.s2zy * (self.Over_y - 1) + self.s2zx * (self.Over_x - 1)) / ((self.Over_y - 1) + (self.Over_x - 1))**2)
        )

        return np.max(self.sz1)

class SimulationMethod(PredictionMethod):
    def __init__(self, drone, sensor, X, Y, h, Rl, Rt, scoll, delta, target_sz):
        super(SimulationMethod, self).__init__(drone, sensor, X, Y, h, Rl, Rt, scoll, delta, target_sz=target_sz)
        self.method_name = "Simulation (without DTM)"

    def plot_error(self):
        '''
            This function plots the error map
        '''
        fig1, ((ax111, ax112), (ax121, ax122)) = plt.subplots(nrows=2, ncols=2)
        fig2, ((ax211, ax212), (ax221, ax222)) = plt.subplots(nrows=2, ncols=2)
        x_rect = np.array([0, self.X, self.X, 0, 0])
        y_rect = np.array([0, 0, self.Y, self.Y, 0])

        # LS sigma-z (theoretical)
        ax111.plot(x_rect, y_rect, 'r')
        ax111.axis('equal')
        im = ax111.imshow(self.sz2, extent=[0, self.X, 0, self.Y])
        ax111.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax111.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig1.colorbar(im, ax=ax111)
        ax111.set_title("Estimated error [mm] - LS σz (theor)", fontsize=8)

        # LS sigma-z (empirical)
        ax211.plot(x_rect, y_rect, 'r')
        ax211.axis('equal')
        im = ax211.imshow(self.sz3, extent=[0, self.X, 0, self.Y])
        ax211.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax211.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig2.colorbar(im, ax=ax211)
        ax211.set_title("Estimated error [mm] - LS σz (emp)", fontsize=8)

        # LS sigma-x (theoretical)
        ax112.plot(x_rect, y_rect, 'r')
        ax112.axis('equal')
        im = ax112.imshow(self.sx2, extent=[0, self.X, 0, self.Y])
        ax112.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax112.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig1.colorbar(im, ax=ax112)
        ax112.set_title("Estimated error [mm] - LS σx (theor)", fontsize=8)

        # LS sigma-x (empirical)
        ax212.plot(x_rect, y_rect, 'r')
        ax212.axis('equal')
        im = ax212.imshow(self.sx3, extent=[0, self.X, 0, self.Y])
        ax212.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax212.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig2.colorbar(im, ax=ax212)
        ax212.set_title("Estimated error [mm] - LS σx (emp)", fontsize=8)

        # LS sigma-y (theoretical)
        ax121.plot(x_rect, y_rect, 'r')
        ax121.axis('equal')
        im = ax121.imshow(self.sy2, extent=[0, self.X, 0, self.Y])
        ax121.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax121.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig1.colorbar(im, ax=ax121)
        ax121.set_title("Estimated error [mm] - LS σy (theor)", fontsize=8)

        # LS sigma-y (empirical)
        ax221.plot(x_rect, y_rect, 'r')
        ax221.axis('equal')
        im = ax221.imshow(self.sy3, extent=[0, self.X, 0, self.Y])
        ax221.plot(self.Xo, self.Yo, '.k', markersize=0.5)
        ax221.plot(self.flpath[:, 0], self.flpath[:, 1], 'r', linewidth=0.5)
        fig2.colorbar(im, ax=ax221)
        ax221.set_title("Estimated error [mm] - LS σy (emp)", fontsize=8)

        fig1.delaxes(ax122)
        fig2.delaxes(ax222)
        fig1.tight_layout(pad=2)
        fig2.tight_layout(pad=2)
        plt.show()

    def simulate_observation(self):
        '''
            in this function we simulate the observations
        '''
        self.pt_obs, self.im_obs, self.xsi_obs, self.eta_obs = ([] for i in range(4))

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

                # store the observations
                self.pt_obs.append(self.ptName[mask[:]==1])
                self.im_obs.append((i*self.nstrip_y+j)*mask[mask[:]==1])
                self.xsi_obs.append(xsiGrid[mask[:]==1])
                self.eta_obs.append(etaGrid[mask[:]==1])

        self.im_obs = np.concatenate(self.im_obs)
        self.pt_obs = np.concatenate(self.pt_obs)
        self.xsi_obs = np.concatenate(self.xsi_obs)
        self.eta_obs = np.concatenate(self.eta_obs)

        self.n_obs = len(self.pt_obs)            # number of couples of observations
        self.n_pt = np.amax(self.ptName[:])+1    # number of ground points

        noise1 = randn(len(self.xsi_obs))
        noise2 = randn(len(self.eta_obs))
        # simulate with noise
        self.xsi_obs = self.xsi_obs + (self.scoll * self.fw / self.npx) * noise1
        self.eta_obs = self.eta_obs + (self.scoll * self.fw / self.npx) * noise2

        self.im_obs_x = self.im_obs // self.nstrip_y
        self.im_obs_y = self.im_obs % self.nstrip_y
        self.my_Xo = np.zeros(len(self.im_obs))
        self.my_Yo = np.zeros(len(self.im_obs))
        for i in range(len(self.im_obs)):
            self.my_Xo[i] = self.Xo[self.im_obs_x[i]][self.im_obs_y[i]]
            self.my_Yo[i] = self.Yo[self.im_obs_x[i]][self.im_obs_y[i]]

        self.xsi = self.xsi_obs * self.Zo + self.c * self.my_Xo
        self.eta = self.eta_obs * self.Zo + self.c * self.my_Yo

    def calculate_A(self):
        ''' this function computes the design matrix A '''

        # csi
        A = csr_matrix((self.c * np.ones(self.n_obs), (np.arange(self.n_obs), self.pt_obs)),
                       shape=(int(self.n_obs)*2, int(self.n_pt)*3))

        A = A + csr_matrix((self.xsi_obs, (np.arange(self.n_obs), self.pt_obs + 2*self.n_pt)),
                           shape=(int(self.n_obs)*2, int(self.n_pt)*3))

        # eta
        A = A + csr_matrix((self.c * np.ones(self.n_obs), (np.arange(self.n_obs, 2*self.n_obs), self.pt_obs+self.n_pt)),
                           shape=(int(self.n_obs)*2, int(self.n_pt)*3))

        A = A + csr_matrix((self.eta_obs, (np.arange(self.n_obs, 2*self.n_obs), self.pt_obs+2*self.n_pt)),
                           shape=(int(self.n_obs)*2, int(self.n_pt)*3))
        return A

    def invert_matrix(self, N):
        ''' this function inverts a matrix; in this case we give the normal matrix N '''
        # inverse of N using LU decomposition and iteration
        # this method is less compact, but the matrix was too big to be allocated using other methods
        lu = scipy.sparse.linalg.splu(N)

        x_coo = []  # x coordinate of non-zero values
        y_coo = []  # y coordinate of non-zero values
        values = []  # values different from zero in N matrix

        for i in range(N.shape[0]):
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

        # here we create the sparse matrix from the positions of non zero values and their actual value
        invN = csr_matrix((values, (x_coo, y_coo)), shape=(N.shape[0], N.shape[1]))
        return invN

    def generate_error_map(self, A, x, yo, invN):
        '''
            this function generates the error map, that will be both needed for plotting and to choose the
            best values for the overlapping in the automatic generation method
        '''
        # LS residuals
        v_est = yo - A @ x
        # a-posteriori variance
        s02 = (v_est.transpose() @ v_est) / (A.shape[0] - A.shape[1])

        # vector of standard deviation (diagonal of the parameter covariance
        # matrix, rescaled by the a-priori collimation standard deviation)
        invN_diag = invN.diagonal()

        s2 = np.sqrt(s02 * invN_diag)  # [m] from "empirical" s02
        s3 = (self.scoll * self.fw / self.npx * self.Zo) * np.sqrt(invN_diag)  # [m] from "theoretical" s02

        # extract the error map in the three components("empirical")
        self.sx2 = np.reshape(s2[0: self.n_pt], self.XGrid.shape) * 1e3  # [mm]
        self.sy2 = np.reshape(s2[self.n_pt: 2 * self.n_pt], self.XGrid.shape) * 1e3  # [mm]
        self.sz2 = np.reshape(s2[2 * self.n_pt: 3 * self.n_pt], self.XGrid.shape) * 1e3  # [mm]

        # extract the error map in the three components("theoretical")
        self.sx3 = np.reshape(s3[0: self.n_pt], self.XGrid.shape) * 1e3  # [mm]
        self.sy3 = np.reshape(s3[self.n_pt: 2 * self.n_pt], self.XGrid.shape) * 1e3  # [mm]
        self.sz3 = np.reshape(s3[2 * self.n_pt: 3 * self.n_pt], self.XGrid.shape) * 1e3  # [mm]

    def algorithm(self):
        '''
            Simulation implementation.
            This function will return the maximum sigma z in the error map. This value will be used in the automatic
            generation procedure to select the best values of overlapping.
            Since the DTM algorithm is still a simulation method, this function will be used also by Simulation using
            DTM.
        '''
        self.simulate_observation()
        A = self.calculate_A()

        # Normal matrix N
        A_transpose = A.transpose()
        N = A_transpose @ A

        invN = self.invert_matrix(N)

        # observation vector yo
        yo = np.array([self.xsi, self.eta])
        yo = yo.reshape(-1)  # create a vector from a matrix

        # normal known term
        nt = A_transpose @ yo
        # estimated values
        x = (invN @ nt)

        self.generate_error_map(A=A, x=x, yo=yo, invN=invN)

        return np.max(self.sz2)     # empirical sigma z


class SimulationDTM(SimulationMethod):
    def __init__(self, drone, sensor, X, Y, h, Rl, Rt, scoll, dtm_in, shp_in, target_sz):
        self.dtm_in = dtm_in
        self.shp_in = shp_in
        super(SimulationDTM, self).__init__(drone, sensor, X, Y, h, Rl, Rt, scoll, delta=0, target_sz=target_sz)
        self.method_name = "Simulation (with DTM)"

    def setup_simulation(self):
        '''
            This function retrieves the values from the DTM and overwrites the simulated parameters.
            Specifically:
            1- Extract vertices from user shapefile
            2- Creating new rectangle polygon from extracted vertices
            3- Clipping the user DEM over the Rectangular Surveying area
            4- Extract Data From the Clipped DEM
        '''
        # User inputs:
        # Shapefile path of the user area
        fn = self.shp_in
        # Path of the user DEM
        rasin = self.dtm_in

        ######1- Extract vertices from user shapefile ######
        # reading User shape file
        gdf = gpd.read_file(fn)
        # call the Coordinate reference system of the shapefile
        cr = gdf.crs
        # Initiation of polygon point coordinate such that allocation is avoided in next iteration
        max_x = float("-inf")
        max_y = float("-inf")
        min_x = float("inf")
        min_y = float("inf")

        # iteration through polygon`s points
        for index, row in gdf.iterrows():
            for pt in list(row['geometry'].exterior.coords):
                xy = Point(pt)
                if xy.x < min_x:
                    min_x = xy.x
                elif xy.x > max_x:
                    max_x = xy.x

                if xy.y < min_y:
                    min_y = xy.y
                elif xy.y > max_y:
                    max_y = xy.y

        # Vertices of the new Rectangular Polygon
        p1 = [min_x, min_y]
        p2 = [max_x, min_y]
        p3 = [max_x, max_y]
        p4 = [min_x, max_y]

        ####2- Creating new rectangle polygon from extracted points####
        polygon = Polygon([p1, p2, p3, p4, p1])

        # Creating GeoDataframe containing The Rectangular Polygon
        data = {'id': [1], 'geometry': polygon}
        poly_gdf = gpd.GeoDataFrame(data, crs=str(cr))

        # create the path of the rectangular area shapefile
        shpin = newpath(fn, "rectangle")

        # Write the  Rectangle polygon into a shape file
        poly_gdf.to_file(filename=shpin, driver='ESRI Shapefile')

        # creating the path of the new Clipped DEM
        rasout = newpath(rasin, "cut")

        rect_DEM = gdal.Warp(rasout, rasin, cutlineDSName=shpin, cropToCutline=True)

        ds = gdal.Open(rasout)
        Elev = ds.GetRasterBand(1).ReadAsArray()

        ###4-Extract Data From the Clipped DEM###
        Y = Elev.shape[0]  # no of rows ("height" = Y)
        X = Elev.shape[1]  # no of coloumns ("width" = X)

        delta = 10

        # origin x_0, y_0
        x_0 = ds.GetGeoTransform()[0]
        Dem_res_x = np.abs(ds.GetGeoTransform()[1])  # pixel dimension
        y_0 = ds.GetGeoTransform()[3]
        Dem_res_y = np.abs(ds.GetGeoTransform()[5])  # pixel dimension
        self.X = Dem_res_x * X
        self.Y = Dem_res_y * Y
        self.nstrip_y = np.ceil(self.Y / self.b)  # number of strips in y
        self.nstrip_x = np.ceil(self.X / self.interaxie)  # number of strips in x

        flpath_length = self.X + self.Y * (self.nstrip_x + 1)  # flight path for error checking

        self.num_images = self.nstrip_y * self.nstrip_x  # total expected number of images (in both directions)

        if np.floor(self.b / self.shooting_int) < float(self.UAS_v) / 3.6:
            if flpath_length / self.battery < float(self.UAS_v) / 3.6:
                # UAS min speed compliant with shooting interval
                self.UAS_v_min = max(self.b / self.shooting_int, flpath_length / self.battery)
            else:
                # if the length is too big for the battery duration an error is raised
                raise DroneBatteryError
        else:
            # if the drone max speed is not enough to cover the baseline according to the shooting interval
            # an error is raised
            raise DroneMaxSpeedError

        self.max_distance = self.UAS_v * 60 * self.UAS_v_min  # max distance covered [m]
        self.max_distance_proj = self.nstrip_x * self.b * self.nstrip_y + self.b * self.interaxie  # max distance in project [m]

        # adapt the estimated parameters to uniformly cover the area
        self.b_real = (self.Y) / self.nstrip_y  # real baseline
        self.i_real = (self.X) / self.nstrip_x  # real interaxie
        self.Rl_real = 1 - self.b_real / self.H  # real longitudinal overlapping
        self.Rt_real = 1 - self.i_real / self.W  # real transversal overlapping

        # determine the position of camera acquisition
        # moving origin of the acqusition to the bottom left corner of DEM
        self.xo = np.arange(0 + self.i_real/2, self.X, self.i_real)   # [m]
        self.yo = np.arange(0,  self.Y - self.b_real/2, self.b_real)  # [m]
        self.yo = np.flipud(self.yo)
        self.Xo, self.Yo = np.meshgrid(self.xo, self.yo)                # grid of coordinates [m]

        # determine the position of camera acquisition
        # set Nan values
        Elev[Elev == -32767.] = np.nan
        # translating the Elev to our conventional Z
        self.Zo = np.nanmean(Elev) + self.h  # height of acquisition [m]

        # compute the flight path (sort the couple x and y according to the drone path
        self.nstrip_x = self.nstrip_x.astype(np.int64)
        self.nstrip_y = self.nstrip_y.astype(np.int64)
        self.flpath = np.zeros((self.nstrip_x * self.nstrip_y, 2))  # [Xo, Yo]

        for i in range(self.nstrip_x):
            self.flpath[i * self.nstrip_y: (i + 1) * self.nstrip_y, 0] = self.xo[i]  # [Xo]
            if i % 2 == 1:  # [Yo]
                self.flpath[i * self.nstrip_y: (i + 1) * self.nstrip_y, 1] = np.flipud(self.yo)
            else:
                self.flpath[i * self.nstrip_y: (i + 1) * self.nstrip_y, 1] = self.yo

        # define the grid of ground points -------------------------------------------
        self.xGrid = np.arange(0, self.X, Dem_res_x * delta)  # vector of x
        self.yGrid = np.arange(0, self.Y, Dem_res_y * delta)  # vector of y coordinates
        self.XGrid, self.YGrid = np.meshgrid(self.xGrid, self.yGrid)  # matrices of all coordinates (couples x and y)
        # index ("name") of each point (to be used in the simulation)
        self.ptName = np.arange(len(self.XGrid[0]) * len(self.YGrid)).reshape((len(self.YGrid), len(self.XGrid[0])), order='F')

        # overlapping map ------------------------------------------------------------
        # map of the number of images from which each point is seen
        # overlapping in y direction (vector) - longitudinal
        over_y = np.zeros(len(self.yGrid))
        for i in range(self.nstrip_y):
            for j, val in enumerate(self.yGrid):
                if val >= self.yo[i] - self.H / 2 and val <= self.yo[i] + self.H / 2:
                    over_y[j] += 1

        # overlapping in x direction (vector) - transversal
        over_x = np.zeros(len(self.xGrid))
        for i in range(self.nstrip_x):
            for j, val in enumerate(self.xGrid):
                if val >= self.xo[i] - self.W / 2 and val <= self.xo[i] + self.W / 2:
                    over_x[j] += 1

        # determine the overlapping maps
        self.Over_x, self.Over_y = np.meshgrid(over_x, over_y)
