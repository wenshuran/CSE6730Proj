import numpy as np
from ..utils import visualize
import matplotlib.pyplot as plt
import ffmpeg
from scipy.stats import multivariate_normal


class liquid:
    def __init__(self, N_x=150, N_y=150):

        # --------------- Physical prameters ---------------
        self.L_x = 1E+6  # Length of domain in x-direction
        self.L_y = 1E+6  # Length of domain in y-direction
        self.g = 9.81  # Acceleration of gravity [m/s^2]
        self.H = 100  # Depth of fluid [m]
        self.f_0 = 4E-4  # Fixed part ofcoriolis parameter [1/s]
        self.beta = 2E-11  # gradient of coriolis parameter [1/ms]
        self.rho_0 = 1024.0  # Density of fluid [kg/m^3)]
        self.tau_0 = 0.1  # Amplitude of wind stress [kg/ms^2]
        self.use_coriolis = False  # True if you want coriolis force
        self.use_friction = False  # True if you want bottom friction
        self.use_wind = False  # True if you want wind stress
        self.use_beta = False  # True if you want variation in coriolis
        self.use_source = False  # True if you want mass source into the domain
        self.use_sink = False  # True if you want mass sink out of the domain
        self.param_string = "\n================================================================"
        self.param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(self.use_coriolis, self.use_beta)
        self.param_string += "\nuse_friction = {}\nuse_wind = {}".format(self.use_friction, self.use_wind)
        self.param_string += "\nuse_source = {}\nuse_sink = {}".format(self.use_source, self.use_sink)
        self.param_string += "\ng = {:g}\nH = {:g}".format(self.g, self.H)

        # --------------- Computational prameters ---------------
        self.N_x = N_x  # Number of grid points in x-direction
        self.N_y = N_y  # Number of grid points in y-direction
        self.dx = self.L_x / (self.N_x - 1)  # Grid spacing in x-direction
        self.dy = self.L_y / (self.N_y - 1)  # Grid spacing in y-direction
        self.dt = 0.5 * min(self.dx, self.dy) / np.sqrt(self.g * self.H)  # Time step (defined from the CFL condition)
        self.x = np.linspace(-self.L_x / 2, self.L_x / 2, self.N_x)  # Array with x-points
        self.y = np.linspace(-self.L_y / 2, self.L_y / 2, self.N_y)  # Array with y-points
        X, Y = np.meshgrid(self.x, self.y)  # Meshgrid for plotting
        self.X = np.transpose(X)  # To get plots right
        self.Y = np.transpose(Y)  # To get plots right

        # Define friction array if friction is enabled.
        if (self.use_friction is True):
            kappa_0 = 1 / (5 * 24 * 3600)
            self.kappa = np.ones((self.N_x, self.N_y)) * kappa_0

        # Define wind stress arrays if wind is enabled.
        if (self.use_wind is True):
            self.tau_x = -self.tau_0 * np.cos(np.pi * self.y / self.L_y) * 0
            self.tau_y = np.zeros((1, len(self.x)))

        # Define coriolis array if coriolis is enabled.
        if (self.use_coriolis is True):
            if (self.use_beta is True):
                self.f = self.f_0 + self.beta * self.y  # Varying coriolis parameter
                self.L_R = np.sqrt(self.g * self.H) / self.f_0  # Rossby deformation radius
                self.c_R = self.beta * self.g * self.H / self.f_0 ** 2  # Long Rossby wave speed
            else:
                self.f = self.f_0 * np.ones(len(self.y))  # Constant coriolis parameter

            self.alpha = self.dt * self.f  # Parameter needed for coriolis scheme
            self.beta_c = self.alpha ** 2 / 4  # Parameter needed for coriolis scheme

        # Define source array if source is enabled.
        if (self.use_source):
            sigma = np.zeros((self.N_x, self.N_y))
            self.sigma = 0.0001 * np.exp(-((self.X - self.L_x / 2) ** 2 / (2 * (1E+5) ** 2) + (self.Y - self.L_y / 2) ** 2 / (2 * (1E+5) ** 2)))

        # Define source array if source is enabled.
        if (self.use_sink is True):
            self.w = np.ones((self.N_x, self.N_y)) * self.sigma.sum() / (self.N_x * self.N_y)

        # print out starting parameters
        print("Starting parameters: \n" + self.param_string)

        # ============================= Parameter stuff done ===============================

        # ==================================================================================
        # ==================== Allocating arrays and initial conditions ====================
        # ==================================================================================
        self.u_n = np.zeros((self.N_x, self.N_y))  # To hold u at current time step
        self.v_n = np.zeros((self.N_x, self.N_y))  # To hold v at current time step
        self.eta_n = np.zeros((self.N_x, self.N_y))  # To hold eta at current time step


        # Initial conditions for u and v.
        self.u_n[:, :] = 0.0  # Initial condition for u
        self.v_n[:, :] = 0.0  # Initial condition for u
        self.u_n[-1, :] = 0.0  # Ensuring initial u satisfy BC
        self.v_n[:, -1] = 0.0  # Ensuring initial v satisfy BC


        # Sampling variables.
        self.eta_list = list()
        self.u_list = list()
        self.v_list = list()  # Lists to contain eta and u,v for animation
#         self.hm_sample = list()
#         self.ts_sample = list()
#         self.t_sample = list()  # Lists for Hovmuller and time series
#         self.hm_sample.append(self.eta_n[:, int(self.N_y / 2)])  # Sample initial eta in middle of domain
#         self.ts_sample.append(self.eta_n[int(self.N_x / 2), int(self.N_y / 2)])  # Sample initial eta at center of domain
#         self.t_sample.append(0.0)  # Add initial time to t-samples
        self.anim_interval = 20  # How often to sample for time series
        self.sample_interval = 1000  # How often to sample for time series
        # =============== Done with setting up arrays and initial conditions ===============


    def clear(self):
        self.eta_n = np.zeros(self.N_x, self.N_y)
        return self.value

    def clear_region(self, hstart, wstart, hend, wend):
        # TODO: clear rectangular region
        pass

    def shape(self):
        return (self.N_x, self.N_y)

    def display_2d(self):
        visualize.pmesh_plot(self.X, self.Y, self.eta_n, "Final state of surface elevation $\eta$")
        return

    def display_field(self):
        quiv_anim = visualize.velocity_animation(self.X, self.Y, self.u_list, self.v_list, self.anim_interval*self.dt, "velocity")
        return quiv_anim

    def animation_2d(self):
        eta_anim = visualize.eta_animation(self.X, self.Y, self.eta_list, self.anim_interval * self.dt, "eta")
        return eta_anim

    def animation_3d(self):
        eta_surf_anim = visualize.eta_animation3D(self.X, self.Y, self.eta_list, self.anim_interval*self.dt, "eta_surface")
        return eta_surf_anim

    def take_one_drop(self, drop):
        x = np.linspace(0,self.N_x, self.N_x)
        y = np.linspace(0,self.N_y, self.N_y)
        X, Y = np.meshgrid(x,y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y

        F = multivariate_normal(mean=[drop.x, drop.y],cov=[[drop.width, 0], [0, drop.width]])
        Z = F.pdf(pos) * drop.amplitude

        self.eta_n = np.add(self.eta_n, Z)
        return

    def take_drops(self, drops):
        # TODO: if drops is only one drop object instead of array of drops
        # automatically handle the case
        """
        The surface gets some drops and update the canvas.

        Parameters
        ----------
        drops : arr
            input arr of drops. Each drop has its own location and amplitute

        Returns
        -------
        N/A

        See also
        -------
        /water/drop.py
        """
        for drop in drops:
            try:
                self.take_one_drop(drop)
            except:
                raise Exception("can not set drop at {},{} with value {}".format(drop.x, drop.y, drop.amplitude))

    def __update_one_step(self):

        u_np1 = np.zeros((self.N_x, self.N_y))  # To hold u at next time step
        v_np1 = np.zeros((self.N_x, self.N_y))  # To hold v at enxt time step
        eta_np1 = np.zeros((self.N_x, self.N_y))  # To hold eta at next time step

        # ------------ Computing values for u and v at next time step --------------
        u_np1[:-1, :] = self.u_n[:-1, :] - self.g * self.dt / self.dx * (self.eta_n[1:, :] - self.eta_n[:-1, :])
        v_np1[:, :-1] = self.v_n[:, :-1] - self.g * self.dt / self.dy * (self.eta_n[:, 1:] - self.eta_n[:, :-1])

        # Add friction if enabled.
        if (self.use_friction is True):
            u_np1[:-1, :] -= self.dt * self.kappa[:-1, :] * self.u_n[:-1, :]
            v_np1[:-1, :] -= self.dt * self.kappa[:-1, :] * self.v_n[:-1, :]

        # Add wind stress if enabled.
        if (self.use_wind is True):
            u_np1[:-1, :] += self.dt * self.tau_x[:] / (self.rho_0 * self.H)
            v_np1[:-1, :] += self.dt * self.tau_y[:] / (self.rho_0 * self.H)

        # Use a corrector method to add coriolis if it's enabled.
        if (self.use_coriolis is True):
            u_np1[:, :] = (u_np1[:, :] - self.beta_c * self.u_n[:, :] + self.alpha * self.v_n[:, :]) / (1 + self.beta_c)
            v_np1[:, :] = (v_np1[:, :] - self.beta_c * self.v_n[:, :] - self.alpha * self.u_n[:, :]) / (1 + self.beta_c)

        v_np1[:, -1] = 0.0  # Northern boundary condition
        u_np1[-1, :] = 0.0  # Eastern boundary condition
        # -------------------------- Done with u and v -----------------------------

        # Temporary variables (each time step) for upwind scheme in eta equation
        h_e = np.zeros((self.N_x, self.N_y))
        h_w = np.zeros((self.N_x, self.N_y))
        h_n = np.zeros((self.N_x, self.N_y))
        h_s = np.zeros((self.N_x, self.N_y))
        uhwe = np.zeros((self.N_x, self.N_y))
        vhns = np.zeros((self.N_x, self.N_y))

        # --- Computing arrays needed for the upwind scheme in the eta equation.----
        h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, self.eta_n[:-1, :] + self.H, self.eta_n[1:, :] + self.H)
        h_e[-1, :] = self.eta_n[-1, :] + self.H

        h_w[0, :] = self.eta_n[0, :] + self.H
        h_w[1:, :] = np.where(u_np1[:-1, :] > 0, self.eta_n[:-1, :] + self.H, self.eta_n[1:, :] + self.H)

        h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, self.eta_n[:, :-1] + self.H, self.eta_n[:, 1:] + self.H)
        h_n[:, -1] = self.eta_n[:, -1] + self.H

        h_s[:, 0] = self.eta_n[:, 0] + self.H
        h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, self.eta_n[:, :-1] + self.H, self.eta_n[:, 1:] + self.H)

        uhwe[0, :] = u_np1[0, :] * h_e[0, :]
        uhwe[1:, :] = u_np1[1:, :] * h_e[1:, :] - u_np1[:-1, :] * h_w[1:, :]

        vhns[:, 0] = v_np1[:, 0] * h_n[:, 0]
        vhns[:, 1:] = v_np1[:, 1:] * h_n[:, 1:] - v_np1[:, :-1] * h_s[:, 1:]
        # ------------------------- Upwind computations done -------------------------

        # ----------------- Computing eta values at next time step -------------------
        eta_np1[:, :] = self.eta_n[:, :] - self.dt * (uhwe[:, :] / self.dx + vhns[:, :] / self.dy)  # Without source/sink

        # Add source term if enabled.
        if (self.use_source is True):
            eta_np1[:, :] += self.dt * self.sigma

        # Add sink term if enabled.
        if (self.use_sink is True):
            eta_np1[:, :] -= self.dt * self.w
        # ----------------------------- Done with eta --------------------------------

        eta_np1 = eta_np1 * (0.4/np.max(eta_np1))

        self.u_n = np.copy(u_np1)  # Update u for next iteration
        self.v_n = np.copy(v_np1)  # Update v for next iteration
        self.eta_n = np.copy(eta_np1)  # Update eta for next iteration

        # Samples for Hovmuller diagram and spectrum every sample_interval time step.
        # if (time_step % sample_interval == 0):
#         hm_sample.append(eta_n[:, int(N_y / 2)])  # Sample middle of domain for Hovmuller
#         ts_sample.append(eta_n[int(N_x / 2), int(N_y / 2)])  # Sample center point for spectrum
#         t_sample.append(time_step * dt)  # Keep track of sample times.
        #
        # # Store eta and (u, v) every anin_interval time step for animations.
        # if (time_step % anim_interval == 0):
#         print("Time: \t{:.2f} hours".format(time_step * dt / 3600))
#         print("Step: \t{} / {}".format(time_step, max_time_step))
#         print("Mass: \t{}\n".format(np.sum(eta_n)))
        self.u_list.append(self.u_n)
        self.v_list.append(self.v_n)
        self.eta_list.append(self.eta_n)

#         return self.u_n, self.v_n, self.eta_n
        return


    def update_n_step(self, n=1):
        for i in range(n):
            self.__update_one_step()
        return
