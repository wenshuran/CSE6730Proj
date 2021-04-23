import numpy as np
class liquid:
    def __init__(self, height, width, dampening=0.95):
        # self.height = height
        # self.width = width
        # self.dampening = 0.95
        # self.value = np.zeros((self.height, self.width), dtype=float)
        # TODO: store self.value in self.history everytime it gets update
        # scipy.sparse may be needed to save memory
        # self.history = np.array([])

        # --------------- Physical prameters ---------------
        self.L_x = 1E+6  # Length of domain in x-direction
        self.L_y = 1E+6  # Length of domain in y-direction
        self.g = 9.81  # Acceleration of gravity [m/s^2]
        self.H = 100  # Depth of fluid [m]
        self.f_0 = 1E-4  # Fixed part ofcoriolis parameter [1/s]
        self.beta = 2E-11  # gradient of coriolis parameter [1/ms]
        self.rho_0 = 1024.0  # Density of fluid [kg/m^3)]
        self.tau_0 = 0.1  # Amplitude of wind stress [kg/ms^2]
        self.use_coriolis = True  # True if you want coriolis force
        self.use_friction = False  # True if you want bottom friction
        self.use_wind = False  # True if you want wind stress
        self.use_beta = True  # True if you want variation in coriolis
        self.use_source = False  # True if you want mass source into the domain
        self.use_sink = False  # True if you want mass sink out of the domain
        self.param_string = "\n================================================================"
        self.param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(self.use_coriolis, self.use_beta)
        self.param_string += "\nuse_friction = {}\nuse_wind = {}".format(self.use_friction, self.use_wind)
        self.param_string += "\nuse_source = {}\nuse_sink = {}".format(self.use_source, self.use_sink)
        self.param_string += "\ng = {:g}\nH = {:g}".format(self.g, self.H)

        # --------------- Computational prameters ---------------
        self.N_x = 150  # Number of grid points in x-direction
        self.N_y = 150  # Number of grid points in y-direction
        self.dx = self.L_x / (self.N_x - 1)  # Grid spacing in x-direction
        self.dy = self.L_y / (self.N_y - 1)  # Grid spacing in y-direction
        self.dt = 0.1 * min(self.dx, self.dy) / np.sqrt(self.g * self.H)  # Time step (defined from the CFL condition)
        self.x = np.linspace(-self.L_x / 2, self.L_x / 2, self.N_x)  # Array with x-points
        self.y = np.linspace(-self.L_y / 2, self.L_y / 2, self.N_y)  # Array with y-points
        X, Y = np.meshgrid(self.x, self.y)  # Meshgrid for plotting
        self.X = np.transpose(X)  # To get plots right
        self.Y = np.transpose(Y)  # To get plots right
        self.param_string += "\ndx = {:.2f} km\ndy = {:.2f} km\ndt = {:.2f} s".format(self.dx, self.dy, self.dt)

        # Define friction array if friction is enabled.
        if (self.use_friction is True):
            kappa_0 = 1 / (5 * 24 * 3600)
            self.kappa = np.ones((self.N_x, self.N_y)) * kappa_0
            self.param_string += "\nkappa = {:g}\nkappa/beta = {:g} km".format(kappa_0, kappa_0 / (self.beta * 1000))

        # Define wind stress arrays if wind is enabled.
        if (self.use_wind is True):
            self.tau_x = -self.tau_0 * np.cos(np.pi * self.y / self.L_y) * 0
            self.tau_y = np.zeros((1, len(self.x)))
            self.param_string += "\ntau_0 = {:g}\nrho_0 = {:g} km".format(self.tau_0, self.rho_0)

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

            self.param_string += "\nf_0 = {:g}".format(self.f_0)
            self.param_string += "\nMax alpha = {:g}\n".format(self.alpha.max())
            self.param_string += "\nRossby radius: {:.1f} km".format(self.L_R / 1000)
            self.param_string += "\nRossby number: {:g}".format(np.sqrt(self.g * self.H) / (self.f_0 * self.L_x))
            self.param_string += "\nLong Rossby wave speed: {:.3f} m/s".format(self.c_R)
            self.param_string += "\nLong Rossby transit time: {:.2f} days".format(self.L_x / (self.c_R * 24 * 3600))
            self.param_string += "\n================================================================\n"

        # Define source array if source is enabled.
        if (self.use_source):
            sigma = np.zeros((self.N_x, self.N_y))
            self.sigma = 0.0001 * np.exp(-((self.X - self.L_x / 2) ** 2 / (2 * (1E+5) ** 2) + (self.Y - self.L_y / 2) ** 2 / (2 * (1E+5) ** 2)))

        # Define source array if source is enabled.
        if (self.use_sink is True):
            w = np.ones((self.N_x, self.N_y)) * sigma.sum() / (self.N_x * self.N_y)

        # Write all parameters out to file.
        with open("param_output.txt", "w") as output_file:
            output_file.write(self.param_string)

        print(self.param_string)  # Also print parameters to screen
        # ============================= Parameter stuff done ===============================

        # ==================================================================================
        # ==================== Allocating arrays and initial conditions ====================
        # ==================================================================================
        self.u_n = np.zeros((self.N_x, self.N_y))  # To hold u at current time step
        self.v_n = np.zeros((self.N_x, self.N_y))  # To hold v at current time step
        self.eta_n = np.zeros((self.N_x, self.N_y))  # To hold eta at current time step

        # Temporary variables (each time step) for upwind scheme in eta equation
        self.h_e = np.zeros((self.N_x, self.N_y))
        self.h_w = np.zeros((self.N_x, self.N_y))
        self.h_n = np.zeros((self.N_x, self.N_y))
        self.h_s = np.zeros((self.N_x, self.N_y))
        self.uhwe = np.zeros((self.N_x, self.N_y))
        self.vhns = np.zeros((self.N_x, self.N_y))

        # Initial conditions for u and v.
        self.u_n[:, :] = 0.0  # Initial condition for u
        self.v_n[:, :] = 0.0  # Initial condition for u
        self.u_n[-1, :] = 0.0  # Ensuring initial u satisfy BC
        self.v_n[:, -1] = 0.0  # Ensuring initial v satisfy BC

        # Initial condition for eta.
        # eta_n[:, :] = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
        # eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
        self.eta_n = np.exp(-((self.X - self.L_x / 2.7) ** 2 / (2 * (0.05E+6) ** 2) + (self.Y - self.L_y / 4) ** 2 / (2 * (0.05E+6) ** 2)))
        # eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
        # eta_n[int(6*N_x/8):int(7*N_x/8),int(6*N_y/8):int(7*N_y/8)] = 1.0
        # eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
        # eta_n[:, :] = 0.0

        # viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))

        # Sampling variables.
        self.eta_list = list();
        self.u_list = list();
        self.v_list = list()  # Lists to contain eta and u,v for animation
        self.hm_sample = list();
        self.ts_sample = list();
        self.t_sample = list()  # Lists for Hovmuller and time series
        self.hm_sample.append(self.eta_n[:, int(self.N_y / 2)])  # Sample initial eta in middle of domain
        self.ts_sample.append(self.eta_n[int(self.N_x / 2), int(self.N_y / 2)])  # Sample initial eta at center of domain
        self.t_sample.append(0.0)  # Add initial time to t-samples
        self.anim_interval = 20  # How often to sample for time series
        self.sample_interval = 1000  # How often to sample for time series
        # =============== Done with setting up arrays and initial conditions ===============
        
        
    def clear(self):
        self.value = np.zeros(self.height, self.width)
        return self.value
    
    def clear_region(self, hstart, wstart, hend, wend):
        # TODO: clear rectangular region
        pass
        
    def shape(self):
        return (self.height, self.width)
    
    def inspect(self):
        return self.value
    
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
                self.value[drop.x][drop.y] = drop.amplitude
            except:
                raise Exception("can not set drop at {},{} with value {}".format(drop.x, drop.y, drop.amplitude))
                
    def update_one_step(self):
        # TODO: 4 edges with special care
        # TODO: vectorization to speed things up
        # curr = np.copy(self.inspect())
        # nxt = np.copy(self.inspect())
        # for i in range(1, self.height - 1):
        #     for j in range(1, self.width - 1):
        #         nxt[i][j] = (curr[i-1][j]+curr[i+1][j]+curr[i][j-1]+curr[i][j+1])/2 - curr[i][j]
        #         nxt[i][j] = nxt[i][j] * self.dampening
        # self.value = nxt
        # return self.inspect()

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

        # --- Computing arrays needed for the upwind scheme in the eta equation.----
        self.h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, self.eta_n[:-1, :] + self.H, self.eta_n[1:, :] + self.H)
        self.h_e[-1, :] = self.eta_n[-1, :] + self.H

        self.h_w[0, :] = self.eta_n[0, :] + self.H
        self.h_w[1:, :] = np.where(u_np1[:-1, :] > 0, self.eta_n[:-1, :] + self.H, self.eta_n[1:, :] + self.H)

        self.h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, self.eta_n[:, :-1] + self.H, self.eta_n[:, 1:] + self.H)
        self.h_n[:, -1] = self.eta_n[:, -1] + self.H

        self.h_s[:, 0] = self.eta_n[:, 0] + self.H
        self.h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, self.eta_n[:, :-1] + self.H, self.eta_n[:, 1:] + self.H)

        self.uhwe[0, :] = u_np1[0, :] * self.h_e[0, :]
        self.uhwe[1:, :] = u_np1[1:, :] * self.h_e[1:, :] - u_np1[:-1, :] * self.h_w[1:, :]

        self.vhns[:, 0] = v_np1[:, 0] * self.h_n[:, 0]
        self.vhns[:, 1:] = v_np1[:, 1:] * self.h_n[:, 1:] - v_np1[:, :-1] * self.h_s[:, 1:]
        # ------------------------- Upwind computations done -------------------------

        # ----------------- Computing eta values at next time step -------------------
        eta_np1[:, :] = self.eta_n[:, :] - self.dt * (self.uhwe[:, :] / self.dx + self.vhns[:, :] / self.dy)  # Without source/sink

        # Add source term if enabled.
        if (self.use_source is True):
            eta_np1[:, :] += self.dt * self.sigma

        # Add sink term if enabled.
        if (self.use_sink is True):
            eta_np1[:, :] -= self.dt * self.w
        # ----------------------------- Done with eta --------------------------------

        u_n = np.copy(u_np1)  # Update u for next iteration
        v_n = np.copy(v_np1)  # Update v for next iteration
        eta_n = np.copy(eta_np1)  # Update eta for next iteration

        # Samples for Hovmuller diagram and spectrum every sample_interval time step.
        # if (time_step % sample_interval == 0):
        #     hm_sample.append(eta_n[:, int(N_y / 2)])  # Sample middle of domain for Hovmuller
        #     ts_sample.append(eta_n[int(N_x / 2), int(N_y / 2)])  # Sample center point for spectrum
        #     t_sample.append(time_step * dt)  # Keep track of sample times.
        #
        # # Store eta and (u, v) every anin_interval time step for animations.
        # if (time_step % anim_interval == 0):
        #     print("Time: \t{:.2f} hours".format(time_step * dt / 3600))
        #     print("Step: \t{} / {}".format(time_step, max_time_step))
        #     print("Mass: \t{}\n".format(np.sum(eta_n)))
        #     u_list.append(u_n)
        #     v_list.append(v_n)
        #     eta_list.append(eta_n)
        
        
    def update_n_step(self, n=1):
        for i in range(n):
            self.update_one_step()
        return self.inspect()
    
    def record(self):
        # TODO: store self.value in self.history everytime it gets update
        # scipy.sparse may be needed to save memory
        pass
    
    def history(n=None):
        # TODO: retrieve records at step n
        # if n is None, retrieve all records
        pass
        # if n is not None:
        #     return self.history(n)
        # return self.history()
    
        
        