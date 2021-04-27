import numpy as np
from ..utils import visualize
from scipy.stats import multivariate_normal


class liquid:
    def __init__(self, N_x=150, N_y=150):

        self.L_x = 1E+6
        self.L_y = 1E+6
        self.g = 9.81
        self.H = 100

        self.N_x = N_x
        self.N_y = N_y
        self.dx = self.L_x / (self.N_x - 1)
        self.dy = self.L_y / (self.N_y - 1)
        self.dt = 0.5 * min(self.dx, self.dy) / np.sqrt(self.g * self.H)
        self.x = np.linspace(-self.L_x / 2, self.L_x / 2, self.N_x)
        self.y = np.linspace(-self.L_y / 2, self.L_y / 2, self.N_y)
        X, Y = np.meshgrid(self.x, self.y)
        self.X = np.transpose(X)
        self.Y = np.transpose(Y)

        self.u_n = np.zeros((self.N_x, self.N_y))
        self.v_n = np.zeros((self.N_x, self.N_y))
        self.eta_n = np.zeros((self.N_x, self.N_y))

        self.u_n[:, :] = 0.0
        self.v_n[:, :] = 0.0
        self.u_n[-1, :] = 0.0
        self.v_n[:, -1] = 0.0

        self.eta_list = list()
        self.u_list = list()
        self.v_list = list()
        self.anim_interval = 20
        self.sample_interval = 1000


    def clear(self):
        self.eta_n = np.zeros(self.N_x, self.N_y)
        return

    def clear_region(self, hstart, wstart, hend, wend):
        # TODO: clear rectangular region
        pass

    def shape(self):
        return (self.N_x, self.N_y)

    def display_2d(self):
        visualize.pmesh_plot(self.X, self.Y, self.eta_n, "Final state of surface elevation $\eta$")
        return

    def display_field(self):
        quiv_anim = visualize.velocity_simulation(self.X, self.Y, self.u_list, self.v_list, self.anim_interval*self.dt, "velocity_simulation")
        return quiv_anim

    def animation_2d(self):
        eta_anim = visualize.surface_simulation_2d(self.X, self.Y, self.eta_list, self.anim_interval * self.dt, "surface_simulation_2d")
        return eta_anim

    def animation_3d(self):
        eta_surf_anim = visualize.surface_simulation_3d(self.X, self.Y, self.eta_list, self.anim_interval*self.dt, "surface_simulation_3d")
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

        u_np1 = np.zeros((self.N_x, self.N_y))
        v_np1 = np.zeros((self.N_x, self.N_y))
        eta_np1 = np.zeros((self.N_x, self.N_y))

        u_np1[:-1, :] = self.u_n[:-1, :] - self.g * self.dt / self.dx * (self.eta_n[1:, :] - self.eta_n[:-1, :])
        v_np1[:, :-1] = self.v_n[:, :-1] - self.g * self.dt / self.dy * (self.eta_n[:, 1:] - self.eta_n[:, :-1])

        v_np1[:, -1] = 0.0
        u_np1[-1, :] = 0.0

        h_e = np.zeros((self.N_x, self.N_y))
        h_w = np.zeros((self.N_x, self.N_y))
        h_n = np.zeros((self.N_x, self.N_y))
        h_s = np.zeros((self.N_x, self.N_y))
        uhwe = np.zeros((self.N_x, self.N_y))
        vhns = np.zeros((self.N_x, self.N_y))

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


        eta_np1[:, :] = self.eta_n[:, :] - self.dt * (uhwe[:, :] / self.dx + vhns[:, :] / self.dy)  # Without source/sink

        eta_np1 = eta_np1 * (0.4/np.max(eta_np1))

        self.u_n = np.copy(u_np1)
        self.v_n = np.copy(v_np1)
        self.eta_n = np.copy(eta_np1)

        self.u_list.append(self.u_n)
        self.v_list.append(self.v_n)
        self.eta_list.append(self.eta_n)

#         return self.u_n, self.v_n, self.eta_n
        return


    def update_n_step(self, n=1):
        for i in range(n):
            self.__update_one_step()
        return
