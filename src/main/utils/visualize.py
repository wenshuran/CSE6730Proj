import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def surface_simulation_2d(X, Y, eta_list, frame_interval, filename):
    fig, ax = plt.subplots(1, 1)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [m]", fontname = "serif", fontsize = 12)
    pmesh = plt.pcolormesh(X, Y, eta_list[0], vmin = -0.7*np.abs(eta_list[int(len(eta_list)/2)]).max(),
        vmax = np.abs(eta_list[int(len(eta_list)/2)]).max(), cmap = plt.cm.RdBu_r, shading='auto')
    plt.colorbar(pmesh, orientation = "vertical")

    def update_eta(num):
        ax.set_title("Surface elevation $\eta$ after t = {:.2f} seconds".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        pmesh.set_array(eta_list[num][:-1, :-1].flatten())
        return pmesh,

    anim = animation.FuncAnimation(fig, update_eta,
        frames = len(eta_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim

def velocity_simulation(X, Y, u_list, v_list, frame_interval, filename):
    fig, ax = plt.subplots(figsize = (12, 12), facecolor = "white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 seconds", fontname = "serif", fontsize = 19)
    plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
    plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
    q_int = 3
    Q = ax.quiver(X[::q_int, ::q_int]/1000.0, Y[::q_int, ::q_int]/1000.0, u_list[0][::q_int,::q_int], v_list[0][::q_int,::q_int],
        scale=0.2, scale_units='inches')

    def update_quiver(num):
        u = u_list[num]
        v = v_list[num]
        ax.set_title("Velocity field $\mathbf{{u}}(x,y,t)$ after t = {} seconds".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19)
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = animation.FuncAnimation(fig, update_quiver,
        frames = len(u_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    fig.tight_layout()
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim

def surface_simulation_3d(X, Y, eta_list, frame_interval, filename):
    fig = plt.figure(figsize = (12, 12), facecolor = "white")
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, eta_list[0], cmap = plt.cm.RdBu_r)

    def update_surf(num):
        ax.clear()
        surf = ax.plot_surface(X/1000, Y/1000, eta_list[num]*10, cmap = plt.cm.RdBu_r)
        ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:2f}$ seconds".format(
            num*frame_interval), fontname = "serif", fontsize = 19, y=1.04)
        ax.set_xlabel("x [km]", fontname = "serif", fontsize = 14)
        ax.set_ylabel("y [km]", fontname = "serif", fontsize = 14)
        ax.set_zlabel("$\eta$ [m]", fontname = "serif", fontsize = 16)
        ax.set_xlim(X.min()/1000, X.max()/1000)
        ax.set_ylim(Y.min()/1000, Y.max()/1000)
        ax.set_zlim(-3, 7)
        plt.tight_layout()
        return surf,

    anim = animation.FuncAnimation(fig, update_surf,
                                   frames = len(eta_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
                                         codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim


def pmesh_plot(X, Y, eta, plot_title):
    plt.figure(figsize = (9, 8))
    plt.pcolormesh(X, Y, eta, cmap = plt.cm.RdBu_r)
    plt.colorbar(orientation = "vertical")
    plt.title(plot_title, fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [s]", fontname = "serif", fontsize = 12)