import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from matplotlib.patches import Circle
import threading
from numba import jit, int32

"""
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""
N = 100    # Number of particles

def getAcc( pos, mass, G, softening ):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations 
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax,ay,az))

    return a

def getEnergy( pos, vel, mass, G ):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum( mass * vel**2 ))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations 
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))

    return KE, PE;


# prep figure
fig = plt.figure()
grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
ax1 = plt.subplot()
ax1.set_aspect("equal")
# ax2 = plt.subplot(grid[2,0])
tEnd = 10

def main():
    """ N-body simulation """

    t         = 0      # current time of the simulation
    dt        = 0.01   # timestep
    softening = 0.1    # softening length
    G         = 1.0    # Newton's Gravitational Constant

    # Simulation parameters
    # Generate Initial Conditions
    np.random.seed(time.localtime().tm_sec)

    mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
    pos  = np.random.randn(N,3)   # randomly selected positions and velocities
    vel  = np.random.randn(N,3)

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel,0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc( pos, mass, G, softening )

    # calculate initial energy of system
    #KE, PE  = getEnergy( pos, vel, mass, G )

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N,3,Nt+1))
    pos_save[:,:,0] = pos
    # KE_save = np.zeros(Nt+1)
    # KE_save[0] = KE
    # PE_save = np.zeros(Nt+1)
    # PE_save[0] = PE
    t_all = np.arange(Nt+1)*dt

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc( pos, mass, G, softening )

        # (1/2) kick
        vel += acc * dt/2.0

        # update time
        t += dt

        # get energy of system
        # KE, PE  = getEnergy( pos, vel, mass, G )

        # save energies, positions for plotting trail
        pos_save[:,:,i+1] = pos
        # KE_save[i+1] = KE
        # PE_save[i+1] = PE

    animationFunction(pos_save)
    anim1 = FuncAnimation(fig, animate, frames = Nt, fargs=(pos_save, ), interval = 4)
    plt.legend()
    plt.show()

    return 0

patches = np.empty(shape=(N), dtype=Circle)

def animationFunction(pos):
    N = pos.shape[0]

    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])

    for i in range(N):
        p = plt.Circle(xy=(pos[i, 0, 0], pos[i, 1, 0]), radius = 0.05, color='r')
        ax1.add_patch(p)
        patches[i] = p

def animate(i, pos):
    for n in range(len(patches)):
        patches[n].center = (pos[n, 0, i], pos[n, 1, i])
    # ax2.plot(i, PE[i], '.b', label = "PE")
    # ax2.plot(i, KE[i], '.r', label = "KE")
    # ax2.plot(i, PE[i] + KE[i], '.k', label = "Total Energy")
    return patches


if __name__== "__main__":
    main()
