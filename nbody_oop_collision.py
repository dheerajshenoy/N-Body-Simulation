import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from matplotlib.patches import Circle

class NBodySim:
    def __init__(self, N, t, tEnd, dt, softening, G):
        self.N = N
        self.t = t
        self.tEnd = tEnd
        self.dt = dt
        self.softening = softening
        self.G = G

        self.fig = plt.figure()
        self.ax1 = plt.subplot()

        self.simulate()
        #grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
        # ax2 = plt.subplot(grid[2,0])

    def getAcc(self, pos):
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
        inv_r3 = (dx**2 + dy**2 + dz**2 + self.softening**2)
        inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

        ax = self.G * (dx * inv_r3) @ self.mass
        ay = self.G * (dy * inv_r3) @ self.mass
        az = self.G * (dz * inv_r3) @ self.mass

        # pack together the acceleration components
        a = np.hstack((ax,ay,az))

        return a

    def simulate(self):

        np.random.seed(time.localtime().tm_sec)

        self.mass = 20.0*np.ones((self.N,1))/self.N  # total mass of particles is 20
        pos  = np.random.randn(self.N,3)   # randomly selected positions and velocities
        self.vel  = np.random.randn(self.N,3)

        # Convert to Center-of-Mass frame
        #self.vel -= np.mean(self.mass * self.vel,0) / np.mean(self.mass)

    # calculate initial gravitational accelerations
        self.acc = self.getAcc(pos)

        # calculate initial energy of system
        #KE, PE  = getEnergy( pos, vel, mass, G )

    # number of timesteps
        self.Nt = int(np.ceil(self.tEnd/self.dt))

        self.pos_save = np.zeros((self.N,3,self.Nt+1))
        self.pos_save[:,:,0] = pos
        # KE_save = np.zeros(Nt+1)
        # KE_save[0] = KE
        # PE_save = np.zeros(Nt+1)
        # PE_save[0] = PE
        #t_all = np.arange(self.Nt+1)*self.dt

        # Simulation Main Loop
        for i in range(self.Nt):
            # (1/2) kick
            self.vel += self.acc * self.dt/2.0

            # drift
            pos += self.vel * self.dt

            # update accelerations
            self.acc = self.getAcc(pos)

            # (1/2) kick
            self.vel += self.acc * self.dt/2.0

            # update time
            self.t += self.dt

            # get energy of system
            # KE, PE  = getEnergy( pos, vel, mass, G )

            # save energies, positions for plotting trail
            self.pos_save[:,:,i+1] = pos
            # KE_save[i+1] = KE
            # PE_save[i+1] = PE

        self.animationFunction()
        anim1 = FuncAnimation(self.fig, self.animate, frames = self.Nt, interval = 4)
        plt.legend()
        plt.show()

    def animationFunction(self):

        self.patches = np.empty(shape=(self.N), dtype=Circle)
        
        self.ax1.set_xlim([-5, 5])
        self.ax1.set_ylim([-5, 5])

        for i in range(self.N):
            p = plt.Circle(xy=(self.pos_save[i, 0, 0], self.pos_save[i, 1, 0]), radius = 0.05, color='r')
            self.ax1.add_patch(p)
            self.patches[i] = p


    def animate(self, i):
        for n in range(len(self.patches)):
            self.patches[n].center = (self.pos_save[n, 0, i], self.pos_save[n, 1, i])
        # ax2.plot(i, PE[i], '.b', label = "PE")
        # ax2.plot(i, KE[i], '.r', label = "KE")
        # ax2.plot(i, PE[i] + KE[i], '.k', label = "Total Energy")
        return self.patches

if __name__ == "__main__":
    N         = 100
    t         = 0      # current time of the simulation
    tEnd      = 10     # current time of the simulation
    dt        = 0.01   # timestep
    softening = 0.1    # softening length
    G         = 1.0    # Newton's Gravitational Constant
    s = NBodySim(N, t, tEnd, dt, softening, G)
