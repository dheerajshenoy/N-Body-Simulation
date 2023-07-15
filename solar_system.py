import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from matplotlib.patches import Circle


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Vector2({}, {})".format(self.x, self.y)
    
    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

class Body:
    def __init__(self, mass, position : np.ndarray, velocity : np.ndarray):
        self.mass = mass
        self.position = position
        self.velocity = velocity

class Simulator:
    def __init__(self, planets, t, tEnd, dt, softening, G):
        self.planets = planets
        self.t = t
        self.tEnd = tEnd
        self.dt = dt
        self.softening = softening
        self.G = G
        self.N = len(self.planets)

        self.fig = plt.figure()
        self.ax = plt.subplot()

        self.simulate()

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
        x = np.array([p[0] for p in pos])
        y = np.array([p[1] for p in pos])

        dx = x.T - x
        dy = y.T - y

        # matrix that stores 1/r^3 for all particle pairwise particle separations 
        inv_r3 = (dx**2 + dy**2 + self.softening**2)
        inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

        ax = self.G * (dx * inv_r3) @ self.mass
        ay = self.G * (dy * inv_r3) @ self.mass

        # pack together the acceleration components
        a = np.hstack((ax,ay))

        return a


    def simulate(self):

        self.mass = []
        pos = []
        self.vel = []

        for planet in self.planets:
            self.mass.append(planet.mass)
            pos.append(planet.position)
            self.vel.append(planet.velocity)

        self.acc = self.getAcc(pos)

        self.Nt = int(np.ceil(self.tEnd/self.dt))

        self.pos_save = np.zeros((self.N,2,self.Nt+1))
        
        for i in range(len(pos)):
            self.pos_save[:, 0, 0] = pos[i][0]
            self.pos_save[:, 1, 0] = pos[i][1]

        # Simulation Main Loop
        for i in range(self.Nt):
            
            self.vel =+ self.acc * self.dt/2.0
            # drift
            pos += self.vel * self.dt

            # update accelerations
            self.acc = self.getAcc(pos)
            
            # (1/2) kick
            self.vel += self.acc * self.dt/2.0

            # update time
            self.t += self.dt
            
            print(pos)
            self.pos_save[:,:,i+1] = pos

        self.animationFunction()
        anim1 = FuncAnimation(self.fig, self.animate, frames = self.Nt, interval = 4)
        plt.legend()
        plt.show()

    def animationFunction(self):

        self.patches = np.empty(shape=(self.N), dtype=Circle)
        
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])

        for i in range(self.N):
            p = plt.Circle(xy=(self.pos_save[i, 0, 0], self.pos_save[i, 1, 0]), radius = 0.05, color='r')
            self.ax.add_patch(p)
            self.patches[i] = p

    def animate(self, i):
        for n in range(len(self.patches)):
            print(self.pos_save[n, 0, i], self.pos_save[n, 1, i])
            self.patches[n].center = (self.pos_save[n, 0, i], self.pos_save[n, 1, i])
        return self.patches

if __name__ == "__main__":
    t         = 0      # current time of the simulation
    tEnd      = 10     # current time of the simulation
    dt        = 0.01   # timestep
    softening = 0.1    # softening length
    G         = 1.0    # Newton's Gravitational Constant
    
    p1 = Body(2e30, np.array([0, 0]), np.array([5, 5])) # sun
    p2 = Body(6e24, np.array([5, 5]), np.array([10, 10])) # earth
    p3 = Body(6.4e23, np.array([2, 2]), np.array([10, 10]))
    planets = [p1, p2, p3]

    s = Simulator(planets, t, tEnd, dt, softening, G)
