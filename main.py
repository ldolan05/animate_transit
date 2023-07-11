import numpy as np
import matplotlib.pyplot as plt
import imageio
from glob import glob

class system(object):

    def __init__(self, n_pixs, R1, R2, a_R1, b, theta, L2_L1):
        self.n_pixs = n_pixs    # number of pixels in grid (npixs by npixs)

        pix_per_unit = n_pixs/(3*R1+2*(a_R1*R1)+3*R2)  #conversion factor between pixels and unit

        self.R1 = pix_per_unit*R1  # radius of object 1 (stationary object) in same units as R2
        self.R2 = pix_per_unit*R2  # radius of object 2 in same units as R1

        self.a_R1 = a_R1        # orbital distance over radius of object 1
        self.b = b              # impact parameter
        self.theta = theta      # spin-orbit angle
        self.grid1 = np.zeros((self.n_pixs, self.n_pixs))
        self.L2_L1 = L2_L1

    def model_object1(self):
        centre_x = self.n_pixs/2
        centre_y = self.n_pixs/2
        for x in range(self.n_pixs):
            for y in range(self.n_pixs):
                r = np.sqrt((x-centre_x)**2 + (y-centre_y)**2)
                if r <= self.R1:
                    self.grid1[y, x]=1.

    def model_object2(self, phase):
        grid = self.grid1.copy()
        ## calculate pixel position of planet centre (x1 and y1)
        x0=self.a_R1*np.sin(2.*np.pi*phase)
        y0=self.b*np.cos(2.*np.pi*phase)
        x1=(x0*np.cos(self.theta)-y0*np.sin(self.theta))*self.R1+(self.n_pixs/2)
        y1=(x0*np.sin(self.theta)+y0*np.cos(self.theta))*self.R1+(self.n_pixs/2)

        for x in range(self.n_pixs):
            for y in range(self.n_pixs):
                r = np.sqrt((x-x1)**2 + (y-y1)**2)
                if r <= self.R2:
                    if abs(phase)<0.25:
                        grid[y, x]=self.L2_L1
                    elif self.grid1[y, x]!=1:
                        grid[y, x]=self.L2_L1
        
        return grid

def gif_maker(file_list):
    with imageio.get_writer('figs/test1.gif', mode='I', duration=0.1) as writer:
        for file in file_list:
            image = imageio.imread(file)
            writer.append_data(image)


test_system = system(n_pixs=1000, R1=0.32, R2=0.11, a_R1=12.7, b=0.8, theta=0, L2_L1=0.1)

test_system.model_object1()

# model = test_system.model_object2(phase = 0.)

# plt.figure()
# plt.imshow(model)
# plt.show()

for i, phase in enumerate(np.arange(-0.5,0.5,0.01)):
    model = test_system.model_object2(phase=phase)
    plt.figure()
    plt.imshow(model)
    plt.savefig("figs/model_{0:04d}.png".format(i))
    plt.close()

file_list = glob("figs/*.png")
gif_maker(sorted(file_list))

