import numpy as np
import matplotlib.pyplot as plt
import imageio
from glob import glob

def model_star(n_pixs):

    star = np.zeros((n_pixs, n_pixs))
    centre_x = n_pixs/2
    centre_y = n_pixs/2
    for x in range(n_pixs):
        for y in range(n_pixs):
            r = np.sqrt((x-centre_x)**2 + (y-centre_y)**2)
            if r <= n_pixs/2:
                star[y, x]=1.

    return star

def model_planet(n_pixs, RpRs, phase, a_Rs, b, theta, grid):
    Rs = n_pixs/2
    pl_rad_pix = Rs*RpRs

    ## calculate pixel position of planet centre (x1 and y1)
    x0=a_Rs*np.sin(2.*np.pi*phase)
    y0=b*np.cos(2.*np.pi*phase)
    x1=(x0*np.cos(theta)-y0*np.sin(theta))*Rs+(n_pixs/2)
    y1=(x0*np.sin(theta)+y0*np.cos(theta))*Rs+(n_pixs/2)

    for x in range(n_pixs):
        for y in range(n_pixs):
            r = np.sqrt((x-x1)**2 + (y-y1)**2)
            if r <= pl_rad_pix:
                grid[y, x]=0.

    return grid

def gif_maker(file_list):
    with imageio.get_writer('figs/test1.gif', mode='I', duration=0.1) as writer:
        for file in file_list:
            image = imageio.imread(file)
            writer.append_data(image)

star = model_star(1000)
# model = model_planet(1000,0.3, -0.01, 8, 0, 0, star)
# plt.figure()
# plt.imshow(model)
# plt.show()

for i, phase in enumerate(np.arange(-0.05,0.05,0.005)):
    model = model_planet(n_pixs = 1000,RpRs=0.3, phase=phase, a_Rs=4, b=0.5, theta=np.pi/4, grid = star.copy())
    plt.figure()
    plt.imshow(model)
    plt.savefig("figs/model_{0:04d}.png".format(i))
    plt.close()

file_list = glob("figs/*.png")
gif_maker(sorted(file_list))

