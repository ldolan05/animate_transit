import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import os

class system(object):
    """
    A class to store information about the system and model the primary and secondary object

    Args:
        n_pixs (int): Size of the image canvas size in pixels. The canvas will be a square.
        R1 (float): Radius of stationary primary object in same units as R2
        R2 (float): Radius of secondary object in same units as R1
        a_R1 (float): Ratio of semi-major axis and radius of primary object (R1)
        b (float): Impact parameter of transit
        theta (float): Spin-orbit angle
        L2_L1 (float): Ratio of luminosity of secondary object to luminosity primary object
        u1_1 (float, optional): Quadratic limb darkening coefficient for object 1 (default: None) TODO add ref
        u2_1 (float, optional): Quadratic limb darkening coefficient for object 1 (default: None)
        u1_2 (float, optional): Quadratic limb darkening coefficient for object 2 (default: None)
        u2_2 (float, optional): Quadratic limb darkening coefficient for object 2 (default: None)
    """
    def __init__(self, n_pixs, R1, R2, a_R1, b, theta, L2_L1, u1_1=None, u2_1=None, u1_2=None, u2_2=None):
        self.n_pixs = n_pixs    # number of pixels in grid (npixs by npixs)

        pix_per_unit = n_pixs/(3*R1+2*(a_R1*R1)+3*R2)  #conversion factor between pixels and unit

        self.R1 = pix_per_unit*R1  # radius of object 1 (stationary object) in same units as R2
        self.R2 = pix_per_unit*R2  # radius of object 2 in same units as R1

        self.a_R1 = a_R1        # orbital distance over radius of object 1
        self.b = b              # impact parameter
        self.theta = theta      # spin-orbit angle
        self.grid1 = np.zeros((self.n_pixs, self.n_pixs))   # grid for object 1 (serves as background grid)
        self.L2_L1 = L2_L1      # luminosity ratio 

        self.u1_1= u1_1         # limb darkening coefficents for object 1
        self.u2_1 = u2_1

        self.u1_2= u1_2         # limb darkening coefficents for object 2
        self.u2_2 = u2_2

    def model_object1(self):
        """
        Models the flux of the stationary object in the centre of the pixel canvas

        Returns:
            grid1 (np.ndarray of float): Base pixel grid with the stationary object at the centre
        """
        centre_x = self.n_pixs/2
        centre_y = self.n_pixs/2
        for x in range(self.n_pixs):
            for y in range(self.n_pixs):
                # calculate pixel distance from centre of object 1
                r = np.sqrt((x-centre_x)**2 + (y-centre_y)**2)
                if r <= self.R1:
                    if self.u1_1 != None and self.u2_1 != None:
                        # calculate flux using quadratic limb darkening
                        onemu=1.-np.sqrt(1.-r**2/(self.R1)**2)
                        self.grid1[y, x]=1.-self.u1_1*onemu - self.u2_1*onemu*onemu
                    else:
                        self.grid1[y, x]=1.

        return self.grid1

    def model_object2(self, phase):
        """
        Models the flux of the orbiting object at the given phase

        Args:
            phase (float): Orbital phase of object 2 to model
            
        Returns:
            grid (np.ndarray of float): 2D pixel map showing visualisation of the system
        """
        grid = self.grid1.copy()

        ## calculate pixel coordinates of centre of object 2 (x1 and y1) from phase (using orbital properties)
        x0=self.a_R1*np.sin(2.*np.pi*phase)
        y0=self.b*np.cos(2.*np.pi*phase)
        x1=(x0*np.cos(self.theta)-y0*np.sin(self.theta))*self.R1+(self.n_pixs/2)
        y1=(x0*np.sin(self.theta)+y0*np.cos(self.theta))*self.R1+(self.n_pixs/2)

        for x in range(self.n_pixs):
            for y in range(self.n_pixs):
                # calculate distance of pixel from centre of object 2
                r = np.sqrt((x-x1)**2 + (y-y1)**2)
                if r <= self.R2:
                    if abs(phase)<0.25 or self.grid1[y, x]==0:
                        grid[y, x]=self.L2_L1
                        if self.u1_2!=None and self.u2_2!=None:
                            onemu=1.-np.sqrt(1.-r**2/(self.R2)**2)
                            grid[y, x]=grid[y, x]*(1-self.u1_2*onemu - self.u2_2*onemu*onemu)

        return grid

def gif_maker(file_list, output_file):
    """Makes gif (pronounced jif)

    Creates the gif by combining all the image files in the given list.

    Args:
        file_list (:obj:`list` of :obj:`str`): List of filepaths to individual images to be compiled into gif
        output_file (str): Filename for output gif

    Returns:
        None
    """
    with imageio.get_writer('{}.gif'.format(output_file), mode='I', duration=0.1) as writer:
        for file in file_list:
            image = imageio.imread(file)
            writer.append_data(image)

def normalised_flux(grid, grid1):
    """
    Calculates the normalised flux for the given model by comparing to the background pixel image of just the stationary object

    Args:
        grid (np.ndarray of float): Output model grid from model_object2 for given phase
        grid1 (np.ndarray of float): 2D pixel map of object 1 (returned from model_object1)
        
    Returns:
        flux (float): The normalised flux for the given model grid
    """
    ref_flux = np.sum(grid1)
    flux = np.sum(grid)/ref_flux
    return flux

def get_fluxes(model_list, grid1):
    """Flux Calculation
     
    Calculates full array of fluxes for the light curve.

    Args:
       model_list (:obj:`list` of :obj:`str`): list of model outputs (generated using model_object1 and model_object2)
       grid1 (array): 2D pixel map of object 1 (returned from model_object1)
       
    Returns:
        flux_arr (array): Flux values for the given model list.
    """
    flux_arr=[]
    for model in model_list:
        flux_arr.append(normalised_flux(model, grid1=grid1))
    return np.array(flux_arr)

def create_animation_pixs(phase_arr, output_gif, n_pixs=1000, R1=0.4, R2=0.1, a_R1=4., b=0.1, theta=0.1, L2_L1=0., u1_1 = 0.5, u2_1 = 0.2):
    """Make animation 

    Creates 2D grid of a 2-body system for a range of phases. Also combines all figures into a gif.
    We include example parameters for ease of running.
    
    Args:
        phase_arr (array): Phase values for animation to cover.
        output_gif (str): File path for final gif to be saved under.
        n_pixs (int): Size of the image canvas size in pixels. The canvas will be a square.
        R1 (float): Radius of stationary primary object in same units as R2
        R2 (float): Radius of secondary object in same units as R1
        a_R1 (float): Ratio of semi-major axis and radius of primary object (R1)
        b (float): Impact parameter of transit
        theta (float): Spin-orbit angle
        L2_L1 (float): Ratio of luminosity of secondary object to luminosity primary object
        u1_1 (float, optional): Quadratic limb darkening coefficient for object 1 (default: None)
        u2_1 (float, optional): Quadratic limb darkening coefficient for object 1 (default: None)
        u1_2 (float, optional): Quadratic limb darkening coefficient for object 2 (default: None)
        u2_2 (float, optional): Quadratic limb darkening coefficient for object 2 (default: None)
    
    Returns:
        None
    """
    dirname = "figs"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # TODO implement multiprocessing for plotting to speed code up

    test_system = system(n_pixs, R1, R2, a_R1, b, theta, L2_L1, u1_1, u2_1)

    # Model the fixed object to create the base grid
    master_grid = test_system.model_object1()

    # Creates a multiprocessing pool
    pool = Pool(8)
    # Creates models for all phases using multiprocess pool for speed
    model_list = list(tqdm(pool.imap(test_system.model_object2, phase_arr), total=len(phase_arr)))
    flux_arr = get_fluxes(model_list, master_grid)

    for i, (phase, model) in enumerate(tqdm(zip(phase_arr, model_list), total=len(phase_arr))):
        #Create temporary arrays of the phase and flux for the current phase value for plotting
        phase_tmp = phase_arr[phase_arr<=phase]
        flux_tmp = flux_arr[phase_arr<=phase]

        fig, (ax1, ax2) = plt.subplots(2, figsize=(8,15))
        
        # Plots visualisation of the system
        ax1.imshow(model)
        ax1.axis("off")

        # Plots light curve
        ax2.scatter(phase_tmp, flux_tmp, marker='.', c='c')
        ax2.set_xlim(min(phase_arr), max(phase_arr))

        ax2.set_xlabel("Phase")
        ax2.set_ylabel("Normalized Flux")

        #Sets the aspect ratios of the subplots to be the same
        asp2 = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
        asp2 /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
        ax2.set_aspect(asp2)

        fig.savefig("figs/model_{0:04d}.png".format(i), dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    file_list = glob("figs/*.png")
    gif_maker(sorted(file_list), "{}/{}.gif".format(dirname, output_gif))
