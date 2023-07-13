import animate_transit.main as ani
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pytest

def test_transitdepth():

    ## initial paramteres for system NGTS-1b (Bayliss et al. 2018)
    R1 = 0.573
    R2 = 0.44
    a_R1 = 12.72
    b = 1.6
    theta = 0.
    L2_L1 = 0.
    u1_1 = 0.1
    u2_1 = 0.14

    phase_arr = np.arange(-0.5, 0.5, 0.05)

    test_system = ani.system(n_pixs=1024, R1=R1, R2=R2, a_R1=a_R1, b=b, theta=theta, L2_L1=L2_L1, u1_1=u1_1, u2_1=u2_1)

    # Model the fixed object to create the base grid
    master_grid = test_system.model_object1()

    # Creates a multiprocessing pool
    pool = Pool(8)
    # Creates models for all phases using multiprocess pool for speed
    model_list = list(tqdm(pool.imap(test_system.model_object2, phase_arr), total=len(phase_arr)))

    flux_arr = ani.get_fluxes(model_list, master_grid)

    assert min(flux_arr)==pytest.approx(0.975, abs=1e-3)
    
test_transitdepth()
