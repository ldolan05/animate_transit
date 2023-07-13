import animate_transit.main as ani
import numpy as np

def end_to_end_ngts1b():

    ## initial paramteres for system NGTS-1b (Bayliss et al. 2018)
    R1 = 0.573
    R2 = 0.44
    a_R1 = 12.72
    b = 1.6
    theta = 0.
    L2_L1 = 0.
    u1_1 = 0.1
    u2_1 = 0.14

    phase_arr = np.arange(-0.5, 0.5, 0.01)

    ani.create_animation_pixs(phase_arr, 'test_ngts1b', n_pixs=1024, R1=R1, R2=R2, a_R1=a_R1, b=b, theta=theta, L2_L1=L2_L1, u1_1=u1_1, u2_1=u2_1)

end_to_end_ngts1b()