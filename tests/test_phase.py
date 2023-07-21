import animate_transit.main as ani
import numpy as np

def test_phase():
    ani.create_animation_pixs(np.arange(0., 1., 0.01), 'test')

test_phase()