import numpy as np

"""
    This file is to define constants
"""

# IAD path
IAD_PATH = "D:/OneDrive - University of Cambridge/IAD-win-3-11-4/iad.exe"

# Parameter constants
G = 0.7  # anisotropy factor g is fixed
THETA = 8  # incident angle theta is fixed
THICKNESS_SAMPLE = np.array([2.36, 2.4]) +1e-7#Todo np.arange(2.24, 2.38, 0.02) +1e-7  # [mm] sample thickness is fixed, add 1e-7 for numerical stability 0.999999->1

# Todo: Parameters only for 20230403, no sphere correction considered
STANDARD_REFLECTANCE = None
REFRACTIVE_INDEX_SAMPLE = 1.41
REFRACTIVE_INDEX_GLASS = None
THICKNESS_GLASS = None
BEAM_DIAMETER = 8.0
WALL_REFLECTANCE = None
DETECTOR_REFLECTANCE = None
NUMBER_OF_SPHERES = 2  # CHECK!!!
SPHERE_DIAMETER = None
SAMPLE_PORT_DIAMETER = 10.0
ENTRANCE_PORT_DIAMETER = None
DETECTOR_PORT_DIAMETER = None
SPHERE_DIAMETER_2 = None
SAMPLE_PORT_DIAMETER_2 = SAMPLE_PORT_DIAMETER
ENTRANCE_PORT_DIAMETER_2 = None
DETECTOR_PORT_DIAMETER_2 = None

MU = None

