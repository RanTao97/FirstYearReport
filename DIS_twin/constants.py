import numpy as np

"""
    This file is to define constants
"""

# IAD path
IAD_PATH = "D:/Users/taora/OneDrive - University of Cambridge/IAD-win-3-11-4/iad.exe"

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
# # default parameters previously
# STANDARD_REFLECTANCE = 0.985
# REFRACTIVE_INDEX_SAMPLE = 1.41
# REFRACTIVE_INDEX_GLASS = 1.5
# THICKNESS_GLASS = 0.0
# BEAM_DIAMETER = 8.0
#
# WALL_REFLECTANCE = 0.985
# DETECTOR_REFLECTANCE = 0
# NUMBER_OF_SPHERES = 2  # CHECK!!!
# SPHERE_DIAMETER = 50.0
# SAMPLE_PORT_DIAMETER = 10
# ENTRANCE_PORT_DIAMETER = 4.9
# DETECTOR_PORT_DIAMETER = 0.2
# SPHERE_DIAMETER_2 = 50.0
# SAMPLE_PORT_DIAMETER_2 = SAMPLE_PORT_DIAMETER
# ENTRANCE_PORT_DIAMETER_2 = 0.0
# DETECTOR_PORT_DIAMETER_2 = 0.2

"""
# for MCX
MCX_PATH = "D:/mcx-bin/bin/Release/mcx-exe.exe"
MCX_JSON_PATH = "D:/OneDrive - University of Cambridge/DIS_twin/mcx-python.json"
MCX_MCH_PATH = "D:/OneDrive - University of Cambridge/DIS_twin/mcx-python.mch"
"""
# for MCX WMC
MCX_PATH = "D:/mcx-bin/bin/Release/mcx-exe.exe"
MCX_JSON_DISK_PATH = "D:/OneDrive - University of Cambridge/DIS_twin/mcx-python-disk.json"
MCX_MCH_DISK_PATH = "D:/OneDrive - University of Cambridge/DIS_twin/mcx-python-disk.mch"
MCX_JSON_DIFFUSE_PATH = "D:/OneDrive - University of Cambridge/DIS_twin/mcx-python-diffuse.json"
MCX_MCH_DIFFUSE_PATH = "D:/OneDrive - University of Cambridge/DIS_twin/mcx-python-diffuse.mch"
# MCX constants
NPHOTON = 1e7  # number of photos
UNITMM = 0.02  # voxel length unit [mm]

# mu_a, mu_sp constants in mm^-1
"""
MU_A = np.concatenate((np.array([0, 0.0015, 0.003, 0.0045, 0.006, 0.008, 0.01, 0.012, 0.014]),
                       np.power(10, -1.25 + np.tan(np.linspace(-0.5, -0.08, num=10, endpoint=False))),
                       np.power(10, -1.25 + np.tan(np.linspace(-0.08, 0.5, num=18, endpoint=True))),
                       np.array([0.22, 0.25, 0.28, 0.32, 0.36, 0.42, 0.5, 0.6, 0.75, 1.0, 10.0])),
                      axis=0)

MU_SP = np.concatenate((np.array([0., 0.008, 0.015, 0.023, 0.032, 0.04, 0.048]),
                        np.power(10, np.tan(np.linspace(-0.9, -0.4, num=25, endpoint=False))),
                        np.power(10, np.tan(np.linspace(-0.4, -0.1, num=15, endpoint=False))),
                        np.power(10, np.tan(np.linspace(-0.1, 0.8, num=40, endpoint=True))),
                        np.array([12.5, 15, 18.5, 25, 40, 100])),
                        axis=0)

MU_A = MU_A[MU_A < 0.05]
MU_SP = MU_SP[(MU_SP >= 0.5) & (MU_SP <= 2)]
"""
MU_A = np.arange(0, 0.2+0.001, 0.001)
MU_SP = np.arange(0.05, 2+0.01, 0.01)

# mu containing all mu_sp, mu_a pair
# size: n_sample * n_feature
MU = np.zeros((MU_SP.size*MU_A.size, 2))
mu_count = 0
for mu_sp in MU_SP:
    for mu_a in MU_A:
        MU[mu_count, 0] = mu_a
        MU[mu_count, 1] = mu_sp
        mu_count += 1


print(f"Check .json z: \n{[int(THICKNESS_SAMPLE_i / UNITMM) for THICKNESS_SAMPLE_i in THICKNESS_SAMPLE]}")
print(f"MU array size: {MU.shape}")
