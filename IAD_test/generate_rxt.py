import numpy as np

import constants
from correct_spheres import correct_spheres

if __name__ == '__main__':
    """Update Constants"""
    Constants = np.load("Data/Constants.npz", allow_pickle=True)
    constants.G = Constants['G']
    constants.THETA = Constants['THETA']
    constants.THICKNESS_SAMPLE = 2.3  # [mm]
    constants.STANDARD_REFLECTANCE = 0.985
    constants.REFRACTIVE_INDEX_SAMPLE = Constants['REFRACTIVE_INDEX_SAMPLE']
    constants.REFRACTIVE_INDEX_GLASS = 1.5
    constants.THICKNESS_GLASS = 0  # [mm]
    constants.BEAM_DIAMETER = Constants['BEAM_DIAMETER']
    constants.WALL_REFLECTANCE = 0.985
    constants.DETECTOR_REFLECTANCE = 0.0
    constants.NUMBER_OF_SPHERES = Constants['NUMBER_OF_SPHERES']
    constants.SPHERE_DIAMETER = 50  # [mm]
    constants.SAMPLE_PORT_DIAMETER = Constants['SAMPLE_PORT_DIAMETER']
    constants.ENTRANCE_PORT_DIAMETER = 4.9  # [mm]
    constants.DETECTOR_PORT_DIAMETER = 0.2  # [mm]
    constants.SPHERE_DIAMETER_2 = constants.SPHERE_DIAMETER
    constants.SAMPLE_PORT_DIAMETER_2 = constants.SAMPLE_PORT_DIAMETER
    constants.ENTRANCE_PORT_DIAMETER_2 = 0.0  # [mm]
    constants.DETECTOR_PORT_DIAMETER_2 = constants.DETECTOR_PORT_DIAMETER
    constants.MU = Constants['MU']
    Constants.close()

    print(constants.MU)

    """Do Sphere Corrections and Write IAD_test.rxt"""
    RT_Diffuse = np.load("Data/thickness2300um/RT_Diffuse_lost.npy")
    RT_Disk = np.load("Data/thickness2300um/RT_Disk_lost.npy")
    # correct sphere and write .rxt results
    # correct spheres: dummy Rc, Tc
    Rc = 0
    Tc = 0
    with open("Data/IAD_test.rxt", "w") as rxt_file:
        rxt_file.writelines(f"IAD1   # Must be first four characters\n\n"
                            f"# The order of entries is important\n"
                            f"# Anything after a '#' is ignored, blank lines are also ignored\n\n"
                            f"{constants.REFRACTIVE_INDEX_SAMPLE:.6f}\t# Index of refraction of the sample\n"
                            f"{constants.REFRACTIVE_INDEX_GLASS:.6f}\t# Index of refraction of the top and bottom slides\n"
                            f"{constants.THICKNESS_SAMPLE:.6f}\t# [mm] Thickness of sample\n"
                            f"{constants.THICKNESS_GLASS:.6f}\t# [mm] Thickness of slides\n"
                            f"{constants.BEAM_DIAMETER:.6f}\t# [mm] Diameter of illumination beam\n"
                            f"{constants.STANDARD_REFLECTANCE:.6f}\t# Reflectivity of the reflectance calibration standard\n\n"
                            f"{constants.NUMBER_OF_SPHERES:.2f}\t# Number of spheres used during each measurement\n\n"
                            f"# Properties of sphere used for reflection measurements\n"
                            f"{constants.SPHERE_DIAMETER:.6f}\t# [mm] Sphere Diameter\n"
                            f"{constants.SAMPLE_PORT_DIAMETER:.6f}\t# [mm] Sample Port Diameter\n"
                            f"{constants.ENTRANCE_PORT_DIAMETER:.6f}\t# [mm] Entrance Port Diameter\n"
                            f"{constants.DETECTOR_PORT_DIAMETER:.6f}\t# [mm] Detector Port Diameter\n"
                            f"{constants.WALL_REFLECTANCE:.6f}\t# Reflectivity of the sphere wall\n\n"
                            f"# Properties of sphere used for transmission measurements\n"
                            f"{constants.SPHERE_DIAMETER_2:.6f}\t# [mm] Sphere Diameter\n"
                            f"{constants.SAMPLE_PORT_DIAMETER_2:.6f}\t# [mm] Sample Port Diameter\n"
                            f"{constants.ENTRANCE_PORT_DIAMETER_2:.6f}\t# [mm] Entrance Port Diameter\n"
                            f"{constants.DETECTOR_PORT_DIAMETER_2:.6f}\t# [mm] Detector Port Diameter\n"
                            f"{constants.WALL_REFLECTANCE:.6f}\t# Reflectivity of the sphere wall\n\n"
                            f"2.000000\t# Number of measurements, M_R, M_T\n\n"
                            f"#lambda\t\tM_R\t\tM_T\t\t#\tmu_a\t\tmu_sp\t\tg\n")
        for idx in range(constants.MU.shape[0]):
            mu_a = constants.MU[idx, 0]
            mu_sp = constants.MU[idx, 1]
            # correct spheres
            M_R, M_T = correct_spheres(RT_Disk[idx, 0], RT_Disk[idx, 1], Rc, Tc, RT_Diffuse[idx, 0], RT_Diffuse[idx, 1])
            # write to the output file
            # idx starting from 2
            rxt_file.writelines(f"{idx + 2}\t\t{M_R:.9f}\t{M_T:.9f}\t"
                                f"#\t{mu_a:.4f}\t\t{mu_sp:.4f}\t\t{constants.G:.3f}\n")
    rxt_file.close()

