import numpy as np
import subprocess
from os import path
from multiprocessing import Pool

import constants


def run_iad(input_file):
    """
    this is to run IAD given the input file
    :param input_file: input file
    :return:
    """
    if not path.isfile(input_file):
        print(f"Cannot find input file")
        return

    print("Starting IAD program")
    cmd = list()
    cmd.append(constants.IAD_PATH)
    cmd.append("-q 12")  # number of quadrature points (default=8)
    cmd.append(f"-i {constants.THETA}")  # light is incident at this angle in degrees
    cmd.append(f"-g {constants.G}")  # fixed scattering anisotropy (default 0)
    cmd.append(input_file)
    subprocess.run(cmd)

if __name__ == "__main__":
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

    print(constants.THETA, constants.G)

    with Pool(10) as p:
        p.map(run_iad, ["Data/IAD_test.rxt"])
