import os

import numpy as np
import json
import subprocess
import time

import constants
from correct_spheres import correct_spheres
from run_mcx_torch_WMC import *

def DIS_twin(date):
    """
    :param date: date for simulation
    :return: nothing
    """
    # save constants
    np.savez(f"Data_WMC/{date}/Constants", G=constants.G, THETA=constants.THETA,
             THICKNESS_SAMPLE=constants.THICKNESS_SAMPLE,
             STANDARD_REFLECTANCE=constants.STANDARD_REFLECTANCE,
             REFRACTIVE_INDEX_SAMPLE=constants.REFRACTIVE_INDEX_SAMPLE, REFRACTIVE_INDEX_GLASS=constants.REFRACTIVE_INDEX_GLASS,
             THICKNESS_GLASS=constants.THICKNESS_GLASS,
             BEAM_DIAMETER=constants.BEAM_DIAMETER,
             WALL_REFLECTANCE=constants.WALL_REFLECTANCE, DETECTOR_REFLECTANCE=constants.DETECTOR_REFLECTANCE,
             NUMBER_OF_SPHERES=constants.NUMBER_OF_SPHERES,
             SPHERE_DIAMETER=constants.SPHERE_DIAMETER, SAMPLE_PORT_DIAMETER=constants.SAMPLE_PORT_DIAMETER,
             ENTRANCE_PORT_DIAMETER=constants.ENTRANCE_PORT_DIAMETER, DETECTOR_PORT_DIAMETER=constants.DETECTOR_PORT_DIAMETER,
             SPHERE_DIAMETER_2=constants.SPHERE_DIAMETER_2, SAMPLE_PORT_DIAMETER_2=constants.SAMPLE_PORT_DIAMETER_2,
             ENTRANCE_PORT_DIAMETER_2=constants.ENTRANCE_PORT_DIAMETER_2, DETECTOR_PORT_DIAMETER_2=constants.DETECTOR_PORT_DIAMETER_2,
             NPHOTON=constants.NPHOTON, UNITMM=constants.UNITMM,
             MU=constants.MU)

    # store RT_Disk, RT_Diffuse
    RT_Disk = np.zeros((constants.MU.shape[0], 2))
    RT_Diffuse = np.zeros((constants.MU.shape[0], 2))

    # Todo: for sphere correction later, not now
    """
    # correct spheres: dummy Rc, Tc
    Rc = 0
    Tc = 0
    # store total corrected M_R, M_T
    M_R = np.zeros((constants.MU.shape[0],))
    M_T = np.zeros((constants.MU.shape[0],))
    """

    for thickness in constants.THICKNESS_SAMPLE:

        mu_idx = 0  # mu idx counter
        for mu_sp in constants.MU_SP:
            mu_a = 0.0  # WMC assumes 0 mu_a

            # Source = Disk
            vol_Disk = generate_json_Disk(constants.MCX_JSON_DISK_PATH, mu_a, mu_sp, thickness)
            run_mcx(constants.MCX_JSON_DISK_PATH)
            pdata_Disk, detflag_Disk = read_mch(constants.MCX_MCH_DISK_PATH)

            # Source = Diffuse
            vol_Diffuse = generate_json_Diffuse(constants.MCX_JSON_DIFFUSE_PATH, mu_a, mu_sp, thickness)
            run_mcx(constants.MCX_JSON_DIFFUSE_PATH)
            pdata_Diffuse, detflag_Diffuse = read_mch(constants.MCX_MCH_DIFFUSE_PATH)

            # scale to a specific mu_a
            for mu_a in constants.MU_A:
                RT_Disk[mu_idx, 0], RT_Disk[mu_idx, 1] = find_RT(pdata_Disk, detflag_Disk, constants.UNITMM, vol_Disk,
                                                                 mu_a, constants.NPHOTON, constants.SAMPLE_PORT_DIAMETER)

                RT_Diffuse[mu_idx, 0], RT_Diffuse[mu_idx, 1] = find_RT(pdata_Diffuse, detflag_Diffuse, constants.UNITMM, vol_Diffuse,
                                                                 mu_a, constants.NPHOTON,
                                                                 constants.SAMPLE_PORT_DIAMETER)

                # print(f"UR1: {RT_Disk[mu_idx, 0]}\tUT1: {RT_Disk[mu_idx, 1]}")
                # print(f"URU: {RT_Diffuse[mu_idx, 0]}\tUTU: {RT_Diffuse[mu_idx, 1]}")
                # Todo: no sphere corrections for now
                # # sphere corrections
                # M_R[mu_idx], M_T[mu_idx] = correct_spheres(RT_Disk[mu_idx, 0], RT_Disk[mu_idx, 1],
                #                                            Rc, Tc,
                #                                            RT_Diffuse[mu_idx, 0], RT_Diffuse[mu_idx, 1])
                mu_idx += 1  # increment mu_idx counter

        # create a data folder for this thickness
        thickness_path = f"Data_WMC/{date}/thickness{int(thickness*1000):d}um"
        os.mkdir(thickness_path)

        # save RT results
        np.save(thickness_path+"/RT_Disk_lost", RT_Disk)
        np.save(thickness_path+"/RT_Diffuse_lost", RT_Diffuse)
        # Todo: no sphere corrections now
        """
        # save M_R, M_T
        np.save(thickness_path+"/M_R", M_R)
        np.save(thickness_path+"/M_T", M_T)
        """


if __name__ == "__main__":
    date = "20230612_BioPixS_D7"
    t = time.time()
    DIS_twin(date)
    print(time.time()-t)

