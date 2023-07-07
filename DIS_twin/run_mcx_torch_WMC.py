"""
functions are similar to run_mcx_torch
but adapts to WMC
hard-coded for json generation
"""

import numpy as np
import torch
import json
import subprocess
from os import path

import constants


def run_mcx(json_path):
    """
    run mcx
    :param json_path: the .json file path
    :return:
    """
    cmd = list()
    cmd.append(constants.MCX_PATH)
    # .json path
    cmd.append(f"-f")
    cmd.append(f"{json_path}")
    # use GPU 1
    cmd.append(f"--gpu")
    cmd.append(f"1")
    # per-face boundary condition, capture photons exiting from top and bottom surfaces
    # allow light to be lost from lateral surfaces
    cmd.append(f"--bc")
    cmd.append(f"______001001")
    # max number of detected photons
    cmd.append(f"--maxdetphoton")
    cmd.append(f"{2 * constants.NPHOTON}")
    subprocess.run(cmd)


def generate_json_Disk(file, mu_a, mu_sp, thickness):
    """
    to generate .json file for mcx.exe
    :param file: .json file path
    :param mu_a: mu_a [mm-1]
    :param mu_sp: mu_sp [mm-1]
    :param thickness: sample thickness [mm]
    :return: vol: volume dimensions
    """
    # set volume dimensions
    dim_z = int(thickness / constants.UNITMM + 2)
    # Todo: 18 for UNITMM=0.02
    #       20 for UNITMM=0.1
    #       To meet the GPU memory
    dim_x = int(thickness / constants.UNITMM) * 15
    dim_y = dim_x
    dim_centre = dim_x / 2
    vol = [dim_x, dim_y, dim_z]
    beam_r = constants.BEAM_DIAMETER / 2 / constants.UNITMM

    src_dict = {
        "Source": {
            "Type": "Disk",  # source type
            # source position: ensure incident at volume centre
            "Pos": [dim_centre - (beam_r+1/constants.UNITMM)*np.sin(constants.THETA / 180 * np.pi)*np.tan(constants.THETA / 180 * np.pi),
                    dim_centre,
                    1 - (beam_r+1/constants.UNITMM)*np.sin(constants.THETA / 180 * np.pi)],
            # source direction
            "Dir": [np.sin(constants.THETA / 180 * np.pi), 0, np.cos(constants.THETA / 180 * np.pi)],
            # source parameter: beam radius in voxel grid
            "Param1": [beam_r, 0, 0, 0]
        }
    }

    json_dict = {
            "Session": {
                "ID": "mcx-python-disk",
                "DoAutoThread": 1,  # automatically set threads and blocks
                "Photons": constants.NPHOTON,  # number of photons
                "RNGSeed": -1,  # set seed to make the simulation reproducible

                "DoMismatch": 1,  # consider refractive index mismatch
                "DoNormalize": 0,  # not normalise output fluence
                "DoSpecular": 1,  # consider specular reflection

                "SaveDataMask": "DSPXV",  # save DSPXV of the detected photon
                "DoSaveVolume": 0  # not save volumetric fluence, to speed up
            },

            "Forward": {
                "T0": 0,  # starting time
                "T1": 10e-9,  # ending time, need sufficiently long time window for CW illumination
                "Dt": 10e-9  # time-gate width = overall time window
            },

            "Optode": src_dict,

            "Domain": {
                "LengthUnit": constants.UNITMM,
                "OriginType": 1,  # first voxel is [0, 0, 0]
                "Media": [
                    {
                        "mua": 0,
                        "mus": 0,
                        "g": 1,
                        "n": 1
                    },
                    {
                        "mua": mu_a,
                        "mus": mu_sp / (1 - constants.G),
                        "g": constants.G,
                        "n": constants.REFRACTIVE_INDEX_SAMPLE
                    },
                    {
                        "mua": 0,
                        "mus": 0,
                        "g": 1,
                        "n": 1
                    }
                ],
                "Dim": vol
            },
            "Shapes": [
                {"Origin": [0, 0, 0]},
                {"Grid": {"Tag": 1, "Size": vol}},
                # air layer above and beneath the sample, to allow wide-field detection
                {"ZLayers": [[1,1,2], [dim_z, dim_z, 2]]}
            ]
        }

    # write to .json file
    with open(file, "w") as f:
        json.dump(json_dict, f)
        f.close()

    return vol

def generate_json_Diffuse(file, mu_a, mu_sp, thickness):
    """
    to generate .json file for mcx.exe
    :param file: .json file path
    :param mu_a: mu_a [mm-1]
    :param mu_sp: mu_sp [mm-1]
    :param thickness: sample thickness [mm]
    :return: vol: volume dimensions
    """
    # set volume dimensions
    dim_z = int(thickness / constants.UNITMM + 2)
    # Todo: 18 for UNITMM=0.02
    #       20 for UNITMM=0.1
    #       To meet the GPU memory
    dim_x = int(thickness / constants.UNITMM) * 15
    dim_y = dim_x
    dim_centre = dim_x / 2
    vol = [dim_x, dim_y, dim_z]

    src_dict = {
        "Source": {
            "Type": "Diffuse",  # source type
            # source position: ensure incident at volume centre
            "Pos": [dim_centre, dim_centre, 1 - 1e-7],
            # source direction
            "Dir": [0, 0, 1],
            # source parameter: beam radius in voxel grid
            "Param1": [constants.SAMPLE_PORT_DIAMETER / 2 / constants.UNITMM, 0, 0, 0]
        }
    }

    json_dict = {
            "Session": {
                "ID": "mcx-python-diffuse",
                "DoAutoThread": 1,  # automatically set threads and blocks
                "Photons": constants.NPHOTON,  # number of photons
                "RNGSeed": -1,  # set seed to make the simulation reproducible

                "DoMismatch": 1,  # consider refractive index mismatch
                "DoNormalize": 0,  # not normalise output fluence
                "DoSpecular": 1,  # consider specular reflection

                "SaveDataMask": "DSPXV",  # save DSPXV of the detected photon
                "DoSaveVolume": 0  # not save volumetric fluence, to speed up
            },

            "Forward": {
                "T0": 0,  # starting time
                "T1": 10e-9,  # ending time, need sufficiently long time window for CW illumination
                "Dt": 10e-9  # time-gate width = overall time window
            },

            "Optode": src_dict,

            "Domain": {
                "LengthUnit": constants.UNITMM,
                "OriginType": 1,  # first voxel is [0, 0, 0]
                "Media": [
                    {
                        "mua": 0,
                        "mus": 0,
                        "g": 1,
                        "n": 1
                    },
                    {
                        "mua": mu_a,
                        "mus": mu_sp / (1 - constants.G),
                        "g": constants.G,
                        "n": constants.REFRACTIVE_INDEX_SAMPLE
                    },
                    {
                        "mua": 0,
                        "mus": 0,
                        "g": 1,
                        "n": 1
                    }
                ],
                "Dim": vol
            },
            "Shapes": [
                {"Origin": [0, 0, 0]},
                {"Grid": {"Tag": 1, "Size": vol}},
                # air layer above and beneath the sample, to allow wide-field detection
                {"ZLayers": [[1,1,2], [dim_z, dim_z, 2]]}
            ]
        }

    # write to .json file
    with open(file, "w") as f:
        json.dump(json_dict, f)
        f.close()

    return vol


def read_mch(file):
    """
    to read .mch file
    :param file: .mch file path
    :return: pdata_torch, detflag
    pdata is a numpy array in the order of:
    [detid(1)       nscat(M)        ppath(M)            mom(M)      p(3)            v(3)        w0(1)]
    detector id     #scattering     pathlength [mm]     momentum    position [mm]   direction   initial weight
    M: #media
    some are optional depending on input file
    """
    if not path.isfile(file):
        raise Exception(f"Cannot find file")

    with open(file, mode="rb") as f:
        # read magic header
        magicheader = f.read(4).decode('ascii')
        if magicheader != "MCXH":
            f.close()
            raise Exception("Not .mch file")

        # read header
        # version, maxmedia, detnum, colcount, totalphoton, detected, savedphoton
        version, maxmedia, detnum, colcount, totalphoton, detected, savedphoton = np.fromfile(f, dtype='I', count=7)

        # read properties
        unitmm, seedbyte, normalizer, respin, srcnum, savedetflag, _, _ = np.fromfile(f,
                dtype=np.dtype(('f,I,f,i,I,I,I,I')), count=1)[0]

        # read photon data
        pdata = np.fromfile(f, dtype='f', count=colcount*savedphoton)
        # convert to torch for acceleration
        pdata_torch = torch.from_numpy(pdata)
        pdata_torch = torch.reshape(pdata_torch, (savedphoton, colcount))

        # close file
        f.close()

        # check det flag
        # DSPMXVW
        detflag = np.unpackbits(np.uint8(savedetflag), bitorder="little")

        # convert ppath to [mm]
        # ppath column = (detflag[0] + detflag[1] * maxmedia) : (detflag[0] + (detflag[1] + detflag[2]) * maxmedia)
        # i.e., after id, nscat
        if detflag[2] == 1:
            pdata_torch[:, (detflag[0]+detflag[1]*maxmedia):(detflag[0]+(detflag[1]+detflag[2])*maxmedia)] = \
                pdata_torch[:, (detflag[0]+detflag[1]*maxmedia):(detflag[0]+(detflag[1]+detflag[2])*maxmedia)] * unitmm

    return pdata_torch, detflag



def mcxdetweight(pdata_torch, detflag, mu_a):
    """
    to calculate the remaining photon weight after absorption
    :param pdata_torch: photon data read from .mch file
    :param detflag: det flag from simulation "DSPMXVW", determines pdata columns
    :param mu_a: mu_a in [mm-1]
    :return: detw: total detected weight
    """
    maxmedia = 2  # always 2 media in the simulation
    detw = torch.exp(-1 * pdata_torch[:, (detflag[0]+detflag[1]*maxmedia)] * mu_a)

    return torch.sum(detw)


def find_RT(pdata_torch, detflag, unitinmm, vol, mu_a, nphoton, d_sample_port):
    """
    to find the total reflectance (R) and transmittance (T)
    :param pdata_torch: photon data read from .mch file
    :param detflag: det flag from simulation "DSPMXVW", determines pdata columns
    :param unitinmm: unit in [mm]
    :param vol: volume dimension array
    :param mu_a: mu_a in [mm-1]
    :param nphoton: number of photons
    :param d_sample_port: sample port diameter [mm]
    :return: R, T
    """
    #  source position, assume in the centre of xy plane
    srcpos = [vol[0]/2, vol[1]/2]

    # sample port radius in defined volume grid
    Nr = d_sample_port / unitinmm / 2

    # convert detected position to exit position
    # in [x, y, z]
    # use torch for acceleration
    exit_position = torch.zeros((pdata_torch.size(dim=0), 3))

    maxmedia = 2  # always 2 media
    px_col = detflag[0]+(detflag[1]+detflag[2]+detflag[3])*maxmedia  # position x column in pdata
    py_col = px_col + 1  # position y column in pdata
    pz_col = py_col + 1  # position z column in pdata

    vx_col = detflag[0]+(detflag[1]+detflag[2]+detflag[3])*maxmedia + detflag[4]*3  # direction x column in pdata
    vy_col = vx_col + 1  # direction y column in pdata
    vz_col = vy_col + 1  # direction z column in pdata

    top_surface_idx = (np.abs(pdata_torch[:, pz_col] - 0) < 1)

    exit_position[top_surface_idx, 0:2] = pdata_torch[top_surface_idx, px_col:pz_col] - pdata_torch[top_surface_idx, vx_col:vz_col] / \
                                    (pdata_torch[top_surface_idx, vz_col][:,None] + 1e-7) * (pdata_torch[top_surface_idx, pz_col][:,None] - 1)
    exit_position[top_surface_idx, 2] = pdata_torch[top_surface_idx, pz_col] - (pdata_torch[top_surface_idx, pz_col] - 1)
    """ slower:
    exit_position[top_surface_idx, 0] = pdata_torch[top_surface_idx, px_col] - pdata_torch[top_surface_idx, vx_col] / \
                                    (pdata_torch[top_surface_idx, vz_col] + 1e-7) * (pdata_torch[top_surface_idx, pz_col] - 1)
    exit_position[top_surface_idx, 1] = pdata_torch[top_surface_idx, py_col] - pdata_torch[top_surface_idx, vy_col] / \
                                    (pdata_torch[top_surface_idx, vz_col] + 1e-7) * (pdata_torch[top_surface_idx, pz_col] - 1)
    exit_position[top_surface_idx, 2] = pdata_torch[top_surface_idx, pz_col] - (pdata_torch[top_surface_idx, pz_col] - 1)
    """

    bottom_surface_idx = (np.abs(pdata_torch[:, pz_col] - vol[2]) < 1)
    exit_position[bottom_surface_idx, 0:2] = pdata_torch[bottom_surface_idx, px_col:pz_col] - pdata_torch[bottom_surface_idx, vx_col:vz_col] / \
                                           (pdata_torch[bottom_surface_idx, vz_col][:,None] + 1e-7) * (pdata_torch[bottom_surface_idx, pz_col][:,None] - (vol[2] - 1))
    exit_position[bottom_surface_idx, 2] = pdata_torch[bottom_surface_idx, pz_col] - (pdata_torch[bottom_surface_idx, pz_col] - (vol[2] - 1))
    """ slower:
    exit_position[bottom_surface_idx, 0] = pdata_torch[bottom_surface_idx, px_col] - pdata_torch[bottom_surface_idx, vx_col] / \
                                    (pdata_torch[bottom_surface_idx, vz_col] + 1e-7) * (pdata_torch[bottom_surface_idx, pz_col] - (vol[2]-1))
    exit_position[bottom_surface_idx, 1] = pdata_torch[bottom_surface_idx, py_col] - pdata_torch[bottom_surface_idx, vy_col] / \
                                    (pdata_torch[bottom_surface_idx, vz_col] + 1e-7) * (pdata_torch[bottom_surface_idx, pz_col] - (vol[2]-1))
    exit_position[bottom_surface_idx, 2] = pdata_torch[bottom_surface_idx, pz_col] - (pdata_torch[bottom_surface_idx, pz_col] - (vol[2]-1))
    """

    # find R and T within sample port
    port_idx = (((exit_position[:,0] - srcpos[0])**2 + (exit_position[:,1] - srcpos[1])**2) <= Nr**2)
    R = mcxdetweight(pdata_torch[port_idx*top_surface_idx, :], detflag, mu_a) / nphoton
    T = mcxdetweight(pdata_torch[port_idx*bottom_surface_idx, :], detflag, mu_a) / nphoton

    return R, T
