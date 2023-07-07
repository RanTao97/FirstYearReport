# EMVA1288_M3 is to measure EMVA1288 spatial nonuniformity and defect pixels

import numpy as np
import time
import PIL.Image

import matplotlib.pyplot as plt

from LUCID_Camera import Camera

import EMVA1288_utils
import EMVA1288_M3_utils

PATH = "20230602_LUCID_ATP013S_230101680_LED1200nm"  # path to save data, has the name of measurement date

WARMUP_TIME = 1  # [min], camera warmup to reach thermal equilibrium
IMG_SEQ = 100  # take IMG_SEQ images at each exposure time

if __name__ == '__main__':
    """Import EMVA M1 Measurement Variables"""
    meas_var = np.load(f"{PATH}/M1Data/measurement.npz")
    M = meas_var['M']
    N = meas_var['N']
    pixel_format_high = str(meas_var['pixel_format_high'])
    pixel_format_high_num = int(meas_var['pixel_format_high_num'])
    gain_a = float(meas_var['gain_a'])
    gain_d = float(meas_var['gain_d'])
    black_level = float(meas_var['black_level'])
    t_exp_sat = float(meas_var['t_exp_sat'])
    meas_var.close()

    """Import EMVA M1 Measurement Results"""
    meas_results = np.load(f"{PATH}/M1Data/results.npz")
    t_exp_dist = meas_results['t_exp_dist']
    muy = meas_results['muy']

    # open camera
    camera = Camera()
    if camera.open_camera() == 0:
        """ Disable Auto Corrections"""
        camera.disable_auto()

        """Set camera bit depth to highest as in EMVA M1"""
        print(f"Setting to the Highest Bit Depth (Unpacked) {pixel_format_high} as in EMVA M1")
        camera.set_PixelFormat(pixel_format_high)

        """Set camera gain to min as in EMVA M1"""
        print(f"Setting to the Min Analog {gain_a} and Digital Gain {gain_d} as in EMV1 M1")
        camera.set_Gain(gain_a, gain_d)

        """Set camera black level as in EMVA M1"""
        print(f"Setting to the Black Level {black_level} as in EMV1 M1")
        camera.set_BlackLevel(black_level)

        """Camera warmup, 6.5 Measurement Condition 1"""
        print(f"\nCamera Warming Up for {WARMUP_TIME} mins, Light Source Should be Warmed Up Too If Necessary **********")
        camera.set_AcquisitionMode("FreeRun")
        camera.prepare_acquisition()
        camera.start_acquisition()
        time.sleep(WARMUP_TIME * 60)  # warmup
        camera.stop_acquisition()
        print(f"\nWarmpup Finishes, "
              f"Double Check Pixel Format {camera.get_PixelFormat()}, Gain {camera.get_Gain()}, and Black Level {camera.get_BlackLevel()}")

        """Verify 100% Saturation Measurement"""
        input(f"\n100% Saturation Verification, Unblock Light Source, Remove Camera Cap"
              f"\nPress Any Key to Continue")
        # set to t_exp_sat
        print(f"Updating Exposure Time **********")
        camera.set_ExposureTime(t_exp_sat)
        # take 9 images and use the last 1
        print(f"Taking 100% Saturation Image **********")
        camera.set_AcquisitionMode("FreeRun")
        frame_rate = camera.get_FrameRate()
        camera.prepare_acquisition()
        camera.start_acquisition()
        for i in range(9):
            time.sleep(1 / frame_rate)
            camera.receive_image()
        time.sleep(1 / frame_rate)
        img_temp = camera.receive_image()
        img = np.copy(img_temp)
        camera.stop_acquisition()
        img_PIL = PIL.Image.fromarray(img)
        img_PIL.save(f"{PATH}/M3Data/Saturation.tif")

        """Find t_exp_50"""
        t_exp_50 = EMVA1288_M3_utils.find_t_exp_50(t_exp_dist, muy)

        """50% Saturation Measurement"""
        input(f"\n50% Saturation Measurement, Unblock Light Source, Remove Camera Cap"
              f"\nPress Any Key to Continue")
        # set to t_exp_50
        print(f"Updating Exposure Time **********")
        camera.set_ExposureTime(t_exp_50)
        # the real t_exp is set by camera, subject to rounding errors
        t_exp_50 = camera.get_ExposureTime()
        # take (IMG_SEQ + 10) images and discard the first 10 images
        print(f"Taking Bright Images **********")
        camera.set_AcquisitionMode("FreeRun")
        frame_rate = camera.get_FrameRate()
        camera.prepare_acquisition()
        camera.start_acquisition()
        for i in range(10):
            time.sleep(1 / frame_rate)
            camera.receive_image()
        for i in range(IMG_SEQ):
            time.sleep(1 / frame_rate)
            img_temp = camera.receive_image()
            img = np.copy(img_temp)
            # save image
            np.save(f"{PATH}/M3Data/Exp50_{i}.npy", img)
            img_PIL = PIL.Image.fromarray(img)
            img_PIL.save(f"{PATH}/M3Data/Exp50_{i}.tif")
        camera.stop_acquisition()

        """Dark Measurement"""
        input(f"\nDark Measurement, Block Light Source, Put On Camera Cap"
              f"\nPress Any Key to Continue")
        # double check camera parameters
        print(f"Double Check Pixel Format {camera.get_PixelFormat()}, Gain {camera.get_Gain()}, "
              f"Black Level {camera.get_BlackLevel()}, and Exposure Time {camera.get_ExposureTime()}")
        # take (IMG_SEQ + 10) images and discard the first 10 images
        print(f"Taking Dark Images **********")
        camera.set_AcquisitionMode("FreeRun")
        frame_rate = camera.get_FrameRate()
        camera.prepare_acquisition()
        camera.start_acquisition()
        for i in range(10):
            time.sleep(1 / frame_rate)
            camera.receive_image()
        for i in range(IMG_SEQ):
            time.sleep(1 / frame_rate)
            img_temp = camera.receive_image()
            img = np.copy(img_temp)
            # save image
            np.save(f"{PATH}/M3Data/Dark_{i}.npy", img)
            img_PIL = PIL.Image.fromarray(img)
            img_PIL.save(f"{PATH}/M3Data/Dark_{i}.tif")
        camera.stop_acquisition()

        """Save Measurement Variables"""
        np.savez(f"{PATH}/M3Data/measurement.npz",
                 IMG_SEQ=IMG_SEQ,
                 M=M, N=N,
                 pixel_format_high=pixel_format_high, pixel_format_high_num=pixel_format_high_num,
                 gain_a=gain_a, gain_d=gain_d, black_level=black_level,
                 t_exp_50=t_exp_50)
    camera.close_camera()
