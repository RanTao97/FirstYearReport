# EMVA1288_M4 is to measure EMVA1288 spectral sensitivity

import numpy as np
import time
import PIL.Image

import matplotlib.pyplot as plt

from LUCID_Camera import Camera

import EMVA1288_utils
import EMVA1288_M4_utils

PATH = "20230518_LUCID_ATP013S_230101680"  # path to save data, has the name of measurement date

WARMUP_TIME = 0.1  # [min], camera warmup to reach thermal equilibrium
IMG_SEQ = 100  # take IMG_SEQ images at each wavelength

No_WAVELENGTH = 6  # number of measured wavelengths

if __name__ == '__main__':
    """Import EMVA M1 Measurement Variables"""
    meas_var = np.load(f"{PATH}/M1Data/measurement.npz")
    A = meas_var['A']
    M = meas_var['M']
    N = meas_var['N']
    pixel_format_high = str(meas_var['pixel_format_high'])
    pixel_format_high_num = int(meas_var['pixel_format_high_num'])
    gain_a = float(meas_var['gain_a'])
    gain_d = float(meas_var['gain_d'])
    black_level = float(meas_var['black_level'])
    mup_dist_M1 = (meas_var['mup_dist']).astype(float)
    meas_var.close()

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

        """Measure at Each Wavelength"""
        WAVELENGTH_dist = np.zeros(No_WAVELENGTH)
        E_dist = np.zeros(No_WAVELENGTH)
        t_exp_dist = np.zeros(No_WAVELENGTH)
        mup_dist = np.zeros(No_WAVELENGTH)
        img0 = np.zeros((M, N))
        img1 = np.zeros((M, N))
        for idx in range(No_WAVELENGTH):
            print(f"\nMeasurement {idx}: **********")
            while True:
                inp = input("Enter wavelength (nm) measured by the spectrometer: ")
                try:
                    WAVELENGTH_dist[idx] = inp
                    break
                except:
                    print("Invalid Input, Try Again")
            while True:
                inp = input("Enter irradiance (uW/cm^2) measured by the calibrated photodiode: ")
                try:
                    E_dist[idx] = inp
                    break
                except:
                    print("Invalid Input, Try Again")
            input(f"Going to Start the Bright Measurement, Unblock Light Source, Remove Camera Cap\n"
                  f"Ensure the Light Source is Warmed Up if Necessary\n"
                  f"Press Any Key to Continue")
            # find the exposure time that will produce similar mup
            print(f"Updating Exposure Time **********")
            t_exp = EMVA1288_M4_utils.find_t_exp(mup_dist_M1, A, E_dist[idx], WAVELENGTH_dist[idx])
            # set the exposure time
            camera.set_ExposureTime(t_exp)
            # the real t_exp is set by camera, subject to rounding errors
            t_exp = camera.get_ExposureTime()
            t_exp_dist[idx] = t_exp
            # update the corresponding mup at the measured wavelength
            mup_dist[idx] = EMVA1288_utils.calc_mup(A, E_dist[idx], t_exp, WAVELENGTH_dist[idx])
            # take IMG_SEQ consecutive images, use the No.40% and No.60% image
            print(f"Taking Bright Images **********")
            camera.set_AcquisitionMode("FreeRun")
            frame_rate = camera.get_FrameRate()
            camera.prepare_acquisition()
            camera.start_acquisition()
            for i in range(IMG_SEQ):
                time.sleep(1 / frame_rate)
                if i == int(IMG_SEQ * 0.4):
                    img_temp = camera.receive_image()
                    img0 = np.copy(img_temp)
                elif i == int(IMG_SEQ * 0.6):
                    img_temp = camera.receive_image()
                    img1 = np.copy(img_temp)
                else:
                    camera.receive_image()
            camera.stop_acquisition()
            np.save(f"{PATH}/M4Data/Wave{idx}_b0.npy", img0)
            np.save(f"{PATH}/M4Data/Wave{idx}_b1.npy", img1)
            img_PIL = PIL.Image.fromarray(img0)
            img_PIL.save(f"{PATH}/M4Data/Wave{idx}_b0.tif")
            img_PIL = PIL.Image.fromarray(img1)
            img_PIL.save(f"{PATH}/M4Data/Wave{idx}_b1.tif")
            print(f"Bright Measurement {idx} Finishes\n"
                  f"Double Check Pixel Format {camera.get_PixelFormat()}, Gain {camera.get_Gain()}, "
                  f"Black Level {camera.get_BlackLevel()}, and Exposure Time {camera.get_ExposureTime()} us")
        print(f"All Bright Measurement Finishes")

        input(f"\nDark Measurement, Block Light Source, Put On Camera Cap\n"
              f"Press Any Key to Continue")
        for idx in range(No_WAVELENGTH):
            print(f"Updating Exposure Time for Measurement {idx} **********")
            camera.set_ExposureTime(t_exp_dist[idx])
            # take IMG_SEQ consecutive images, use the No.40% and No.60% image
            print(f"Taking Dark Images **********")
            camera.set_AcquisitionMode("FreeRun")
            frame_rate = camera.get_FrameRate()
            camera.prepare_acquisition()
            camera.start_acquisition()
            for i in range(IMG_SEQ):
                time.sleep(1 / frame_rate)
                if i == int(IMG_SEQ * 0.4):
                    img_temp = camera.receive_image()
                    img0 = np.copy(img_temp)
                elif i == int(IMG_SEQ * 0.6):
                    img_temp = camera.receive_image()
                    img1 = np.copy(img_temp)
                else:
                    camera.receive_image()
            camera.stop_acquisition()
            np.save(f"{PATH}/M4Data/Wave{idx}_d0.npy", img0)
            np.save(f"{PATH}/M4Data/Wave{idx}_d1.npy", img1)
            img_PIL = PIL.Image.fromarray(img0)
            img_PIL.save(f"{PATH}/M4Data/Wave{idx}_d0.tif")
            img_PIL = PIL.Image.fromarray(img1)
            img_PIL.save(f"{PATH}/M4Data/Wave{idx}_d1.tif")
            print(f"Dark Measurement {idx} Finishes\n"
                  f"Double Check Pixel Format {camera.get_PixelFormat()}, Gain {camera.get_Gain()}, "
                  f"Black Level {camera.get_BlackLevel()}, and Exposure Time {camera.get_ExposureTime()} us")
        print(f"All Dark Measurement Finishes")

        """Save Measurement Variables"""
        np.savez(f"{PATH}/M4Data/measurement.npz",
                 No_WAVELENGTH=No_WAVELENGTH,
                 E_dist=E_dist, A=A, WAVELENGTH_dist=WAVELENGTH_dist,
                 M=M, N=N,
                 pixel_format_high=pixel_format_high, pixel_format_high_num=pixel_format_high_num,
                 gain_a=gain_a, gain_d=gain_d, black_level=black_level,
                 t_exp_dist=t_exp_dist, mup_dist=mup_dist)
    camera.close_camera()




