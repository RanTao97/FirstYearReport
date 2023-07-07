# EMVA1288_M2 is to measure EMVA1288 dark current

import numpy as np
import time
import PIL.Image

import matplotlib.pyplot as plt

from LUCID_Camera import Camera

import EMVA1288_utils
import EMVA1288_M2_utils

PATH = "20230529_LUCID_ATP013S_230101680_TEC15"  # path to save data, has the name of measurement date

WARMUP_TIME = 0.1  # [min], camera warmup to reach thermal equilibrium
IMG_SEQ = 50#100  # take IMG_SEQ images at each exposure time

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
        # Todo:2023/05/28 set to double black_level for more accurate variance
        # i.e.: ~1312
        camera.set_BlackLevel(black_level*2)
        black_level = camera.get_BlackLevel()

        # Todo: 2023/05/29 add temperature results
        camera.set_TECControlTemperature(15.0)

        """Camera warmup, 6.5 Measurement Condition 1"""
        print(f"\nCamera Warming Up for {WARMUP_TIME} mins, Light Source Should be Warmed Up Too If Necessary **********")
        camera.set_AcquisitionMode("FreeRun")
        camera.prepare_acquisition()
        camera.start_acquisition()
        time.sleep(WARMUP_TIME * 60)  # warmup
        camera.stop_acquisition()
        print(f"\nWarmpup Finishes, "
              f"Double Check Pixel Format {camera.get_PixelFormat()}, Gain {camera.get_Gain()}, and Black Level {camera.get_BlackLevel()}")

        """Measure Dark Current"""
        input(f"\nDark Current Measurement, Block Light Source, Put on Camera Cap"
              f"\nPress Any Key to Continue")

        """Find t_exp_dist"""
        t_exp_min = camera.get_ExposureTime_min()
        t_exp_max = 5000000#camera.get_ExposureTime_max()
        t_exp_dist = EMVA1288_M2_utils.find_t_exp_vals(t_exp_min, t_exp_max)
        # Todo: 2023/05/29 add temperature results
        T_sensor_dist = np.zeros_like(t_exp_dist)
        T_TEC_dist = np.zeros_like(t_exp_dist)

        """Measure at Each t_exp"""
        img0 = np.zeros((M, N))
        img1 = np.zeros((M, N))
        for idx, t_exp in enumerate(t_exp_dist):
            # Todo: 2023/05/29 add temperature results
            T_sensor_dist[idx], T_TEC_dist[idx] = camera.get_DeviceTemperature()
            print(f'Sensor Temperature {T_sensor_dist[idx]} deg C, TEC Temperature {T_TEC_dist[idx]} deg C')

            # set to the t_exp
            print(f"Updating Exposure Time **********")
            camera.set_ExposureTime(t_exp)
            # the real t_exp is set by camera, subject to rounding errors
            t_exp = camera.get_ExposureTime()
            t_exp_dist[idx] = t_exp
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
            np.save(f"{PATH}/M2Data/Exp{idx}_d0.npy", img0)
            np.save(f"{PATH}/M2Data/Exp{idx}_d1.npy", img1)
            img_PIL = PIL.Image.fromarray(img0)
            img_PIL.save(f"{PATH}/M2Data/Exp{idx}_d0.tif")
            img_PIL = PIL.Image.fromarray(img1)
            img_PIL.save(f"{PATH}/M2Data/Exp{idx}_d1.tif")

        """Save Measurement Variables"""
        # Todo: 2023/05/29 add temperature results
        np.savez(f"{PATH}/M2Data/measurement.npz",
                 M=M, N=N,
                 pixel_format_high=pixel_format_high, pixel_format_high_num=pixel_format_high_num,
                 gain_a=gain_a, gain_d=gain_d, black_level=black_level,
                 t_exp_dist=t_exp_dist,
                 T_sensor_dist=T_sensor_dist, T_TEC_dist=T_TEC_dist)
    camera.close_camera()