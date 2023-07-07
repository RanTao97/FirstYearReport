# EMVA1288_M1 is to measure EMVA1288 (1) sensitivity, (2) temporal noise, (3) linearity

import numpy as np
import time
import re
import PIL.Image

import matplotlib.pyplot as plt

from LUCID_Camera import Camera

import EMVA1288_utils
import EMVA1288_M1_utils

PATH = "20230602_LUCID_ATP013S_230101680_LED1200nm"  # path to save data, has the name of measurement date

WARMUP_TIME = 0.1  # [min], camera warmup to reach thermal equilibrium
IMG_SEQ = 100  # take IMG_SEQ images at each exposure time

A = 25  # pixel area [um^2] read from camera datasheet

if __name__ == '__main__':
    # open camera
    camera = Camera()
    if camera.open_camera() == 0:
        """ Disable Auto Corrections"""
        camera.disable_auto()

        """Set camera bit depth to highest, 6.5 Measurement Condition 2"""
        print(f"Setting Camera Bit Depth **********")
        pixel_format_list = camera.get_PixelFormat_list()
        pixel_format_prefix = [re.findall(r'\D+', s)[0] for s in pixel_format_list]
        pixel_format_high_num = sorted([int(re.findall(r'\d+', s)[0]) for s in pixel_format_list])[-1]
        pixel_format_high = pixel_format_prefix[0] + str(pixel_format_high_num)
        print(f"Available Pixel Format {pixel_format_list}")
        print(f"Setting to the Highest Bit Depth (Unpacked) {pixel_format_high}")
        camera.set_PixelFormat(pixel_format_high)

        """Set camera gain to min, 6.5 Measurement Condition 3"""
        print(f"\nSetting Camera Gain **********")
        # Todo: gain
        gain_a = camera.get_Gain_a_min()
        gain_d = camera.get_Gain_d_min()
        print(f"Min Analog Gain {gain_a}\tMin Digital Gain {gain_d}")
        print(f"Setting to the Min Analog and Digital Gain")
        camera.set_Gain(gain_a, gain_d)

        """Camera warmup, 6.5 Measurement Condition 1"""
        print(f"\nCamera Warming Up for {WARMUP_TIME} mins, Light Source Should be Warmed Up Too If Necessary **********")
        camera.set_AcquisitionMode("FreeRun")
        camera.prepare_acquisition()
        camera.start_acquisition()
        time.sleep(WARMUP_TIME*60)  # warmup
        camera.stop_acquisition()
        print(f"\nWarmpup Finishes, Double Check Pixel Format {camera.get_PixelFormat()} and Gain {camera.get_Gain()}")

        """Input Illumination Information"""
        time.sleep(1)  # wait 1s for inputs
        while True:
            inp = input("Enter wavelength (nm) measured by the spectrometer: ")
            try:
                WAVELENGTH = float(inp)  # illumination wavelength [nm] measured by the spectrometer
                break
            except:
                print("Invalid Input, Try Again")
        while True:
            inp = input("Enter irradiance E (uW/cm^2) measured by the calibrated photodiode: ")
            try:
                E = float(inp)  # Irradiance E [uW/cm^2] measured by the calibrated photodiode
                break
            except:
                print("Invalid Input, Try Again")

        """Dark Measurement: Adjust offset, 6.5 Measurement Condition 4"""
        input(f"\nDark Measurement to Adjust Offset, Block Light Source, Put on Camera Cap"
              f"\nPress Any Key to Continue")
        print(f"\nIn Dark Measurement, Setting to Longest Exposure Time **********")
        camera.set_ExposureTime(camera.get_ExposureTime_max())
        print(f"\nTo Find Min Grey Value, Setting to 0 Black Level **********")
        # take 1 image under 0 black level to find the min grey value
        camera.set_BlackLevel(0)
        camera.set_AcquisitionMode("FreeRun")
        frame_rate = camera.get_FrameRate()
        camera.prepare_acquisition()
        camera.start_acquisition()
        time.sleep(1 / frame_rate)
        img_temp = camera.receive_image()
        img = np.copy(img_temp)
        camera.stop_acquisition()
        y = np.amin(img)
        print(f"The Theoretical Min Grey Value is 0. "
              f"Under Min Black Level {camera.get_BlackLevel()}, the Min Grey Value is {y}.\n")
        if EMVA1288_M1_utils.check_con_dark(img, y) == 1:
            print(f"6.5 Measurement Condition 4 Not Met, Adjusting Black Level **********")
            black_level = 1
            while True:
                camera.set_BlackLevel(black_level)
                camera.set_AcquisitionMode("FreeRun")
                frame_rate = camera.get_FrameRate()
                camera.prepare_acquisition()
                camera.start_acquisition()
                # Todo: could take more images
                # take 3 images and use the last one
                for i in range(2):
                    time.sleep(1/frame_rate)
                    camera.receive_image()
                time.sleep(1 / frame_rate)
                img_temp = camera.receive_image()
                img = np.copy(img_temp)
                camera.stop_acquisition()
                if EMVA1288_M1_utils.check_con_dark(img, y) == 1:
                    print(f"6.5 Measurement Condition 4 Not Met, Adjusting Black Level **********")
                    black_level += 1
                else:
                    print(f"6.5 Measurement Condition 4 Met")
                    print(f"Dark Measurement Min {np.amin(img)}")
                    print(f"Dark Measurement Max {np.amax(img)}")
                    img_PIL = PIL.Image.fromarray(img)
                    img_PIL.save(f"{PATH}/M1Data/dark.tif")
                    black_level = camera.get_BlackLevel()
                    break
        else:
            print(f"6.5 Measurement Condition 4 Met")
            print(f"Dark Measurement Min {np.amin(img)}")
            print(f"Dark Measurement Max {np.amax(img)}")
            img_PIL = PIL.Image.fromarray(img)
            img_PIL.save(f"{PATH}/M1Data/dark.tif")
            black_level = camera.get_BlackLevel()
        print(f"\nDark Measurement Finishes, ")
        print(f"Black Level is {black_level} under Longest Exposure Time {camera.get_ExposureTime()} us,"
              f"\nDouble Check Pixel Format {camera.get_PixelFormat()} and Gain {camera.get_Gain()}\n")

        """Saturation Measurement: Find Sat Exposure Time, 6.6 Saturation Condition"""
        input(f"\nSaturation Measurement to Find Sat Exposure Time, Unblock Light Source, Remove Camera Cap"
              f"\nPress Any Key to Continue")
        print(f"\nTo Find Max Grey Value, Setting to Longest Exposure Time **********")
        camera.set_ExposureTime(camera.get_ExposureTime_max())
        # take 1 image to find the max grey value
        camera.set_AcquisitionMode("FreeRun")
        frame_rate = camera.get_FrameRate()
        camera.prepare_acquisition()
        camera.start_acquisition()
        time.sleep(1 / frame_rate)
        img_temp = camera.receive_image()
        img = np.copy(img_temp)
        camera.stop_acquisition()
        y = np.amax(img)
        print(f"The Theoretical Max Grey Value is {2**pixel_format_high_num - 1}. "
              f"Under Longest Exposure Time {camera.get_ExposureTime()} us, the Max Grey Value is {y}.\n")
        if EMVA1288_M1_utils.check_con_sat(img, img, y) == 1:
            # Over Saturated
            # adjust exposure time to meet 6.6 Saturation Condition
            t_exp_min = camera.get_ExposureTime_min()
            t_exp_max = camera.get_ExposureTime_max()
            t_exp_old = -1
            while True:
                t_exp = 0.5 * (t_exp_min + t_exp_max)
                camera.set_ExposureTime(t_exp)
                camera.set_AcquisitionMode("FreeRun")
                frame_rate = camera.get_FrameRate()
                camera.prepare_acquisition()
                camera.start_acquisition()
                # take 10 images and use the last 2
                for i in range(8):
                    time.sleep(1 / frame_rate)
                    camera.receive_image()
                time.sleep(1 / frame_rate)
                img_temp = camera.receive_image()
                img0 = np.copy(img_temp)
                time.sleep(1 / frame_rate)
                img_temp = camera.receive_image()
                img1 = np.copy(img_temp)
                camera.stop_acquisition()
                # if t_exp stuck at the correct value, then break with the correct value
                if t_exp_old == t_exp:
                    img_PIL = PIL.Image.fromarray(img0)
                    img_PIL.save(f"{PATH}/M1Data/Saturation0.tif")
                    img_PIL = PIL.Image.fromarray(img1)
                    img_PIL.save(f"{PATH}/M1Data/Saturation1.tif")
                    print(f"6.6 Saturation Condition Met")
                    print(f"Saturation Measurement Min {min(np.amin(img0), np.amin(img1))}")
                    print(f"Saturation Measurement Max {max(np.amax(img0), np.amax(img1))}")
                    break
                # update t_exp_old
                t_exp_old = t_exp
                if EMVA1288_M1_utils.check_con_sat(img0, img1, y) == -1:
                    # too few saturated pixel, need to increase exposure time
                    print(f"6.6 Saturation Condition Not Met, Adjusting Exposure Time **********")
                    t_exp_min = camera.get_ExposureTime()
                elif EMVA1288_M1_utils.check_con_sat(img0, img1, y) == 1:
                    # too many saturated pixel, need to decrease exposure time
                    print(f"6.6 Saturation Condition Not Met, Adjusting Exposure Time **********")
                    t_exp_max = camera.get_ExposureTime()
                else:
                    img_PIL = PIL.Image.fromarray(img0)
                    img_PIL.save(f"{PATH}/M1Data/Saturation0.tif")
                    img_PIL = PIL.Image.fromarray(img1)
                    img_PIL.save(f"{PATH}/M1Data/Saturation1.tif")
                    print(f"6.6 Saturation Condition Met")
                    print(f"Saturation Measurement Min {min(np.amin(img0), np.amin(img1))}")
                    print(f"Saturation Measurement Max {max(np.amax(img0), np.amax(img1))}")
                    break
            print(f"\nSaturation Measurement Finishes")
        elif EMVA1288_M1_utils.check_con_sat(img, img, y) == -1:
            # Under Saturated
            print(f"Seems Under Saturated Anyway")
        else:
            # longest exposure time leads to proper saturation
            print(f"6.6 Saturation Condition Met")
            print(f"Saturation Measurement Min {np.amin(img)}")
            print(f"Saturation Measurement Max {np.amax(img)}")
        t_exp_sat = camera.get_ExposureTime()
        mup_sat = EMVA1288_utils.calc_mup(A, E, t_exp_sat, WAVELENGTH)
        print(f"Saturation Exposure Time t_exp_sat is {t_exp_sat} us")
        print(f"The Corresponding mup_sat is {mup_sat}")
        print(f"\nDouble Check Pixel Format {camera.get_PixelFormat()}, Gain {camera.get_Gain()}, and Black Level {camera.get_BlackLevel()}\n")

        """Find t_exp Distribution, 6.5 Measurement Condition 5"""
        print(f"Finding t_exp Distribution **********")
        t_exp_min = camera.get_ExposureTime_min()
        t_exp_dist = EMVA1288_M1_utils.find_t_exp_vals(t_exp_min, t_exp_sat)

        """Measure at Each t_exp_dist"""
        print(f"\nMeasuring at Each of t_exp Distribution **********")
        H_dist = np.zeros_like(t_exp_dist)
        mup_dist = np.zeros_like(t_exp_dist)

        # Bright Measurements
        input(f"\nBright Measurement at Each Exposure Time, Unblock Light Source, Remove Camera Cap"
              f"\nPress Any Key to Continue")
        M, N = img.shape
        img0 = np.zeros((M, N))
        img1 = np.zeros((M, N))
        for idx, t_exp in enumerate(t_exp_dist):
            # set to the t_exp
            print(f"Updating Exposure Time **********")
            camera.set_ExposureTime(t_exp)
            # the real t_exp is set by camera, subject to rounding errors
            t_exp = camera.get_ExposureTime()
            t_exp_dist[idx] = t_exp
            # calculate H and mup
            H_dist[idx] = EMVA1288_utils.calc_H(E, t_exp)
            mup_dist[idx] = EMVA1288_utils.calc_mup(A, E, t_exp, WAVELENGTH)
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
            np.save(f"{PATH}/M1Data/Exp{idx}_b0.npy", img0)
            np.save(f"{PATH}/M1Data/Exp{idx}_b1.npy", img1)
            img_PIL = PIL.Image.fromarray(img0)
            img_PIL.save(f"{PATH}/M1Data/Exp{idx}_b0.tif")
            img_PIL = PIL.Image.fromarray(img1)
            img_PIL.save(f"{PATH}/M1Data/Exp{idx}_b1.tif")

        # Dark Measurements
        input(f"\nDark Measurement at Each Exposure Time, Block Light Source, Put On Camera Cap"
              f"\nPress Any Key to Continue")
        for idx, t_exp in enumerate(t_exp_dist):
            # set to the t_exp
            print(f"Updating Exposure Time **********")
            camera.set_ExposureTime(t_exp)
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
            np.save(f"{PATH}/M1Data/Exp{idx}_d0.npy", img0)
            np.save(f"{PATH}/M1Data/Exp{idx}_d1.npy", img1)
            img_PIL = PIL.Image.fromarray(img0)
            img_PIL.save(f"{PATH}/M1Data/Exp{idx}_d0.tif")
            img_PIL = PIL.Image.fromarray(img1)
            img_PIL.save(f"{PATH}/M1Data/Exp{idx}_d1.tif")

        """Save Measurement Variables"""
        np.savez(f"{PATH}/M1Data/measurement.npz",
                 E=E, A=A, WAVELENGTH=WAVELENGTH,
                 M=M, N=N,
                 pixel_format_high=pixel_format_high, pixel_format_high_num=pixel_format_high_num,
                 gain_a=gain_a, gain_d=gain_d, black_level=black_level,
                 t_exp_sat=t_exp_sat, mup_sat=mup_sat,
                 t_exp_dist=t_exp_dist, H_dist=H_dist, mup_dist=mup_dist)
    camera.close_camera()

