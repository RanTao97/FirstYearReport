from arena_api.system import system
import ctypes
import numpy as np
import re


class Camera:
    def __init__(self):
        # property flags
        self.auto_correction_flag = None
        self.bining_flag = None
        # properties: exposure_time, bit_depth, gain_a, gain_d, black_level, mode
        # LUCID specific
        self.device = None
        self.nodemap = None
        self.tl_stream_nodemap = None
        # acquisition flags
        self.acquisition_running = False

    def open_camera(self):
        """
        this is to open the camera
        :return: 0 if successful
        """
        devices = system.create_device()
        if not devices:
            print(f"No device found.")
        else:
            print(f"Find {len(devices)} device(s).")
            # list available devices
            for i_device in range(len(devices)):
                print(f"{system.device_infos[i_device]}")
            # ask user to select a device
            selected_device = None
            while True:
                selected_device = int(input("Select device to open: "))
                if selected_device in range(len(devices)):
                    break
                else:
                    print("Invalid ID. Try Again.")
            # open selected device
            self.device = devices[selected_device]
            # get the nodemaps
            self.nodemap = self.device.nodemap
            self.tl_stream_nodemap = self.device.tl_stream_nodemap
            # print the info of selected device
            print(f"The Selected Device {selected_device} is {self.device}")
            # print resolution
            print(f"Max. resolution (w x h): {self.nodemap['SensorWidth'].value} x {self.nodemap['SensorHeight'].value}")
            # BEFORE STARTING THE STREAM
            # ensure the most recent image is delivered, even if it means skipping frames
            self.tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
            # enable stream auto negotiate packet size
            # increases frame rate and results in fewer interrupts per image
            # thereby reducing CPU load on the host system
            self.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
            # enable stream packet resend
            # to retrieve and redeliver the missing packet in the correct order
            self.tl_stream_nodemap['StreamPacketResendEnable'].value = True
            return 0

    def close_camera(self):
        """
        this is to close the camera
        :return: 0
        """
        # double check to stop acquisition before closing the device
        self.stop_acquisition()
        # destroy device.
        # this call is optional
        # and will automatically be called for any remaining devices when the system module is unloading
        system.destroy_device()
        return 0

    def disable_auto(self):
        """
        this is to disable camera auto functions
        :return: 0 if successful
        """
        self.turn_off_GainAuto()
        self.turn_off_ExposureAuto()

        self.turn_off_GammaCorrection()
        # color corrections are not implemented for mono camera
        # self.turn_off_BalanceWhiteAuto()
        # self.turn_off_ColorTransformation()

        self.turn_off_DefectCorrection()
        self.turn_off_BlackLevelCorrection()

        # set ROI as original ROI
        self.set_ROI()
        # turn off Binning, Decimation, Mirror
        self.turn_off_Binning()
        self.turn_off_Decimation()
        self.turn_off_Mirror()

        # no autocorrection now
        # no binning now
        self.auto_correction_flag = False
        self.bining_flag = False

        return 0

# Auto Corrections -----------------------------------------------------------------------------------------------------
    def turn_off_GainAuto(self):
        """
        this is to turn off GainAuto
        :return: 0 if successful
        """
        # GainAuto has Entries: ['Off', 'Once', 'Continuous']
        # turn off GainAuto
        self.nodemap['GainAuto'].value = 'Off'
        return 0

    def turn_off_ExposureAuto(self):
        """
        this is to turn off ExposureAuto
        :return: 0 if successful
        """
        # ExposureAuto has Entries: ['Off', 'Once', 'Continuous']
        self.nodemap['ExposureAuto'].value = 'Off'
        # also allow AcquisitionFrameRate feature is writable and used to control the acquisition rate
        # as a set frame rate limits the max exposure time
        self.nodemap['AcquisitionFrameRateEnable'].value = True
        return 0

    def turn_off_BalanceWhiteAuto(self):
        """
        this is to turn off the white balance
        BalanceWhiteAuto not implemented in Arena
        BalanceWhiteEnable not implemented in Arena
        makes sense for mono camera
        :return: 0 if successful
        """
        # not implemented
        return -1

    def turn_off_GammaCorrection(self):
        """
        this is to turn off Gamma correction
        :return: 0 if successful
        """
        # GammaEnable is BOOLEAN: true or false
        self.nodemap['GammaEnable'].value = True
        # Gamma is FLOAT, has value 0.2-2, set to 1.0 for unchanged pixel intensity
        self.nodemap['Gamma'].value = 1.0
        return 0

    def turn_off_ColorTransformation(self):
        """
        this is to turn off color transformation
        ColorTransformationEnable not implemented in Arena
        makes sense for mono camera
        :return: 0 if successful
        """
        # not implemented
        return -1

    def turn_off_DefectCorrection(self):
        """
        this is to turn off Defect Pixel Correction
        :return: 0 if successful
        """
        # DefectCorrectionEnable is BOOLEAN
        # Todo: always double check this depending on application
        self.nodemap['DefectCorrectionEnable'].value = False
        return 0

    def turn_off_BlackLevelCorrection(self):
        """
        this is to turn off Black Level Correction, i.e., Dark Current Compensation
        :return: 0 if successful
        """
        # BlackLevelCorrectionEnable is BOOLEAN
        self.nodemap['BlackLevelCorrectionEnable'].value = False
        return 0

    def set_ROI(self):
        """
        this is to set ROI to the entire sensor
        :return: 0 if successful
        """
        # Width is INTEGER: 4-1280, increment = 4
        self.nodemap['Width'].value = self.nodemap['Width'].max
        # Height is INTEGER: 2-1024, increment = 2
        self.nodemap['Height'].value = self.nodemap['Height'].max
        # OffsetX is INTEGER: always 0
        self.nodemap['OffsetX'].value = 0
        # OffsetY is INTEGER: always 0
        self.nodemap['OffsetY'].value = 0
        return 0

    def turn_off_Binning(self):
        """
        this is to turn off Binning
        :return: 0 if successful
        """
        # BinningHorizontal is INTEGER: 1-8, increment = 1
        self.nodemap['BinningHorizontal'].value = 1
        # BinningVertical is INTEGER: 1-8, increment = 1
        self.nodemap['BinningVertical'].value = 1
        return 0

    def turn_off_Decimation(self):
        """
        this is to turn off Decimation
        :return: 0 if successful
        """
        # DecimationHorizontal is INTEGER: 1-2, increment = 1
        self.nodemap['DecimationHorizontal'].value = 1
        # DecimationVertical is INTEGER: 1-2, increment = 1
        self.nodemap['DecimationVertical'].value = 1
        return 0

    def turn_off_Mirror(self):
        """
        this is to turn off Mirror
        :return: 0 if successful
        """
        # ReverseX is BOOLEAN
        self.nodemap['ReverseX'].value = False
        # ReverseY is BOOLEAN
        self.nodemap['ReverseY'].value = False
        return 0

# Bit Depth-------------------------------------------------------------------------------------------------------------
    def get_PixelFormat(self):
        """
        this is to get the current pixel format, i.e., bit depth
        :return: pixel_format, string
        """
        return self.nodemap['PixelFormat'].value

    def get_PixelFormat_list(self):
        """
        this is to get a list of available pixel format
        :return: a string list of available pixel format
        """
        return self.nodemap['PixelFormat'].enumentry_names

    def set_PixelFormat(self, _pixel_format):
        """
        this is to read and set camera PixelFormat
        :param _pixel_format:
        :return: 0 if successful
        """
        print(f"Always Set to Max ADC Bit Depth")
        self.set_ADCBitDepth_max()
        print(f"Current Pixel Format {self.get_PixelFormat()}")
        if _pixel_format in self.get_PixelFormat_list():
            self.nodemap['PixelFormat'].value = _pixel_format
            print(f"Update Pixel Format {self.get_PixelFormat()}")
        else:
            print(f"No Available Pixel Format, Update to {self.get_PixelFormat()} Instead")
        return 0

    def set_ADCBitDepth_max(self):
        """
        this is to set ADCBitDepth to the max value
        :return: 0 if successful
        """
        self.nodemap['ADCBitDepth'].value = 'Bits12'
        return 0

# Gain------------------------------------------------------------------------------------------------------------------
    def get_Gain_a(self):
        """
        this is to get current analog gain
        LUCID Atlas Specific:
        Some cameras feature gain that is purely digital
        while others allow for analog gain control up to a certain value,
        beyond which the gain becomes digital.
        Depending on the camera family and sensor model, the specific gain control can vary.
        :return: gain_a [dB]
        """
        self.nodemap['GainSelector'].value = 'All'
        return self.nodemap['Gain'].value

    def get_Gain_a_min(self):
        """
        this is to get the min available analog gain
        LUCID Atlas Specific:
        Some cameras feature gain that is purely digital
        while others allow for analog gain control up to a certain value,
        beyond which the gain becomes digital.
        Depending on the camera family and sensor model, the specific gain control can vary.
        :return: gain_a_min [dB]
        """
        self.nodemap['GainSelector'].value = 'All'
        return self.nodemap['Gain'].min

    def get_Gain_d(self):
        """
        this is to get current digital gain
        LUCID Atlas Specific:
        Some cameras feature gain that is purely digital
        while others allow for analog gain control up to a certain value,
        beyond which the gain becomes digital.
        Depending on the camera family and sensor model, the specific gain control can vary.
        :return: gain_d [dB]
        """
        self.nodemap['GainSelector'].value = 'All'
        return self.nodemap['Gain'].value

    def get_Gain_d_min(self):
        """
        this is to get the min available digital gain
        LUCID Atlas Specific:
        Some cameras feature gain that is purely digital
        while others allow for analog gain control up to a certain value,
        beyond which the gain becomes digital.
        Depending on the camera family and sensor model, the specific gain control can vary.
        :return: gain_d_min [dB]
        """
        self.nodemap['GainSelector'].value = 'All'
        return self.nodemap['Gain'].min

    def get_Gain(self):
        """
        this is to get current analog gain and digital gain
        :return: (gain_a, gain_d)
        """
        return self.get_Gain_a(), self.get_Gain_d()

    def set_Gain(self, _gain, dummy):
        """
        this is to set gain
        :param _gain: user-input gain [dB]
        :param dummy: dummy variable for "digital gain", because LUCID Atlas integrates gain in 'All'
        :return: 0 if successful
        """
        self.nodemap['GainSelector'].value = 'All'
        # get current gain
        print(f"Current Gain {self.nodemap['Gain'].value} dB")
        # Gain is FLOAT: 0-42 dB
        _gain = float(_gain)
        if (_gain >= self.nodemap['Gain'].min) and (_gain <= self.nodemap['Gain'].max):
            self.nodemap['Gain'].value = _gain
        else:
            print(f"Invalid Input Gain, Setting to the Closest Possible Value")
            self.nodemap['Gain'].value = min(self.nodemap['Gain'].max, max(_gain, self.nodemap['Gain'].min))
        print(f"Update Gain {self.nodemap['Gain'].value} dB")

        return 0

# Black Level-----------------------------------------------------------------------------------------------------------
    # N.B. Always set the pixel format before get/set black level
    # black level depends on the pixel format
    def get_BlackLevel(self):
        """
        this is to get the current black level
        :return: black_level [DN]
        """
        # LUCID Atlas specific:
        # BlackLevel is percentage
        black_level_percentage = self.nodemap['BlackLevel'].value
        pixel_format = self.get_PixelFormat()
        bit_depth = int(re.findall(r'\d+', pixel_format)[0])
        black_level = black_level_percentage / 100 * (2**bit_depth - 1)
        return black_level

    def set_BlackLevel(self, _black_level):
        """
        this is to set black level
        :param _black_level: user-input black level [%] or [DN]
        :return: 0 if successful
        """
        print(f"Current Black Level {self.get_BlackLevel()}")
        # BlackLevel is FLOAT: percentage
        # set to the closest valid value
        _black_level = float(_black_level)
        if _black_level <= 12.0:  # percentage [%]
            self.nodemap['BlackLevel'].value = min(self.nodemap['BlackLevel'].max, max(_black_level, self.nodemap['BlackLevel'].min))
        else:  # DN [DN]
            pixel_format = self.get_PixelFormat()
            bit_depth = int(re.findall(r'\d+', pixel_format)[0])
            _black_level = _black_level / (2**bit_depth - 1) * 100
            self.nodemap['BlackLevel'].value = min(self.nodemap['BlackLevel'].max, max(_black_level, self.nodemap['BlackLevel'].min))
        print(f"Update Black Level {self.get_BlackLevel()}")
        return 0

# Exposure Time---------------------------------------------------------------------------------------------------------
    def get_ExposureTime(self):
        """
        this is to query camera current exposure time
        :return: t_exp [us]
        """
        # ExposureTimeSelector is always 'Common'
        return self.nodemap['ExposureTime'].value

    def get_ExposureTime_max(self):
        """
        this is to query camera max available exposure time
        :return: t_exp_max [us]
        """
        # ExposureTimeSelector is always 'Common'
        # LUCID Atlas camera specific:
        # max exposure time is limited by the frame rate
        # to get the max exposure time, need to set to the lowest frame rate first
        frame_rate_current = self.get_FrameRate()  # record the current frame rate first
        self.set_FrameRate_min()  # set to the lowest frame rate
        exposure_time_max = self.nodemap['ExposureTime'].max  # max exposure time at min frame rate
        self.set_FrameRate(frame_rate_current)  # set back to current frame rate
        # Todo: when gain = 18 dB, noise is amplified, and too long exposure time leads to very large noise
        # Todo: so only return 1/8*exposure_time_max ~1.25s
        # Todo: BW
        return 0.5*exposure_time_max

    def get_ExposureTime_min(self):
        """
        this is to query camera min available exposure time
        :return: t_exp_min [us]
        """
        # ExposureTimeSelector is always 'Common'
        return self.nodemap['ExposureTime'].min

    # def set_ExposureTime(self, _exposure_time):
    #     """
    #     this is to set camera exposure time [us]
    #     :param _exposure_time: user input exposure time [us]
    #     :return: 0 if successful
    #     """
    #     # get the current exposure time and the corresponding frame rate
    #     print(f"Current Exposure Time {self.get_ExposureTime()} us")
    #     print(f"Current Corresponding Frame Rate {self.get_FrameRate()} Hz")
    #     # LUCID Atlas camera specific:
    #     # has free frame rate, and the exposure time depends on frame rate
    #     _exposure_time = float(_exposure_time)  # ExposureTime is float
    #     frame_rate = (1 / (_exposure_time * 1e-6))
    #     # case 1: short exposure timef
    #     if _exposure_time < ((1 / self.get_FrameRate_max()) * 1e6):
    #         # case 1.1: invalid short exposure time
    #         if _exposure_time < self.get_ExposureTime_min():
    #             # set to the min possible short exposure time
    #             self.set_FrameRate_max()
    #             self.nodemap['ExposureTime'].value = self.nodemap['ExposureTime'].min
    #             print(f"No Available Exposure Time, Update to {self.get_ExposureTime()} us Instead")
    #         # case 1.2: valid short exposure time, but frame rate is limited by camera
    #         else:
    #             self.set_FrameRate_max()
    #             # set to the closest valid value
    #             self.nodemap['ExposureTime'].value = min(_exposure_time, self.nodemap['ExposureTime'].max)
    #             print(f"Update Exposure Time {self.get_ExposureTime()} us")
    #     # case 2: too long exposure time, even beyond frame rate restrict, must be invalid
    #     elif _exposure_time > ((1 / self.get_FrameRate_min()) * 1e6):
    #         # set to the max possible long exposure time
    #         self.set_FrameRate_min()
    #         self.nodemap['ExposureTime'].value = self.nodemap['ExposureTime'].max
    #         print(f"No Available Exposure Time, Update to {self.get_ExposureTime()} us Instead")
    #     # case 3: valid exposure time leads to valid frame rate
    #     else:
    #         # set to the corresponding frame rate
    #         # at that frame rate, the exposure time is slightly shorter due to processing time
    #         self.set_FrameRate(frame_rate)
    #         self.nodemap['ExposureTime'].value = self.nodemap['ExposureTime'].max
    #         print(f"Update Exposure Time {self.get_ExposureTime()} us")
    #     print(f"Update Corresponding Frame Rate {self.get_FrameRate()} Hz")
    #     return 0

    def set_ExposureTime(self, _exposure_time):
        """
        this is to set camera exposure time [us]
        :param _exposure_time: user input exposure time [us]
        :return: 0 if successful
        """
        # get the current exposure time and the corresponding frame rate
        print(f"Current Exposure Time {self.get_ExposureTime()} us")
        print(f"Current Corresponding Frame Rate {self.get_FrameRate()} Hz")
        # LUCID Atlas camera specific:
        # has free frame rate, and the exposure time depends on frame rate
        _exposure_time = float(_exposure_time)  # ExposureTime is float
        if _exposure_time <= 0:
            print(f"Non-positive Exposure Time, Set to the Min Valid Value")
            self.set_FrameRate_max()
            self.nodemap['ExposureTime'].value = self.nodemap['ExposureTime'].min
            print(f"Update Exposure Time {self.get_ExposureTime()} us")
            print(f"Update Corresponding Frame Rate {self.get_FrameRate()} Hz")
            return 0
        # Todo: now manually add process time ~ 210 us (2023/05/28)
        frame_rate = 1 / ((_exposure_time + 210) * 1e-6)  # usually need to set frame_rate lower to account for processing time, see below
        while True:
            if frame_rate >= self.get_FrameRate_min():  # valid frame rate
                # set to the closest valid value
                self.set_FrameRate(min(frame_rate, self.get_FrameRate_max()))
                if (_exposure_time <= self.nodemap['ExposureTime'].max) and (
                        _exposure_time >= self.nodemap['ExposureTime'].min):
                    self.nodemap['ExposureTime'].value = _exposure_time
                    break
                elif _exposure_time > self.nodemap['ExposureTime'].max:
                    # need to lower frame rate to allow a longer exposure time
                    # Todo: now manually add process time ~ 210 us (2023/05/28)
                    # by adding 1 x process time
                    frame_rate = 1 / (1 / frame_rate + 210 * 1e-6)
                else:
                    print(f"Too Short Exposure Time, Set to the Min Valid Value")
                    self.set_FrameRate_max()
                    self.nodemap['ExposureTime'].value = self.nodemap['ExposureTime'].min
                    break
            else:
                print(f"Too Long Exposure Time, Set to the Max Valid Value")
                self.set_FrameRate_min()
                self.nodemap['ExposureTime'].value = self.nodemap['ExposureTime'].max
                break
        print(f"Update Exposure Time {self.get_ExposureTime()} us")
        print(f"Update Corresponding Frame Rate {self.get_FrameRate()} Hz")
        return 0

# Frame Rate (Controls Exposure Time Range)-----------------------------------------------------------------------------
    def get_FrameRate(self):
        """
        this is to get the current frame rate [Hz]
        :return: frame_rate
        """
        return self.nodemap['AcquisitionFrameRate'].value

    def get_FrameRate_min(self):
        """
        this is to get the min frame rate [Hz]
        :return: frame_rate
        """
        return self.nodemap['AcquisitionFrameRate'].min

    def get_FrameRate_max(self):
        """
        this is to get the max frame rate [Hz]
        :return: frame_rate
        """
        return self.nodemap['AcquisitionFrameRate'].max

    def set_FrameRate_min(self):
        """
        this is to set the frame rate to the min available value
        :return: 0 if successful
        """
        self.nodemap['AcquisitionFrameRate'].value = self.nodemap['AcquisitionFrameRate'].min
        return 0

    def set_FrameRate_max(self):
        """
        this is to set the frame rate to the max available value
        :return: 0 if successful
        """
        self.nodemap['AcquisitionFrameRate'].value = self.nodemap['AcquisitionFrameRate'].max
        return 0

    def set_FrameRate(self, _frame_rate):
        """
        this is to set the frame rate to the input value
        :param _frame_rate: user input frame rate
        :return: 0 if successful
        """
        if (_frame_rate >= self.get_FrameRate_min()) and (_frame_rate <= self.get_FrameRate_max()):
            self.nodemap['AcquisitionFrameRate'].value = _frame_rate
        else:
            # set to max frame rate if the input frame rate is invalid
            self.set_FrameRate_max()
            print(f"No Available Frame Rate, Update to {self.get_FrameRate()} Hz Instead")
            print(f"Exposure Time is Correspondingly Updated to {self.get_ExposureTime()} us")
        return 0

# Acquisition Mode------------------------------------------------------------------------------------------------------
    def get_AcquisitionMode(self):
        """
        this is to get AcquisitionMode
        :return: mode
        """
        return self.nodemap['AcquisitionMode'].value

    def set_AcquisitionMode(self, _mode):
        """
        this is to set camera AcquisitionMode
        :param _mode: user-input acquisition mode, str
        :return: 0 if successful
        """
        # always in Normal mode
        self.nodemap['AcquisitionStartMode'].value = 'Normal'
        # single frame mode
        if _mode == "SoftwareTrigger":
            self.nodemap['AcquisitionMode'].value = 'SingleFrame'
            return 0
        elif _mode == "FreeRun":
            self.nodemap['AcquisitionMode'].value = 'Continuous'
            return 0
        else:
            print("Acquisition Mode Not Implemented")
            return -1

# Image Acquisition-----------------------------------------------------------------------------------------------------
    def prepare_acquisition(self):
        """
        this is to prepare image acquisition with buffer
        :return: 0 if successful
        """
        return 0

    def start_acquisition(self):
        """
        this is to start image acquisition
        :return: 0 if successful
        """
        if self.acquisition_running is True:
            # already in acquisition mode
            # need to do nothing
            return 0
        self.device.start_stream()
        self.acquisition_running = True
        return 0

    def stop_acquisition(self):
        """
        this is to stop image acquisition
        :return: 0 if successful
        """
        if (self.device is None) or (self.acquisition_running is False):
            # need to do nothing
            return 0
        self.device.stop_stream()
        self.acquisition_running = False
        return 0

    def receive_image(self):
        """
        this is to receive image from the camera
        called between start_acquisition and stop_acquisition
        :return: image_np (numpy float array image) if successful
        """
        # retrieve 1 buffer with timeout = 10000 ms
        image_buffer = self.device.get_buffer()

        if (self.get_PixelFormat() == "Mono16") or (self.get_PixelFormat() == "Mono12"):
            pdata_as16 = ctypes.cast(image_buffer.pdata, ctypes.POINTER(ctypes.c_ushort))
            image_np = np.ctypeslib.as_array(pdata_as16, (image_buffer.height, image_buffer.width))

            # always requeue buffer for next image
            self.device.requeue_buffer(image_buffer)
            return image_np.astype(float)
        else:
            # always requeue buffer for next image
            self.device.requeue_buffer(image_buffer)
            print(f"Not Implemented")
            return -1

# TEC Temperature Control-----------------------------------------------------------------------------------------------
    def get_DeviceTemperature(self):
        """
        this is to get device temperature
        :return: T_sensor, T_TEC
        """
        self.nodemap['DeviceTemperatureSelector'].value = 'Sensor'
        T_sensor = self.nodemap['DeviceTemperature'].value
        self.nodemap['DeviceTemperatureSelector'].value = 'TEC'
        T_TEC = self.nodemap['DeviceTemperature'].value
        return T_sensor, T_TEC

    def set_TECControlTemperature(self, _temperature):
        """
        this is to set TEC control temperature
        :param _temperature: set temperature
        :return: 0 if successful
        """
        # TECControlTemperatureSetPoint is FLOAT
        _temperature = float(_temperature)
        if _temperature < self.nodemap['TECControlTemperatureSetPoint'].min:
            print(f'Too Low Temperature, Setting to Min Available Value')
            self.nodemap['TECControlTemperatureSetPoint'].value = self.nodemap['TECControlTemperatureSetPoint'].min
        elif _temperature > self.nodemap['TECControlTemperatureSetPoint'].max:
            print(f'Too High Temperature, Setting to Max Available Value')
            self.nodemap['TECControlTemperatureSetPoint'].value = self.nodemap['TECControlTemperatureSetPoint'].max
        else:
            print(f'Current TEC Control Temperature is {self.get_TECControlTemperature()} deg C')
            self.nodemap['TECControlTemperatureSetPoint'].value = _temperature
        print(f'Update TEC Control Temperature {self.get_TECControlTemperature()} deg C')

    def get_TECControlTemperature(self):
        """
        this is to get TEC control temperature
        :return: TECControlTemperature
        """
        return self.nodemap['TECControlTemperatureSetPoint'].value
