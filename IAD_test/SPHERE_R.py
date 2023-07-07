import math

import constants

class SPHERE_R:
    def __init__(self):
        self.as_r = (constants.SAMPLE_PORT_DIAMETER / constants.SPHERE_DIAMETER)**2 / 4  # sample aperture
        self.ae_r = (constants.ENTRANCE_PORT_DIAMETER / constants.SPHERE_DIAMETER)**2 / 4  # entrance aperture
        self.ad_r = (constants.DETECTOR_PORT_DIAMETER / constants.SPHERE_DIAMETER) ** 2 / 4  # detector aperture
        self.aw_r = 1 - self.as_r - self.ae_r - self.ad_r
        self.rw_r = constants.WALL_REFLECTANCE
        self.rd_r = constants.DETECTOR_REFLECTANCE
        self.rstd_r = constants.STANDARD_REFLECTANCE
        # Fresnel Reflectance
        # Snell's Law in radius
        theta = constants.THETA / 180 * math.pi
        theta_p = math.asin((math.sin(theta) / constants.REFRACTIVE_INDEX_SAMPLE))
        self.f_r = 0  # fraction of light hit sphere wall before hitting the sample

    def Gain(self, URU):
        """
        p.59: 115-116   iad_calc.c
        :param URU: total transmittance under diffuse illumination
        :return: G
        """
        temp = self.rw_r * (self.aw_r + (1 - self.ae_r) * (self.ad_r * self.rd_r + self.as_r * URU))
        if temp == 1:
            G = 1
        else:
            G = 1 + temp / (1 - temp)
        return G

    def Gain_11(self, sphere_t, URU, UTU):
        """
        p.59: 117-118   iad_calc.c
        :param sphere_t: of class SPHERE_T
        :param URU: total reflectance under diffuse illumination
        :param UTU: total transmittance under diffuse illumination
        :return: G11
        """
        G = self.Gain(URU)
        GP = sphere_t.Gain(URU)

        G11 = G / (1 - self.as_r * sphere_t.as_t * self.aw_r * sphere_t.aw_t * (1 - self.ae_r) * (1 - sphere_t.ae_t) *
                   G * GP * UTU * UTU)

        return G11

    def Gain_22(self, sphere_t, URU, UTU):
        """
        p.59: 119-120   iad_calc.c
        :param sphere_t: of class SPHERE_T
        :param URU: total reflectance under diffuse illumination
        :param UTU: total transmittance under diffuse illumination
        :return: G22
        """
        G = self.Gain(URU)
        GP = sphere_t.Gain(URU)

        G22 = GP / (1 - self.as_r * sphere_t.as_t * self.aw_r * sphere_t.aw_t * (1 - self.ae_r) * (1 - sphere_t.ae_t) *
                    G * GP * UTU * UTU)

        return G22

    def Two_Sphere_R(self, sphere_t, UR1, URU, UT1, UTU):
        """
        p.61: 121-122   iad_calc.c
        :param sphere_t: of class SPHERE_T
        :param UR1: total reflectance under collimated illumination
        :param URU: total reflectance under diffuse illumination
        :param UT1: total transmittance under collimated illumination
        :param UTU: total transmittance under diffuse illumination
        :return:
        """
        GP = sphere_t.Gain(URU)
        x = self.ad_r * (1 - self.ae_r) * self.rw_r * self.Gain_11(sphere_t, URU, UTU)
        x *= (1 - self.f_r) * UR1 + self.rw_r * self.f_r + (1 - self.f_r) * sphere_t.as_t * (1 - sphere_t.ae_t) *\
             sphere_t.rw_t * UT1 * UTU * GP
        return x

