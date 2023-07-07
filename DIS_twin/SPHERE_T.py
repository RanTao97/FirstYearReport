import math

import constants

class SPHERE_T:
    def __init__(self):
        self.as_t = (constants.SAMPLE_PORT_DIAMETER_2 / constants.SPHERE_DIAMETER_2)**2 / 4  # sample aperture
        self.ae_t = (constants.ENTRANCE_PORT_DIAMETER_2 / constants.SPHERE_DIAMETER_2)**2 / 4  # entrance aperture
        self.ad_t = (constants.DETECTOR_PORT_DIAMETER_2 / constants.SPHERE_DIAMETER_2)**2 / 4  # detector aperture
        self.aw_t = 1 - self.as_t - self.ae_t - self.ad_t
        self.rw_t = constants.WALL_REFLECTANCE
        self.rd_t = constants.DETECTOR_REFLECTANCE
        self.rstd_t = 1
        # Fresnel Reflectance
        # Snell's Law in radius
        theta = constants.THETA / 180 * math.pi
        theta_p = math.asin((math.sin(theta) / constants.REFRACTIVE_INDEX_SAMPLE))
        self.f_t = 0  # fraction of light hit sphere wall before hitting the sample

    def Gain(self, URU):
        """
        p.59: 115-116   iad_calc.c
        :param URU: total transmittance under diffuse illumination
        :return: G
        """
        temp = self.rw_t * (self.aw_t + (1 - self.ae_t) * (self.ad_t * self.rd_t + self.as_t * URU))
        if temp == 1:
            G = 1
        else:
            G = 1 + temp / (1 - temp)

        return G

    def Gain_11(self, sphere_r, URU, UTU):
        """
        p.59: 117-118   iad_calc.c
        :param sphere_r: of class SPHERE_R
        :param URU: total reflectance under diffuse illumination
        :param UTU: total transmittance under diffuse illumination
        :return: G11
        """
        G = sphere_r.Gain(URU)
        GP = self.Gain(URU)

        G11 = G / (1 - sphere_r.as_r * self.as_t * sphere_r.aw_r * self.aw_t * (1 - sphere_r.ae_r) * (1 - self.ae_t) *
                   G * GP * UTU * UTU)

        return G11

    def Gain_22(self, sphere_r, URU, UTU):
        """
        p.59: 119-120   iad_calc.c
        :param sphere_r: of class SPHERE_R
        :param URU: total reflectance under diffuse illumination
        :param UTU: total transmittance under diffuse illumination
        :return: G22
        """
        G = sphere_r.Gain(URU)
        GP = self.Gain(URU)

        G22 = GP / (1 - sphere_r.as_r * self.as_t * sphere_r.aw_r * self.aw_t * (1 - sphere_r.ae_r) * (1 - self.ae_t) *
                    G * GP * UTU * UTU)

        return G22

    def Two_Sphere_T(self, sphere_r, UR1, URU, UT1, UTU):
        """
        p.61: 123-124   iad_calc.c
        :param sphere_r: of class SPHERE_R
        :param UR1: total reflectance under collimated illumination
        :param URU: total reflectance under diffuse illumination
        :param UT1: total transmittance under collimated illumination
        :param UTU: total transmittance under diffuse illumination
        :return:
        """
        G = sphere_r.Gain(URU)
        x = self.ad_t * (1 - self.ae_t) * self.rw_t * self.Gain_22(sphere_r, URU, UTU)
        x *= (1 - sphere_r.f_r) * UT1 + (1 - sphere_r.ae_r) * sphere_r.rw_r * sphere_r.as_r * UTU * \
             (sphere_r.f_r * sphere_r.rw_r + (1 - sphere_r.f_r) * UR1) * G
        return x