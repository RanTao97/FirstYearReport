import numpy as np

from SPHERE_R import SPHERE_R
from SPHERE_T import SPHERE_T
import constants

def correct_spheres(UR1, UT1, Rc, Tc, URU, UTU):
    """
    p.78: 169-172   iad_calc.c
    this is to do sphere corrections
    :param UR1: total reflectance under collimated illumination (- MC lost)
    :param UT1: total transmittance under collimated illumination (- MC lost)
    :param Rc: unscattered reflectance
    :param Tc: unscattered transmittance
    :param URU: total reflectance under diffuse illumination (- MC lost)
    :param UTU: total transmittance under diffuse illumination (- MC lost)
    :return: M_R_cor, M_T_cor: corrected M_R, M_T
    """
    sphere_r = SPHERE_R()
    sphere_t = SPHERE_T()

    R_diffuse = URU
    T_diffuse = UTU

    R_direct = UR1 - (1.0 - 1) * Rc  # fraction_of_rc_in_mr = 1 assumed
    T_direct = UT1 - (1.0 - 1) * Tc  # fraction_of_tc_in_mt = 1 assumed


    M_R_white = sphere_r.Two_Sphere_R(sphere_t,
                                      sphere_r.rstd_r, sphere_r.rstd_r,
                                      0, 0)

    #print(M_R_white)

    M_R_dark = sphere_r.Two_Sphere_R(sphere_t, 0, 0, 0, 0)

    M_T_white = sphere_t.Two_Sphere_T(sphere_r,
                                      0, 0,
                                      sphere_t.rstd_t, sphere_t.rstd_t)

    #print(M_T_white)

    M_T_dark = sphere_t.Two_Sphere_T(sphere_r, 0, 0, 0, 0)

    M_R_cor = sphere_r.rstd_r * ((sphere_r.Two_Sphere_R(sphere_t, R_direct, R_diffuse, T_direct, T_diffuse) - M_R_dark) /
                                 (M_R_white - M_R_dark))
    M_T_cor = sphere_t.rstd_t * ((sphere_t.Two_Sphere_T(sphere_r, R_direct, R_diffuse, T_direct, T_diffuse) - M_T_dark) /
                                 (M_T_white - M_T_dark))

    return M_R_cor, M_T_cor

