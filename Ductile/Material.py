import numpy as np
import ufl


class Material:
    def __init__(self) -> None:
        pass


class Ductile(Material):
    def __init__(self):
        ## Units: mm, s, kg
        # Mechanical properties
        self.rho = 7.8e-6  # Mass Density, kg/mm^3
        self.lame = 101163.333333333e3  # Lamé coefficient, kg mm / s^2 / mm^2; bulk modulus = 150000 MPa
        self.mu = 73255e3  # Shear modulus, kg mm / s^2 / mm^2

        # Crack phase properties
        # self.Gc0 = 1000  # Initial critical fracture energy, KJ/m^2 (kg/s^2)
        # self.Gc_inf = 142.5  # Reduced critical fracture energy, KJ/m^2
        # self.omega_f = 42.325  # Saturation exponent (Fracture), -
        self.eta_f = 6e-3  # Fracture viscosity parameter, kg mm / s^2 / mm^2 s
        self.lf = 0.78125  # Fracture length scale, mm
        self.wc = 180e3  # Critical work density, kg mm / s^2 / mm^2
        self.zeta = 1  # Fracture parameter, -
        # this is a parameter determining the slope of the post-critical range of fracture
        # with \zeta < 1 for a convex and \zeta > 1 for a non-convex resistance function \hat{D}^{pf}

        # Plastic phase properties
        self.y0 = 343e3  # Initial yield stress, kg mm / s^2 / mm^2
        self.y_inf = 680e3  # Ultimate yield stress, kg mm / s^2 / mm^2
        self.omega_p = 16.93  # Saturation exponent (Plastic), -
        self.h = 300e3  # Hardening modulus, kg mm / s^2 / mm^2
        self.eta_p = 6e-3  # Plastic viscosity parameter, kg mm / s^2 / mm^2 s
        self.lp = 0.78125  # Plastic length scale, mm

    def hardening(self, p):
        return (
            self.y0
            + (self.y_inf - self.y0) * (1 - ufl.exp(-self.omega_p * p))
            + self.h * p
        )


# Johnson-Cook parameters
class JohnsonCook:
    def __init__(self):
        self.mass_density = 4.43e-6
        self.specific_heat = 5.6e8  # Check its unit again, 560 J/kg K
        self.heat_conduction = 6.6  # Name may be incorrect, W/m K
        self.inelastic_heat = 0.9

        # Mechanical properties
        # E = 110GPa, nu = 0.31
        self.lame = 68501.40618722378
        self.shear_modulus = 41984.732824427476

        # Johnson-Cook parameters
        self.A = 200.0
        self.B = 1072.0
        self.n = 0.34
        self.C = 0.05
        self.m = 1.1
        self.reference_temperature = 298.0
        self.melting_temperature = 1878.0
        self.reference_strain_rate = 1.0

    def getYieldStress(
        self, equivalent_plastic_strain, strain_rate, temperature, damage
    ):
        return (
            (self.A + self.B * max(equivalent_plastic_strain, 1e-6) ** self.n)
            * (1 + self.C * np.log(max(strain_rate, 1)))
            * (
                1
                - (temperature - self.reference_temperature)
                / (self.melting_temperature - self.reference_temperature)
            )
        ) * (1 - damage) ** 2

    def getHardeningModulus(
        self, equivalent_plastic_strain, strain_rate, temperature, damage
    ):
        # if equivalent_plastic_strain > 0:
        #     return (
        #         (self.n * self.B * equivalent_plastic_strain ** (self.n - 1))
        #         * (1 + self.C * np.log(max(strain_rate, 1)))
        #         * (
        #             1
        #             - (
        #                 (temperature - self.reference_temperature)
        #                 / (self.melting_temperature - self.reference_temperature)
        #             )
        #             ** self.m
        #         )
        #     ) * (1 - damage) ** 2
        # else:
        #     return 0
        return (
            (self.n * self.B * max(equivalent_plastic_strain, 1e-6) ** (self.n - 1))
            * (1 + self.C * np.log(max(strain_rate, 1)))
            * (
                1
                - (temperature - self.reference_temperature)
                / (self.melting_temperature - self.reference_temperature)
            )
        ) * (1 - damage) ** 2
