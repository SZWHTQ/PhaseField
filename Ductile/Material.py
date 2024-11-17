import numpy as np
import ufl
import dolfinx as dfx


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
        # %% Units: mm, s, kg
        # TI-6Al-4V, From Prof. Zeng's JMPS, Liao and Duffy (1998) and Zhu et al. (2019)
        self.rho = 4.43e-6  # kg/mm^3
        self.specific_heat = 5.6e8  # mm^2/s^2 # Check its unit again, 560 J/kg K
        self.heat_conduction = 6.6e3  # kg mm / s K  # Name may be incorrect, W/m K
        self.inelastic_heat = 0.9

        # Mechanical properties
        # E = 110GPa, nu = 0.31
        self.lame = 68501406.18722378  # kg / mm / s^2
        self.mu = 41984732.82442748  # kg / mm / s^2

        # Johnson-Cook parameters
        self.A = 1.098e6  # kg / mm / s^2
        self.B = 1.092e6  # kg / mm / s^2
        self.C = 0.014
        self.n = 0.93
        self.m = 1.10
        self.reference_strain_rate = 1.0
        self.reference_temperature = 298.0
        self.melting_temperature = 1878.0

        # Crack phase properties
        self.w0 = 1.5e4  # kg / mm / s^2
        self.Gc = 3.0e4  # kg / s^2
        self.lf = 2e-2  # mm
        self.eta_f = 5e-2  # kg / mm / s
        # self.beta = 2.8 # Useless for now

    def getYieldStress(
        self, damage, equivalent_plastic_strain, strain_rate=1.0, temperature=298.0
    ) -> dfx.fem.Function:
        # if strain_rate is None:
        #     strain_rate = 1
        # if temperature is None:
        #     temperature = self.reference_temperature

        return (1 - damage) ** 2 * (
            (self.A + self.B * equivalent_plastic_strain**self.n)
            * (1 + self.C * ufl.ln(ufl.max_value(strain_rate, 1.0)))
            * (
                1
                - (temperature - self.reference_temperature)
                / (self.melting_temperature - self.reference_temperature)
            )
        )

    def getHardeningModulus(
        self,
        damage,
        equivalent_plastic_strain,
        strain_rate=1.0,
        temperature=298.0,  # noqa
    ) -> dfx.fem.Function:
        # if strain_rate is None:
        #     strain_rate = 1
        # if temperature is None:
        #     temperature = self.reference_temperature

        return ufl.conditional(
            ufl.gt(equivalent_plastic_strain, 0),
            (1 - damage) ** 2
            * (
                (self.n * self.B * equivalent_plastic_strain ** (self.n - 1))
                * (1 + self.C * ufl.ln(ufl.max_value(strain_rate, 1.0)))
                * (
                    1
                    - (temperature - self.reference_temperature)
                    / (self.melting_temperature - self.reference_temperature)
                )
            ),
            0,
        )
