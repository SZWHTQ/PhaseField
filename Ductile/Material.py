import numpy as np

class Material:
    def __init__(self) -> None:
        pass


class Ductile(Material):
    def __init__(self):
        ## Units: mm, s, kg, MPa
        # Mechanical properties
        self.rho = 7.8e-6  # Mass Density, kg/mm^3
        self.lame = 101163.333333333  # Lamé coefficient, MPa; bulk modulus = 150000 MPa
        self.mu = 73255  # Shear modulus, MPa

        # Crack phase properties
        self.Gc0 = 1000  # Initial critical fracture energy, KJ/m^2 (kg/s^2)
        self.Gc_inf = 142.5  # Reduced critical fracture energy, KJ/m^2
        self.omega_f = 42.325  # Saturation exponent (Fracture), -
        self.eta_f = 6e-6  # Fracture viscosity parameter, MPa s
        self.lf = 0.78125  # Fracture length scale, mm
        self.wc = 180  # Critical work density, MPa

        # Plastic phase properties
        self.y0 = 343  # Initial yield stress, MPa
        self.y_inf = 680  # Ultimate yield stress, MPa
        self.omega_p = 16.93  # Saturation exponent (Plastic), -
        self.h = 300  # Hardening modulus, MPa
        self.eta_p = 6e-6  # Plastic viscosity parameter, MPa s
        self.lp = 0.78125  # Plastic length scale, mm

    def hardening(self, equivalent_plastic_strain):
        return (
            self.y_inf
            - (self.y_inf - self.y0)
            * np.exp(-self.omega_p * equivalent_plastic_strain)
            + self.h * equivalent_plastic_strain
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
