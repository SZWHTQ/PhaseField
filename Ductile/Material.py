import numpy as np


class Material:
    def __init__(self) -> None:
        pass


class Brittle(Material):
    def __init__(self):
        # Mechanical properties
        self.rho = 7e-6  # Mass Density
        self.lame = 1.2e5  # Lamé coefficient
        self.mu = 8e4  # Shear modulus

        # Crack phase properties
        self.Gc = 2.7  # Critical energy release rate
        self.lc = 0.5  # Characteristic length
        self.eta = 1e-3  # Crack phase viscosity parameter


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

        # Crack phase properties
        self.Gc = 30  # Critical energy release rate
        self.lc = 2e-2  # Characteristic length
        self.eta = 5e-5  # Crack phase viscosity parameter

    def getYieldStress(
        self, equivalent_plastic_strain, strain_rate, temperature, damage
    ):
        return (
            (self.A + self.B * max(equivalent_plastic_strain, 1e-6) ** self.n)
            * (1 + self.C * np.log(max(strain_rate, 1)))
            * (
                1
                - (
                    (temperature - self.reference_temperature)
                    / (self.melting_temperature - self.reference_temperature)
                )
                ** self.m
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
                - (
                    (temperature - self.reference_temperature)
                    / (self.melting_temperature - self.reference_temperature)
                )
                ** self.m
            )
        ) * (1 - damage) ** 2
