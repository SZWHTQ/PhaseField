# import dataclasses
from dataclasses import dataclass


@dataclass
class Material:
    pass


@dataclass
class ElasticMaterial(Material):
    mass_density: float
    lame: float
    shear_modulus: float
    viscosity: float


@dataclass
class JohnsonCookMaterial(ElasticMaterial):
    initial_yield_stress: float
    strength_coefficient: float
    strain_rate_strength_coefficient: float
    hardening_exponent: float
    temperature_exponent: float
    reference_strain_rate: float
    reference_temperature: float
    melting_temperature: float


@dataclass
class FractureMaterial(Material):
    fracture_viscosity: float
    fracture_characteristic_length: float
    critical_energy_release_rate: float


@dataclass
class DuctileFractureMaterial(JohnsonCookMaterial, FractureMaterial):
    threshold_energy: float


def __multipleInheritanceTest():
    m = DuctileFractureMaterial(
        mass_density=1.0,
        lame=1.0,
        shear_modulus=1.0,
        viscosity=1.0,
        initial_yield_stress=1.0,
        strength_coefficient=1.0,
        strain_rate_strength_coefficient=1.0,
        hardening_exponent=1.0,
        temperature_exponent=1.0,
        reference_strain_rate=1.0,
        reference_temperature=1.0,
        melting_temperature=1.0,
        fracture_viscosity=1.0,
        fracture_characteristic_length=1.0,
        critical_energy_release_rate=1.0,
        threshold_energy=1.0,
    )
    print(m)


if __name__ == "__main__":
    __multipleInheritanceTest()
