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
