import dataclasses

@dataclasses.dataclass
class Material:
    rho: float
    lame: float
    mu: float

@dataclasses.dataclass
class JohnsonCook(Material):
    initial_yield_stress: float
    strength_coefficient: float
    strain_rate_strength_coefficient: float
    hardening_exponent: float
    temperature_exponent: float
    reference_strain_rate: float
    reference_temperature: float
    melting_temperature: float