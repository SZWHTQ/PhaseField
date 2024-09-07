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
