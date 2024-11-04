import ufl
import ufl.algebra
import ufl.algorithms

from Material import Ductile, JohnsonCook


class ConstitutiveRelation:
    def __init__(self, name="Constitutive Relation", linear=False):
        self.name = name
        self.linear = linear

    def getStrain(u):
        raise NotImplementedError

    def getStress(u):
        raise NotImplementedError

    def getStrainEnergyPositive(u):
        raise NotImplementedError

    def __str__(self):
        if self.linear:
            return f"Linear {self.name}"
        else:
            return f"Nonlinear {self.name}"


def macaulayBrackets(x):
    # def ufl_abs(x):
    #     return ufl.sqrt(x**2)

    positive = (x + abs(x)) / 2
    negative = x - positive
    return positive, negative


class Elastic_BourdinFrancfort2008(ConstitutiveRelation):
    # B. Bourdin, G.A. Francfort, J.-J. Marigo, The variational approach to fracture,
    # J. Elasticity 91 (1) (2008) 5–148.
    def __init__(self, material: Ductile):
        super().__init__(name="Bourdin Francfort 2008", linear=True)

        self.mu = material.mu
        self.lame = material.lame

    def getStrain(self, u):
        return ufl.sym(ufl.grad(u))

    def getStress(self, u, d):
        return (1 - d) ** 2 * (
            self.lame * ufl.tr(self.getStrain(u)) * ufl.Identity(len(u))
            + 2.0 * self.mu * self.getStrain(u)
        )

    def getStrainEnergyPositive(self, u, _):
        return 0.5 * self.lame * ufl.tr(self.getStrain(u)) ** 2 + self.mu * ufl.inner(
            self.getStrain(u), self.getStrain(u)
        )

    # def getStrainEnergyPositive(self, u):
    #     return 0.5 * (self.lame + self.mu) * (
    #         0.5 * (ufl.tr(self.getStrain(u)) + abs(ufl.tr(self.getStrain(u))))
    #     ) ** 2 + self.mu * ufl.inner(
    #         ufl.dev(self.getStrain(u)), ufl.dev(self.getStrain(u))
    #     )


class Elastic_AmorMarigo2009(ConstitutiveRelation):
    # H. Amor, J.-J. Marigo, C. Maurini, Regularized formulation of the variational brittle fracture with unilateral contact: Numerical experiments,
    # J. Mech. Phys. Solids 57 (2009) 1209–1229.
    def __init__(self, material: Ductile):
        super().__init__(name="Amor Marigo 2009", linear=False)

        self.mu = material.mu
        self.lame = material.lame

    def getStrain(self, u):
        return ufl.sym(ufl.grad(u))

    def getStress(self, u, d):
        c = self.lame + 2 * self.mu / 3
        tr_epsilon_pos, tr_epsilon_neg = macaulayBrackets(ufl.tr(self.getStrain(u)))

        sigma_p = c * tr_epsilon_pos * ufl.Identity(len(u)) + 2 * self.mu * ufl.dev(
            self.getStrain(u)
        )
        sigma_n = c * tr_epsilon_neg * ufl.Identity(len(u))

        return (1 - d) ** 2 * sigma_p + sigma_n

    def getStrainEnergyPositive(self, u, d):
        c = self.lame / 2 + self.mu / 3
        tr_epsilon_pos, _ = macaulayBrackets(ufl.tr(self.getStrain(u)))

        return c * tr_epsilon_pos**2 + self.mu * ufl.inner(
            ufl.dev(self.getStrain(u)), ufl.dev(self.getStrain(u))
        )


# class Elastic_MieheWelschinger2010(ConstitutiveRelation):
#     def __init__(self, material: Brittle):
#         super().__init__(name="Miehe Welschinger 2010", linear=False)
#         self.mu = material.mu
#         self.lame = material.lame

#     def getStrain(self, u):
#         return ufl.sym(ufl.grad(u))

#     def getStress(self, u):
#         c = self.lame


class ElasticPlastic(ConstitutiveRelation):
    def __init__(self, material: JohnsonCook):
        self.material = material

    def getStrain(u):
        return ufl.sym(ufl.grad(u))

    def getStress(self, u, d, equivalent_plastic_strain, strain_rate, temperature):
        stress = (1 - d) ** 2 * (
            self.lame * ufl.tr(self.getStrain(u)) * ufl.Identity(len(u))
            + 2.0 * self.mu * self.getStrain(u)
        )

        equivalent_stress = ufl.sqrt(
            3.0 / 2.0 * ufl.inner(ufl.dev(stress), ufl.dev(stress))
        )

        yield_stress = self.material.getYieldStress(
            equivalent_plastic_strain, strain_rate, temperature, d
        )

        if equivalent_stress <= yield_stress:
            return stress

    def getStrainEnergyPositive(self, u, _):
        return 0.5 * self.lame * ufl.tr(self.getStrain(u)) ** 2 + self.mu * ufl.inner(
            self.getStrain(u), self.getStrain(u)
        )
