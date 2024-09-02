import ufl
import ufl.algebra
import ufl.algorithms

from Material import Brittle


class ConstitutiveRelation:
    def getStrain(u):
        raise NotImplementedError

    def getStress(u):
        raise NotImplementedError

    def getStrainEnergyPositive(u):
        raise NotImplementedError


def macaulayBrackets(x):
    def ufl_abs(x):
        return ufl.sqrt(x**2)

    p = (x + ufl_abs(x)) / 2
    n = x - p
    return p, n


class Elastic_BourdinFrancfort2008(ConstitutiveRelation):
    # B. Bourdin, G.A. Francfort, J.-J. Marigo, The variational approach to fracture,
    # J. Elasticity 91 (1) (2008) 5–148.
    def __init__(self, material: Brittle):
        self.mu = material.mu
        self.lame = material.lame

    def getStrain(self, u):
        return ufl.sym(ufl.grad(u))

    def getStress(self, u, d):
        return (1 - d) ** 2 * (
            self.lame * ufl.tr(self.getStrain(u)) * ufl.Identity(len(u))
            + 2.0 * self.mu * self.getStrain(u)
        )

    def getStrainEnergyPositive(self, u, d):
        return 0.5 * self.lame * ufl.tr(self.getStrain(u)) ** 2 + self.mu * ufl.inner(
            self.getStrain(u), self.getStrain(u)
        )
        # return (1 - d) ** 2 * (
        #     0.5 * self.lame * ufl.tr(self.getStrain(u)) ** 2
        #     + self.mu * ufl.inner(self.getStrain(u), self.getStrain(u))
        # )

    # def getStrainEnergy(self, u):
    #     return 0.5 * (self.lame + self.mu) * (
    #         0.5 * (ufl.tr(self.getStrain(u)) + abs(ufl.tr(self.getStrain(u))))
    #     ) ** 2 + self.mu * ufl.inner(
    #         ufl.dev(self.getStrain(u)), ufl.dev(self.getStrain(u))
    #     )


class Elastic_AmorMarigo2009(ConstitutiveRelation):
    # H. Amor, J.-J. Marigo, C. Maurini, Regularized formulation of the variational brittle fracture with unilateral contact: Numerical experiments,
    # J. Mech. Phys. Solids 57 (2009) 1209–1229.
    def __init__(self, material: Brittle):
        self.mu = material.mu
        self.lame = material.lame

    def getStrain(self, u):
        return ufl.sym(ufl.grad(u))

    def getStress(self, u, d):
        c = self.lame + 2 * self.mu / 3
        tr_epsilon_p, tr_epsilon_n = macaulayBrackets(ufl.tr(self.getStrain(u)))

        sigma_p = c * tr_epsilon_p * ufl.Identity(len(u)) + 2 * self.mu * ufl.dev(
            self.getStrain(u)
        )
        sigma_n = c * tr_epsilon_n * ufl.Identity(len(u))

        return (1 - d) ** 2 * sigma_p + sigma_n

    def getStrainEnergyPositive(self, u, d):
        c = self.lame / 2 + self.mu / 3
        tr_epsilon_p, _ = macaulayBrackets(ufl.tr(self.getStrain(u)))

        return c * tr_epsilon_p**2 + self.mu * ufl.inner(
            ufl.dev(self.getStrain(u)), ufl.dev(self.getStrain(u))
        )


class ElasticPlastic(ConstitutiveRelation):
    def __init__(self, material):
        self.mu = material.mu

    def getStrain(u):
        return super().getStrain()
