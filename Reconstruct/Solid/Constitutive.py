import dolfinx as dfx
import ufl

import numpy as np

import Material
import Util


class Constitutive:
    def __init__(
        self,
        material: Material.Material,
        mesh: dfx.mesh.Mesh,
        element_type: str,
        degree: int,
    ):
        self._material = material
        self._mesh = mesh

        if element_type is None:
            self._element_type = "Lagrange"
        else:
            self._element_type = element_type

        self._degree = degree

        self._tdim = mesh.topology.dim

    def getStrain(self, displacement):
        return ufl.sym(ufl.nabla_grad(displacement))

    def getStress(self):
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )


class IsotropicJohnsonCook(Constitutive):
    """
    Isotropic plasticity constitutive model
    Solid model
    """

    def __init__(
        self,
        material: Material.JohnsonCookMaterial,
        mesh: dfx.mesh.Mesh,
        element_type: str = None,
        degree: int = 1,
    ):
        super().__init__(material, mesh, element_type, degree)

        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

        self.W = dfx.fem.functionspace(
            self._mesh,
            (
                self._element_type,
                self._degree,
                (
                    self._tdim,
                    self._tdim,
                ),
            ),
        )
        self.V = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (self._tdim,))
        )
        self.S = dfx.fem.functionspace(self._mesh, (self._element_type, self._degree))

        self.stress = dfx.fem.Function(self.W, name="Stress")
        self.stress_old = dfx.fem.Function(self.W)

        self.n_elastic = dfx.fem.Function(self.W, name="Elastic unit normal")

        self.yield_stress = dfx.fem.Function(self.S, name="Yield stress")
        self.hardening = dfx.fem.Function(self.S, name="Hardening")

        self.beta = dfx.fem.Function(self.S)

        self.equivalent_stress = dfx.fem.Function(self.S, name="Equivalent stress")

        self.equivalent_plastic_strain = dfx.fem.Function(
            self.S, name="Equivalent plastic strain"
        )
        self.equivalent_plastic_strain_old = dfx.fem.Function(self.S)
        self.equivalent_plastic_strain_inc = dfx.fem.Function(self.S)

        self.strain_rate = dfx.fem.Function(self.S, name="Strain rate")

        self.temperature = dfx.fem.Function(self.S, name="Temperature")

        self.stress_old.x.array[:] = 0.0
        self.equivalent_plastic_strain_old.x.array[:] = 0.0

        self.yield_stress.x.array[:] = self._material.initial_yield_stress
        self.strain_rate.x.array[:] = self._material.reference_strain_rate
        self.temperature.x.array[:] = self._material.reference_temperature

    def getElasticStress(self, strain):
        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

        return (
            self._material.lame * ufl.tr(strain) * ufl.Identity(self._tdim)
            + 2 * self._material.shear_modulus * strain
        )

    def getStress(self, displacement):
        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

        strain = self.getStrain(displacement)

        mu = self._material.shear_modulus
        H = self.hardening

        return (
            self.getElasticStress(strain)
            - 3
            * mu
            * (3 * mu / (3 * mu + H) - self.beta)
            * ufl.inner(self.n_elastic, strain)
            * self.n_elastic
            - 2 * mu * self.beta * ufl.dev(strain)
        )

    def stressProjection(self, strain_inc):
        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

        Y = self.yield_stress
        H = self.hardening
        mu = self._material.shear_modulus

        stress_trial = self.stress_old + self.getElasticStress(strain_inc)

        deviatoric_stress = ufl.dev(stress_trial)
        equivalent_stress = ufl.sqrt(
            1.5 * ufl.inner(deviatoric_stress, deviatoric_stress)
        )

        elastic_factor = equivalent_stress - Y - H * self.equivalent_plastic_strain

        equivalent_plastic_strain_inc = Util.macaulayBracket(elastic_factor) / (
            3 * mu + H
        )
        n_elas = (
            deviatoric_stress
            / equivalent_stress
            * Util.macaulayBracket(elastic_factor)
            / elastic_factor
        )
        beta = 3 * mu * equivalent_plastic_strain_inc / equivalent_stress

        stress = stress_trial - beta * deviatoric_stress

        return (
            stress,
            n_elas,
            beta,
            equivalent_plastic_strain_inc,
        )

    def getYieldStress(self):
        m = self._material

        assert issubclass(
            type(m), Material.JohnsonCookMaterial
        ), f"Material {m.__class__} is not JohnsonCookMaterial"

        y = (
            m.initial_yield_stress
            + m.strength_coefficient
            * self.equivalent_plastic_strain**m.hardening_exponent
        )
        y *= 1 + m.strain_rate_strength_coefficient * ufl.ln(
            ufl.max_value(self.strain_rate / m.reference_strain_rate, 1.0)
        )
        y *= (
            1
            - (
                (self.temperature - m.reference_temperature)
                / (m.melting_temperature - m.reference_temperature)
            )
            ** m.temperature_exponent
        )
        return y

    def getHardening(self):
        m = self._material

        assert issubclass(
            type(m), Material.JohnsonCookMaterial
        ), f"Material {m.__class__} is not JohnsonCookMaterial"

        _h = (
            m.hardening_exponent
            * m.strength_coefficient
            * self.equivalent_plastic_strain ** (m.hardening_exponent - 1)
        )
        _h *= 1 + m.strain_rate_strength_coefficient * ufl.ln(
            ufl.max_value(self.strain_rate / m.reference_strain_rate, 1.0)
        )
        _h *= (
            1
            - (
                (self.temperature - m.reference_temperature)
                / (m.melting_temperature - m.reference_temperature)
            )
            ** m.temperature_exponent
        )
        # h = Util.macaulayBracket(self.equivalent_plastic_strain) * _h
        h = ufl.conditional(ufl.gt(self.equivalent_plastic_strain, 0.0), _h, 0.0)
        return h


class DuctileFracture(IsotropicJohnsonCook):
    """
    Ductile fracture with isotropic plasticity constitutive model
    Solid model
    """

    def __init__(
        self,
        material: Material.DuctileFractureMaterial,
        mesh: dfx.mesh.Mesh,
        element_type: str = None,
        degree: int = 1,
    ):
        super().__init__(material, mesh, element_type, degree)

        assert issubclass(
            type(self._material), Material.DuctileFractureMaterial
        ), f"Material {self._material.__class__} is not DuctileFractureMaterial"

        self.DS = dfx.fem.functionspace(self._mesh, ("DG", self._degree - 1))

        self.elastic_strain = dfx.fem.Function(self.W, name="Elastic strain")
        self.elastic_strain_old = dfx.fem.Function(self.W)
        # self.plastic_strain_vector = dfx.fem.Function(self.W, name="Plastic strain")

        self._crack_phase = dfx.fem.Function(self.S)
        self.crack_driven_force = dfx.fem.Function(self.DS, name="Crack driven force")

        self.elastic_strain_energy_positive = dfx.fem.Function(
            self.DS, name="Elastic strain energy positive"
        )
        self.plastic_work = dfx.fem.Function(self.DS, name="Plastic work")
        self.plastic_work_old = dfx.fem.Function(self.DS)
        self.plastic_work_inc = dfx.fem.Function(self.DS)

        self.elastic_strain_old.x.array[:] = 0.0
        self.crack_driven_force.x.array[:] = 0.0
        self.plastic_work_old.x.array[:] = 0.0

    def getStrain(self, displacement):
        identity = ufl.Identity(self._tdim)
        deformation_gradient = identity + ufl.grad(displacement)

        return (deformation_gradient.T * deformation_gradient - identity) / 2

    def stressProjection(self, strain_inc):
        assert isinstance(
            self._material, Material.JohnsonCookMaterial
        ), (
            "Material is not JohnsonCookMaterial"
        )  # Type check just for syntax highlighting and autocompletion

        stress_trial = self.stress_old + self.getElasticStress(strain_inc)

        deviatoric_stress = ufl.dev(stress_trial)
        equivalent_stress = ufl.sqrt(
            1.5 * ufl.inner(deviatoric_stress, deviatoric_stress)
        )

        elastic_factor = equivalent_stress - self.yield_stress

        equivalent_plastic_strain_inc = Util.macaulayBracket(elastic_factor) / (
            3 * self._material.shear_modulus + self.hardening
        )

        yield_new = self.yield_stress + self.hardening * equivalent_plastic_strain_inc

        plastic_strain_inc = (
            ufl.conditional(
                ufl.gt(equivalent_stress, 0.0),
                ufl.sqrt(1.5) * equivalent_plastic_strain_inc / equivalent_stress,
                0.0,
            )
            * deviatoric_stress
        )
        elastic_strain = self.elastic_strain_old + strain_inc - plastic_strain_inc
        Util.localProject(elastic_strain, self.elastic_strain)

        stress = self.getElasticStressWithFracture()

        plastic_work_inc = (
            0.5 * (self.yield_stress + yield_new) * equivalent_plastic_strain_inc
        )
        return (
            # Values to be updated in the next iteration
            # for they participate in the calculation of the tangent modulus
            # which is main constitutive relation in the next iteration
            stress,
            # Values to be updated in the next increment not in the next iteration
            equivalent_plastic_strain_inc,
            plastic_work_inc,
        )

    def getYieldStress(self):
        return (1 - self._crack_phase) ** 2 * super().getYieldStress()

    def getHardening(self):
        return (1 - self._crack_phase) ** 2 * super().getHardening()

    def getElasticStressWithFracture(self):
        c = self._material.lame + 2 / 3 * self._material.shear_modulus

        tr_strain_positive = Util.macaulayBracket(ufl.tr(self.elastic_strain))
        tr_strain_negative = ufl.tr(self.elastic_strain) - tr_strain_positive

        stress_positive = c * tr_strain_positive * ufl.Identity(
            self._tdim
        ) + 2 * self._material.shear_modulus * ufl.dev(self.elastic_strain)
        stress_negative = c * tr_strain_negative * ufl.Identity(self._tdim)

        return (1 - self._crack_phase) ** 2 * stress_positive + stress_negative

    def getElasticStrainEnergyPositive(self):
        mu = self._material.shear_modulus
        c = self._material.lame / 2 + mu / 3

        tr_strain_positive = Util.macaulayBracket(ufl.tr(self.elastic_strain))
        dev_strain = ufl.dev(self.elastic_strain)

        return c * tr_strain_positive**2 + mu * ufl.inner(dev_strain, dev_strain)

    def getCrackDrivenForce(self):
        assert isinstance(
            self._material, Material.DuctileFractureMaterial
        ), "Material is not DuctileFractureMaterial"
        Hf = self.elastic_strain_energy_positive + 0.1 * (
            self.plastic_work - self._material.threshold_energy
        )
        return ufl.conditional(
            ufl.gt(Hf, self.crack_driven_force), Hf, self.crack_driven_force
        )


class DuctileFracturePrincipleStrainDecomposition(DuctileFracture):
    """
    Ductile fracture with isotropic plasticity constitutive model
    Solid model
    """

    def __init__(
        self,
        material: Material.DuctileFractureMaterial,
        mesh: dfx.mesh.Mesh,
        element_type: str = None,
        degree: int = 1,
    ):
        super().__init__(material, mesh, element_type, degree)

        self._nodes_num: int = self._crack_phase.x.array[:].size
        self.principle_strain = np.zeros((self._nodes_num, self._tdim))
        self.principle_strain_direction = np.zeros(
            (self._nodes_num, self._tdim, self._tdim)
        )

        self.__WW = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (self._tdim, self._tdim))
        )
        # self.__strain = dfx.fem.Function(self.__WW)
        self.__stress = dfx.fem.Function(self.__WW)
        self.__elastic_strain_energy_positive = dfx.fem.Function(self.S)

        __VV = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (self._tdim,))
        )
        self._principle_strain_vis = dfx.fem.Function(__VV, name="Principle strain")

    def _updatePrincipleStrain(self):
        self.principle_strain[:], self.principle_strain_direction[:] = np.linalg.eigh(
            self.elastic_strain.x.array[:].reshape(-1, self._tdim, self._tdim)
        )

    def getElasticStressWithFracture(self):
        self._updatePrincipleStrain()

        trace = np.sum(self.principle_strain, axis=1)
        trace_factor = (trace > 0).astype(float)
        factor = (self.principle_strain > 0).astype(float)
        d_array = self._crack_phase.x.array[:]
        principle_stress = (
            2
            * self._material.shear_modulus
            * self.principle_strain
            * (1 - factor * d_array[:, None]) ** 2
            + self._material.lame
            * trace[:, None]
            * (1 - trace_factor * d_array)[:, None] ** 2
        )
        self.__stress.x.array[:] = np.einsum(
            "nij,njk,nlk->nil",
            self.principle_strain_direction,
            principle_stress[:, :, None] * np.eye(self._tdim),
            self.principle_strain_direction,
        ).flatten()

        return self.__stress

    def getElasticStrainEnergyPositive(self):
        trace = np.sum(self.principle_strain, axis=1)
        trace_positive = np.maximum(trace, 0.0)
        self.__elastic_strain_energy_positive.x.array[:] = (
            0.5 * self._material.lame * trace_positive**2
        )
        self.__elastic_strain_energy_positive.x.array[:] += np.sum(
            self._material.shear_modulus * np.maximum(self.principle_strain, 0.0) ** 2,
            axis=1,
        )
        return self.__elastic_strain_energy_positive
