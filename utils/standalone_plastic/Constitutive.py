import dolfinx as dfx
import ufl

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


class IsotropicJohnsonCook2DModel(Constitutive):
    """
    Isotropic plasticity constitutive model
    Plane strain model
    """

    def __init__(
        self,
        material: Material.JohnsonCookMaterial,
        mesh: dfx.mesh.Mesh,
        element_type: str = None,
        degree: int = 1,
    ):
        super().__init__(material, mesh, element_type, degree)

        assert isinstance(
            self._material, Material.JohnsonCookMaterial
        ), "Material is not JohnsonCookMaterial"

        self.W = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (self._tdim * self._tdim,))
        )
        self.V = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (self._tdim,))
        )
        self.S = dfx.fem.functionspace(self._mesh, (self._element_type, self._degree))

        self.stress_vector = dfx.fem.Function(self.W, name="Stress")
        self.stress_vector_old = dfx.fem.Function(self.W)

        self.n_elastic_vector = dfx.fem.Function(self.W, name="Elastic unit normal")

        self.yield_stress = dfx.fem.Function(self.S, name="Yield stress")
        self.hardening = dfx.fem.Function(self.S, name="Hardening")

        self.beta = dfx.fem.Function(self.S)

        self.equivalent_stress = dfx.fem.Function(self.S, name="Equivalent stress")

        self.equivalent_plastic_strain = dfx.fem.Function(
            self.S, name="Equivalent plastic strain"
        )

        self.strain_rate = dfx.fem.Function(self.S, name="Strain rate")

        self.temperature = dfx.fem.Function(self.S, name="Temperature")

        self.yield_stress.x.array[:] = self._material.initial_yield_stress
        self.strain_rate.x.array[:] = self._material.reference_strain_rate
        self.temperature.x.array[:] = self._material.reference_temperature

    def getStrain(self, displacement):
        e = ufl.sym(ufl.grad(displacement))
        return ufl.as_tensor(
            [[e[0, 0], e[0, 1], 0.0], [e[1, 0], e[1, 1], 0.0], [0.0, 0.0, 0.0]]
        )

    def _getElasticStress(self, strain):
        assert isinstance(
            self._material, Material.JohnsonCookMaterial
        ), (
            "Material is not JohnsonCookMaterial"
        )  # Type check just for syntax highlighting and autocompletion

        return (
            self._material.lame * ufl.tr(strain) * ufl.Identity(3)  # 3 for Plane strain
            + 2 * self._material.shear_modulus * strain
        )

    def getStress(self, displacement):
        assert isinstance(
            self._material, Material.JohnsonCookMaterial
        ), (
            "Material is not JohnsonCookMaterial"
        )  # Type check just for syntax highlighting and autocompletion

        mu = self._material.shear_modulus
        # H = self._getHardeningModulus()
        H = self.hardening
        e = self.getStrain(displacement)
        N_elastic = self.asThreeDimensionalTensor(self.n_elastic_vector)

        return (
            self._getElasticStress(e)
            - 3
            * mu
            * (3 * mu / (3 * mu + H) - self.beta)
            * ufl.inner(N_elastic, e)
            * N_elastic
            - 2 * mu * self.beta * ufl.dev(e)
        )

    def stressProjection(self, strain_inc):
        assert isinstance(
            self._material, Material.JohnsonCookMaterial
        ), (
            "Material is not JohnsonCookMaterial"
        )  # Type check just for syntax highlighting and autocompletion

        Y = self.yield_stress
        H = self.hardening
        mu = self._material.shear_modulus

        stress_old = self.asThreeDimensionalTensor(self.stress_vector_old)
        stress_trial = stress_old + self._getElasticStress(strain_inc)

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
            ufl.as_vector([stress[0, 0], stress[1, 1], stress[2, 2], stress[0, 1]]),
            ufl.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]),
            beta,
            equivalent_plastic_strain_inc,
        )

    def asThreeDimensionalTensor(self, vector):
        return ufl.as_tensor(
            [
                [vector[0], vector[3], 0.0],
                [vector[3], vector[1], 0.0],
                [0.0, 0.0, vector[2]],
            ]
        )

    def getYieldStress(self):
        m = self._material

        assert isinstance(
            m, Material.JohnsonCookMaterial
        ), (
            "Material is not JohnsonCookMaterial"
        )  # Type check just for syntax highlighting and autocompletion

        y = (
            m.initial_yield_stress
            + m.strength_coefficient
            * self.equivalent_plastic_strain**m.hardening_exponent
        )
        y *= 1 + m.strain_rate_strength_coefficient * ufl.ln(
            ufl.max_value(self.strain_rate / m.reference_strain_rate, 1.0)
        )
        y *= 1 - (self.temperature - m.reference_temperature) / (
            m.melting_temperature - m.reference_temperature
        )
        return y

    def getHardening(self):
        m = self._material

        assert isinstance(
            m, Material.JohnsonCookMaterial
        ), (
            "Material is not JohnsonCookMaterial"
        )  # Type check just for syntax highlighting and autocompletion

        _h = (
            m.hardening_exponent
            * m.strength_coefficient
            * self.equivalent_plastic_strain ** (m.hardening_exponent - 1)
        )
        _h *= 1 + m.strain_rate_strength_coefficient * ufl.ln(
            ufl.max_value(self.strain_rate / m.reference_strain_rate, 1.0)
        )
        _h *= 1 - (self.temperature - m.reference_temperature) / (
            m.melting_temperature - m.reference_temperature
        )
        # h = Util.macaulayBracket(self.equivalent_plastic_strain) * _h
        h = ufl.conditional(ufl.gt(self.equivalent_plastic_strain, 0.0), _h, 0.0)
        return h
