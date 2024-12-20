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

        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

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
        self.equivalent_plastic_strain_old = dfx.fem.Function(self.S)
        self.equivalent_plastic_strain_inc = dfx.fem.Function(self.S)

        self.strain_rate = dfx.fem.Function(self.S, name="Strain rate")

        self.temperature = dfx.fem.Function(self.S, name="Temperature")

        self.stress_vector_old.x.array[:] = 0.0
        self.equivalent_plastic_strain_old.x.array[:] = 0.0

        self.yield_stress.x.array[:] = self._material.initial_yield_stress
        self.strain_rate.x.array[:] = self._material.reference_strain_rate
        self.temperature.x.array[:] = self._material.reference_temperature

    def getStrain(self, displacement):
        e = ufl.sym(ufl.grad(displacement))
        return ufl.as_tensor(
            [[e[0, 0], e[0, 1], 0.0], [e[1, 0], e[1, 1], 0.0], [0.0, 0.0, 0.0]]
        )

    def _getElasticStress(self, strain):
        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

        return (
            self._material.lame * ufl.tr(strain) * ufl.Identity(3)  # 3 for Plane strain
            + 2 * self._material.shear_modulus * strain
        )

    def getStress(self, displacement):
        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

        strain = self.getStrain(displacement)

        mu = self._material.shear_modulus
        H = self.hardening
        n_elastic = self.asTensor3x3(self.n_elastic_vector)

        return (
            self._getElasticStress(strain)
            - 3
            * mu
            * (3 * mu / (3 * mu + H) - self.beta)
            * ufl.inner(n_elastic, strain)
            * n_elastic
            - 2 * mu * self.beta * ufl.dev(strain)
        )

    def stressProjection(self, strain_inc):
        assert issubclass(
            type(self._material), Material.JohnsonCookMaterial
        ), f"Material {self._material.__class__} is not JohnsonCookMaterial"

        Y = self.yield_stress
        H = self.hardening
        mu = self._material.shear_modulus

        stress_old = self.asTensor3x3(self.stress_vector_old)
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

    def asTensor3x3(self, vector):
        return ufl.as_tensor(
            [
                [vector[0], vector[3], 0.0],
                [vector[3], vector[1], 0.0],
                [0.0, 0.0, vector[2]],
            ]
        )


class DuctileFracture(IsotropicJohnsonCook2DModel):
    """
    Ductile fracture with isotropic plasticity constitutive model
    Plane strain model
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

        self.DS = dfx.fem.functionspace(self._mesh, ("DG", 0))

        self.elastic_strain_vector = dfx.fem.Function(self.W, name="Elastic strain")
        self.elastic_strain_vector_old = dfx.fem.Function(self.W)
        # self.plastic_strain_vector = dfx.fem.Function(self.W, name="Plastic strain")

        self._crack_phase = dfx.fem.Function(self.S)
        self.crack_driven_force = dfx.fem.Function(self.DS, name="Crack driven force")

        self.elastic_strain_energy_positive = dfx.fem.Function(
            self.DS, name="Elastic strain energy positive"
        )
        self.plastic_work = dfx.fem.Function(self.DS, name="Plastic work")
        self.plastic_work_old = dfx.fem.Function(self.DS)
        self.plastic_work_inc = dfx.fem.Function(self.DS)

        self.elastic_strain_vector_old.x.array[:] = 0.0
        self.crack_driven_force.x.array[:] = 0.0
        self.plastic_work_old.x.array[:] = 0.0

    # def getStrain(self, displacement):
    #     deformation_gradient = ufl.Identity(self._tdim) + ufl.grad(displacement)
    #     green_lagrange_strain = 0.5 * (
    #         ufl.dot(ufl.transpose(deformation_gradient), deformation_gradient)
    #         - ufl.Identity(self._tdim)
    #     )
    #     return ufl.as_tensor(
    #         [
    #             [green_lagrange_strain[0, 0], green_lagrange_strain[0, 1], 0.0],
    #             [green_lagrange_strain[1, 0], green_lagrange_strain[1, 1], 0.0],
    #             [0.0, 0.0, 0.0],
    #         ]
    #     )

    def stressProjection(self, strain_inc):
        assert isinstance(
            self._material, Material.JohnsonCookMaterial
        ), (
            "Material is not JohnsonCookMaterial"
        )  # Type check just for syntax highlighting and autocompletion

        stress_old = self.asTensor3x3(self.stress_vector_old)
        stress_trial = stress_old + self._getElasticStress(strain_inc)

        deviatoric_stress = ufl.dev(stress_trial)
        equivalent_stress = ufl.sqrt(
            1.5 * ufl.inner(deviatoric_stress, deviatoric_stress)
        )

        elastic_factor = equivalent_stress - self.yield_stress

        equivalent_plastic_strain_inc = Util.macaulayBracket(elastic_factor) / (
            3 * self._material.shear_modulus + self.hardening
        )

        yield_new = self.yield_stress + self.hardening * equivalent_plastic_strain_inc
        # f = yield_new / (
        #     yield_new + 3 * self._material.shear_modulus * equivalent_plastic_strain_inc
        # )
        # No use in that `stress` shall be calculated from `elastic strain` in 'Crack Phase Field problem'
        # stress = f * deviatoric_stress + ufl.tr(stress_trial) / 3 * ufl.Identity(3)

        plastic_strain_inc = (
            ufl.conditional(
                ufl.gt(equivalent_stress, 0.0),
                ufl.sqrt(1.5) * equivalent_plastic_strain_inc / equivalent_stress,
                0.0,
            )
            * deviatoric_stress
        )
        # plastic_strain_inc = ufl.as_matrix(
        #     [
        #         [plastic_strain_inc[0, 0], plastic_strain_inc[0, 1], 0.0],
        #         [plastic_strain_inc[1, 0], plastic_strain_inc[1, 1], 0.0],
        #         [0.0, 0.0, 0.0],
        #     ]
        # )

        elastic_strain = (
            self.asTensor3x3(self.elastic_strain_vector_old)
            + strain_inc
            - plastic_strain_inc
        )
        # self.plastic_strain_vector.interpolate(
        #     dfx.fem.Expression(
        #         ufl.as_vector(
        #             [
        #                 plastic_strain_inc[0, 0],
        #                 plastic_strain_inc[1, 1],
        #                 plastic_strain_inc[2, 2],
        #                 plastic_strain_inc[0, 1],
        #             ]
        #         )
        #         + self.plastic_strain_vector,
        #         self.W.element.interpolation_points(),
        #     )
        # )
        stress = self.getElasticStressWithFracture(elastic_strain)

        plastic_work_inc = (
            0.5 * (self.yield_stress + yield_new) * equivalent_plastic_strain_inc
        )
        return (
            # Values to be updated in the next iteration
            # for they participate in the calculation of the tangent modulus
            # which is main constitutive relation in the next iteration
            ufl.as_vector([stress[0, 0], stress[1, 1], stress[2, 2], stress[0, 1]]),
            # Values to be updated in the next increment not in the next iteration
            equivalent_plastic_strain_inc,
            plastic_work_inc,
        )

    def getYieldStress(self):
        return (1 - self._crack_phase) ** 2 * super().getYieldStress()

    def getHardening(self):
        return (1 - self._crack_phase) ** 2 * super().getHardening()

    def getElasticStressWithFracture(self, strain):
        c = self._material.lame + 2 / 3 * self._material.shear_modulus
        tr_strain_positive = Util.macaulayBracket(ufl.tr(strain))
        tr_strain_negative = ufl.tr(strain) - tr_strain_positive

        stress_positive = (
            c * tr_strain_positive * ufl.Identity(3)  # 3 for Plane strain
            + 2 * self._material.shear_modulus * ufl.dev(strain)
        )
        stress_negative = c * tr_strain_negative * ufl.Identity(3)

        return (1 - self._crack_phase) ** 2 * stress_positive + stress_negative

    def getElasticStrainEnergyPositive(self):
        mu = self._material.shear_modulus
        c = self._material.lame / 2 + mu / 3

        elastic_strain = self.asTensor3x3(self.elastic_strain_vector)
        tr_strain_positive = Util.macaulayBracket(ufl.tr(elastic_strain))
        dev_strain = ufl.dev(elastic_strain)

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
    Plane strain model
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
        self.principle_strain = np.zeros((self._nodes_num, 3))
        self.principle_strain_direction = np.zeros((self._nodes_num, 3, 3))

        self.__WW = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (3, 3))
        )
        # self.__strain = dfx.fem.Function(self.__WW)
        self.__stress = dfx.fem.Function(self.__WW)
        self.__elastic_strain_energy_positive = dfx.fem.Function(self.S)

        __VV = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (3,))
        )
        self._principle_strain_vis = dfx.fem.Function(__VV, name="Principle strain")

        self.__E = np.zeros((self._nodes_num, 3, 3))

    def _updatePrincipleStrain(self, strain):
        # Util.localProject(strain, self.__strain)
        # # self.__strain.interpolate(
        # #     dfx.fem.Expression(strain, self.__WW.element.interpolation_points())
        # # )
        # strain = np.reshape(self.__strain.x.array, (-1, 3, 3))
        # for idx, _strain in enumerate(strain):
        #     w, v = np.linalg.eig(_strain)
        #     self.principle_strain[idx] = w
        #     self.principle_strain_direction[idx] = v

        strain_vector = ufl.as_vector(
            [
                strain[0, 0],
                strain[1, 1],
                0.0,
                strain[0, 1],
            ]
        )
        Util.localProject(strain_vector, self.elastic_strain_vector)
        # self.elastic_strain_vector.interpolate(
        #     dfx.fem.Expression(strain_vector, self.W.element.interpolation_points())
        # )

        # E = self.elastic_strain_vector.x.array.reshape(-1, 4)
        # for i in range(self._nodes_num):
        #     e = np.array(
        #         [
        #             [E[i][0], E[i][3], 0.0],
        #             [E[i][3], E[i][1], 0.0],
        #             [0.0, 0.0, E[i][2]],
        #         ]
        #     )
        #     w, v = np.linalg.eig(e)
        #     self.principle_strain[i] = w
        #     self.principle_strain_direction[i] = v

        # Ev = self.elastic_strain_vector.x.array.reshape(-1, 4)
        # Zero = np.zeros(np.shape(Ev)[0])
        # E = np.array(
        #     [
        #         [Ev[:, 0], Ev[:, 3], Zero],
        #         [Ev[:, 3], Ev[:, 1], Zero],
        #         [Zero, Zero, Ev[:, 2]],
        #     ]
        # ).transpose(2, 1, 0)
        # self.principle_strain[:], self.principle_strain_direction[:] = np.linalg.eigh(E)

        self.__E[:, 0, 0] = self.elastic_strain_vector.x.array[0::4]
        self.__E[:, 0, 1] = self.elastic_strain_vector.x.array[3::4]
        self.__E[:, 1, 0] = self.elastic_strain_vector.x.array[3::4]
        self.__E[:, 1, 1] = self.elastic_strain_vector.x.array[1::4]
        self.__E[:, 2, 2] = self.elastic_strain_vector.x.array[2::4]
        self.principle_strain[:], self.principle_strain_direction[:] = np.linalg.eigh(
            self.__E
        )

    def getElasticStressWithFracture(self, strain):
        self._updatePrincipleStrain(strain)

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
            principle_stress[:, :, None] * np.eye(3),
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
