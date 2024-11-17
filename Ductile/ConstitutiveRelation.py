import ufl
import dolfinx as dfx

import numpy as np

from Material import Ductile, JohnsonCook


class ConstitutiveRelation:
    def __init__(self, name="Constitutive Relation", linear=False):
        self.name = name
        self.linear = linear

    def getStrain(self, u):
        raise NotImplementedError

    def getStress(self, u):
        raise NotImplementedError

    def getStrainEnergyPositive(self, u):
        raise NotImplementedError

    def __str__(self):
        if self.linear:
            return f"Linear {self.name}"
        else:
            return f"Nonlinear {self.name}"


def macaulayBrackets(x):
    positive = (x + abs(x)) / 2
    negative = x - positive
    return positive, negative


class ElastoPlastic_BourdinFrancfort2008(ConstitutiveRelation):
    def __init__(self, material: Ductile | JohnsonCook):
        super().__init__(name="Bourdin Francfort 2008 Plastic")

        self.linear = True

        self.mu = material.mu
        self.lame = material.lame

    def getStrain(self, u: dfx.fem.Function) -> dfx.fem.Function:
        return ufl.sym(ufl.nabla_grad(u))

    def getStress(self, u: dfx.fem.Function, d: dfx.fem.Function) -> dfx.fem.Function:
        return (1 - d) ** 2 * (
            self.lame * ufl.tr(self.getStrain(u)) * ufl.Identity(len(u))
            + 2.0 * self.mu * self.getStrain(u)
        )

    def getStressByStrain(
        self, e: dfx.fem.Function, d: dfx.fem.Function, dim=2
    ) -> dfx.fem.Function:
        return (1 - d) ** 2 * (
            self.lame * ufl.tr(e) * ufl.Identity(dim) + 2 * self.mu * e
        )

    def getStrainEnergyPositive(self, e: dfx.fem.Function) -> dfx.fem.Function:
        return 0.5 * self.lame * ufl.tr(e) ** 2 + self.mu * ufl.inner(e, e)


class ElastoPlastic_AmorMarigo2009(ConstitutiveRelation):
    def __init__(self, material: JohnsonCook):
        super().__init__(name="Amor Marigo 2009", linear=False)

        self.linear = False

        self.mu = material.mu
        self.lame = material.lame

    def getStrain(self, u):
        return ufl.sym(ufl.nabla_grad(u))

    def getStress(self, u, d):
        return self.getStressByStrain(self.getStrain(u), d, dim=len(u))

    def getStressByStrain(self, e, d, dim=2):
        c = self.lame + 2 * self.mu / 3
        tr_epsilon_pos, tr_epsilon_neg = macaulayBrackets(ufl.tr(e))

        sigma_p = c * tr_epsilon_pos * ufl.Identity(dim) + 2 * self.mu * ufl.dev(e)
        sigma_n = c * tr_epsilon_neg * ufl.Identity(dim)

        return (1 - d) ** 2 * sigma_p + sigma_n

    def getStrainEnergyPositive(self, e):
        S = dfx.fem.functionspace(e.function_space.mesh, ("CG", 1))
        strain_energy_positive = dfx.fem.Function(S)
        c = self.lame / 2 + self.mu / 3
        tr_epsilon_pos, _ = macaulayBrackets(ufl.tr(e))

        strain_energy_positive_expr = dfx.fem.Expression(
            c * tr_epsilon_pos**2 + self.mu * ufl.inner(ufl.dev(e), ufl.dev(e)),
            S.element.interpolation_points(),
        )
        strain_energy_positive.interpolate(strain_energy_positive_expr)

        return strain_energy_positive


class Elastoplastic(ConstitutiveRelation):
    def __init__(self, material: JohnsonCook):
        super().__init__(name="Amor Marigo 2009", linear=False)

        self.linear = False

        self.mu = material.mu
        self.lame = material.lame

        self.eigens_prepared = False

    def getStrain(self, u):
        return ufl.sym(ufl.nabla_grad(u))

    def getStress(self, u: dfx.fem.Function, d: dfx.fem.Function):
        # return (1 - d) ** 2 * (
        #     self.lame * ufl.tr(self.getStrain(u)) * ufl.Identity(len(u))
        #     + 2.0 * self.mu * self.getStrain(u)
        # )

        # e = self.getStrain(u)
        # dim = len(u)
        # c = self.lame + 2 * self.mu / 3
        # tr_epsilon_pos, tr_epsilon_neg = macaulayBrackets(ufl.tr(e))

        # sigma_p = c * tr_epsilon_pos * ufl.Identity(dim) + 2 * self.mu * ufl.dev(e)
        # sigma_n = c * tr_epsilon_neg * ufl.Identity(dim)
        # return (1 - d) ** 2 * sigma_p + sigma_n

        dim = len(u)
        W = dfx.fem.functionspace(u.function_space.mesh, ("CG", 1, (dim, dim)))
        e_expr = dfx.fem.Expression(
            self.getStrain(u),
            W.element.interpolation_points(),
        )
        e = dfx.fem.Function(W)
        e.interpolate(e_expr)
        return self.getStressByStrain(e, d, dim)

    # def getStressByStrain(self, e: dfx.fem.Function, d: dfx.fem.Function, dim=2):
    #     # c = self.lame + 2 * self.mu / 3
    #     # tr_epsilon_pos, tr_epsilon_neg = macaulayBrackets(ufl.tr(e))

    #     # sigma_p = c * tr_epsilon_pos * ufl.Identity(dim) + 2 * self.mu * ufl.dev(e)
    #     # sigma_n = c * tr_epsilon_neg * ufl.Identity(dim)

    #     # return (1 - d) ** 2 * sigma_p + sigma_n
    #     self._prepare_eigens(e, dim)
    #     nodewise_stress = np.zeros((self._nodes_num, dim, dim))
    #     s = np.zeros((dim, dim))
    #     for idx in range(self._nodes_num):
    #         tr = np.sum(self._nodewise_principle_strains[idx])
    #         f_tr = 1 if tr > 0 else 0
    #         for j in range(dim):
    #             principle_e = self._nodewise_principle_strains[idx, j]
    #             f = 1 if principle_e > 0 else 0
    #             s[j, j] = (
    #                 2 * self.mu * principle_e * (1 - f * d) ** 2
    #                 + self.lame * tr * (1 - f_tr * d) ** 2
    #             )
    #         principle_d: np.matrix = self._nodewise_principle_directions[idx]
    #         nodewise_stress[idx] = np.matmul(principle_d, np.matmul(s, principle_d.T))
    #     stress = dfx.fem.Function(e.function_space)
    #     stress.x.array[:] = nodewise_stress.flatten()
    #     return stress

    # def getStrainEnergyPositive(self, e: dfx.fem.Function):
    #     assert self.eigens_prepared, "Eigens are not prepared"
    #     S = dfx.fem.functionspace(e.function_space.mesh, ("CG", 1))
    #     strain_energy_positive = dfx.fem.Function(S)
    #     dim = self._nodewise_principle_strains.shape[1]
    #     for idx in range(self._nodes_num):
    #         tr = np.sum(self._nodewise_principle_strains[idx])
    #         strain_energy_positive.x.array[idx] = 0.5 * self.lame * max(tr, 0.0) ** 2
    #         for j in range(dim):
    #             strain_energy_positive.x.array[idx] += (
    #                 self.mu * max(self._nodewise_principle_strains[idx, j], 0.0) ** 2
    #             )
    #     return strain_energy_positive
    def getStressByStrain(self, e: dfx.fem.Function, d, dim=2):
        self._prepare_eigens(e, dim)
        tr = np.sum(self._nodewise_principle_strains, axis=1)
        f_tr = (tr > 0).astype(float)
        f = (self._nodewise_principle_strains > 0).astype(float)
        principle_e = self._nodewise_principle_strains
        # d_array = d.x.array[:self._nodes_num]
        if isinstance(d, dfx.fem.Function):
            d_array = d.x.array[:]
        elif isinstance(d, np.ndarray):
            d_array = d
        elif isinstance(d, float):
            d_array = np.array([d] * self._nodes_num)
        s = (
            2 * self.mu * principle_e * (1 - f * d_array[:, None]) ** 2
            + self.lame * tr[:, None] * (1 - f_tr * d_array)[:, None] ** 2
        )
        nodewise_stress = np.einsum(
            "nij,njk,nlk->nil",
            self._nodewise_principle_directions,
            s[:, :, None] * np.eye(dim),
            self._nodewise_principle_directions,
        )
        stress = dfx.fem.Function(e.function_space)
        stress.x.array[:] = nodewise_stress.reshape(-1)
        return stress

    def getStrainEnergyPositive(self, e: dfx.fem.Function):
        assert self.eigens_prepared, "Eigens are not prepared"
        S = dfx.fem.functionspace(e.function_space.mesh, ("CG", 1))
        strain_energy_positive = dfx.fem.Function(S)
        tr = np.sum(self._nodewise_principle_strains, axis=1)
        tr_positive = np.maximum(tr, 0.0)
        strain_energy_positive_values = 0.5 * self.lame * tr_positive**2
        strain_energy_positive_values += np.sum(
            self.mu * np.maximum(self._nodewise_principle_strains, 0.0) ** 2, axis=1
        )
        strain_energy_positive.x.array[:] = strain_energy_positive_values
        return strain_energy_positive

    def _prepare_eigens(self, e: dfx.fem.Function, dim: int):
        if dim == 2 or dim == 3:
            nodewise_strains = np.reshape(e.x.array[:], (-1, dim, dim))
            self._nodes_num = len(nodewise_strains)
            self._nodewise_principle_strains = np.zeros((self._nodes_num, dim))
            self._nodewise_principle_directions = np.zeros((self._nodes_num, dim, dim))
            for idx, strain in enumerate(nodewise_strains):
                eigenvalues, eigenvectors = np.linalg.eig(strain)
                self._nodewise_principle_strains[idx] = eigenvalues
                self._nodewise_principle_directions[idx] = eigenvectors
            self.eigens_prepared = True
        else:
            raise ValueError("Topology dimension is not valid")

    # # Get positive strain energy and update stress by principle strain
    # if topology_dim == 2:
    #     elementwise_eigenvalues = []
    #     elementwise_eigenvectors = []
    #     for cell in range(len(mesh.cells)):
    #         points = dfx.cpp.mesh.cell_dofs(mesh, cell)

    #         strain_value = strain.eval(points)

    #         for strain_matrix in strain_value:
    #             eigenvalues, eigenvectors = np.linalg.eig(strain_matrix)
    #             elementwise_eigenvalues.append(eigenvalues)
    #             elementwise_eigenvectors.append(eigenvectors)

    #     principle_strain_1 = dfx.fem.Function(S)
    #     principle_strain_2 = dfx.fem.Function(S)
    #     principle_direction_1 = dfx.fem.Function(V)
    #     principle_direction_2 = dfx.fem.Function(V)

    #     with principle_strain_1

    #     principle_strains = []
    #     principle_directions = []

    #     principle_trace = dfx.fem.Function(S)

    #     for i in range(nconv):
    #         principle_strains.append(E.getEigenvalue(i))
    #         vr, vi = A.getVecs()
    #         E.getEigenvector(i, vr, vi)
    #         principle_directions.append(vr.getArray())

    #     principle_direction_matrix = ufl.as_matrix(principle_directions)

    #     principle_trace_expr = dfx.fem.Expression(
    #         principle_strains[0] + principle_strains[1],
    #         S.element.interpolation_points(),
    #     )

    #     positive_strain_energy_expr = dfx.fem.Expression(
    #         0.5
    #         * (
    #             material.lame * ufl.max_value(principle_trace, 0.0) ** 2
    #             + 2
    #             * material.mu
    #             * (
    #                 ufl.max_value(principle_strains[0], 0.0) ** 2
    #                 + ufl.max_value(principle_strains[1], 0.0) ** 2
    #             )
    #         )
    #     )

    #     def get_principle_stress(e):
    #         return (
    #             material.lame
    #             * principle_trace
    #             * (
    #                 1
    #                 - ufl.conditional(ufl.gt(principle_trace, 0.0), 1.0, 0.0)
    #                 * crack_phase
    #             )
    #             ** 2
    #             + 2
    #             * material.mu
    #             * e
    #             * (1 - ufl.conditional(ufl.gt(e, 0.0), 1.0, 0.0) * crack_phase)
    #             ** 2,
    #         )

    #     principle_stress_1 = dfx.fem.Function(S)
    #     principle_stress_2 = dfx.fem.Function(S)

    #     principle_stress_1.interpolate(
    #         dfx.fem.Expression(
    #             get_principle_stress(principle_strains[0]),
    #             S.element.interpolation_points(),
    #         )
    #     )
    #     principle_stress_2.interpolate(
    #         dfx.fem.Expression(
    #             get_principle_stress(principle_strains[1]),
    #             S.element.interpolation_points(),
    #         )
    #     )

    #     update_stress_expr = dfx.fem.Expression(
    #         ufl.dot(
    #             ufl.dot(
    #                 principle_direction_matrix,
    #                 ufl.as_matrix(
    #                     [[principle_stress_1, 0], [0, principle_stress_2]]
    #                 ),
    #             ),
    #             ufl.transpose(principle_direction_matrix),
    #         ),
    #         W.element.interpolation_points(),
    #     )
    #     stress.interpolate(update_stress_expr)

    # elif topology_dim == 3:
    #     raise NotImplementedError("3D is not implemented yet")
    # else:
    #     raise ValueError("Topology dimension is not valid")
