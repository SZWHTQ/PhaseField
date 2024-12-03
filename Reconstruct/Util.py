import dolfinx as dfx
import ufl

import dolfinx.fem.petsc as petsc

from mpi4py import MPI
# from petsc4py import PETSc

import numpy as np


def macaulayBracket(x):
    return (x + abs(x)) / 2


def getLamesParameters(E: float, nu: float) -> tuple[float, float]:
    """
    Compute the Lame's parameters from Young's modulus and Poisson's ratio.

    Parameters:
    E: Young's modulus.
    nu: Poisson's ratio.

    Returns:
    Tuple containing the Lame's parameters (lambda, mu).
    """
    lame = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lame, mu


def localProject(
    v: ufl.Form, u: dfx.fem.Function = None, V: dfx.fem.FunctionSpace = None
) -> dfx.fem.Function | None:
    """
    Project a UFL expression onto a function space in DOLFINx.

    Parameters:
    v (ufl.Form|any): UFL expression or DOLFINx function to project.
    V (dolfinx.fem.FunctionSpace): DOLFINx function space where the projection is performed.
    u (ufl.fem.Function): Optional existing DOLFINx Function to store the projection result.

    Returns:
    DOLFINx Function containing the projection result if `u` is None.
    """
    if u is None:
        if V is None:
            raise ValueError("FunctionSpace is required when u is not provided.")
        u = dfx.fem.Function(V)
    if V is None:
        V = u.function_space
    dx = ufl.Measure("dx", domain=V.mesh)

    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)
    F = ufl.inner(dv, v_) * dx - ufl.inner(v, v_) * dx
    a = dfx.fem.form(ufl.lhs(F))
    L = dfx.fem.form(ufl.rhs(F))
    problem = petsc.LinearProblem(
        a=a,
        L=L,
        u=u,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    return problem.solve()


### The old version of the function which is buggy in mpi mode
### Wonder why. #TODO: Explain this.
### Probably because of the way the solver is created.
### First: Compare the implementation in `LinearProblem` and the old `localProject`.
# def localProject(
#     v: ufl.Form, V: dfx.fem.FunctionSpace, u: dfx.fem.Function = None
# ) -> dfx.fem.Function | None:
#     """
#     Project a UFL expression onto a function space in DOLFINx.

#     Parameters:
#     v (ufl.Form|any): UFL expression or DOLFINx function to project.
#     V (dolfinx.fem.FunctionSpace): DOLFINx function space where the projection is performed.
#     u (ufl.fem.Function): Optional existing DOLFINx Function to store the projection result.

#     Returns:
#     DOLFINx Function containing the projection result if `u` is None.
#     """
#     dx = ufl.Measure("dx", domain=V.mesh)

#     # Define the variational problem
#     dv = ufl.TrialFunction(V)
#     v_ = ufl.TestFunction(V)
#     a_proj = dfx.fem.form(ufl.inner(dv, v_) * dx)
#     b_proj = dfx.fem.form(ufl.inner(v, v_) * dx)

#     # Assemble the matrix and vector
#     # A = petsc.create_matrix(a_proj)
#     A = petsc.assemble_matrix(a_proj)
#     A.assemble()
#     b = petsc.assemble_vector(b_proj)
#     b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

#     # Create the solver
#     solver = PETSc.KSP().create(V.mesh.comm)
#     solver.setOperators(A)
#     solver.setType(PETSc.KSP.Type.PREONLY)
#     solver.getPC().setType(PETSc.PC.Type.LU)

#     # Solve the linear system
#     # return u if flag else None
#     if u is None:
#         u = dfx.fem.Function(V)
#         solver.solve(b, u.x.petsc_vec)
#         return u
#     else:
#         solver.solve(b, u.x.petsc_vec)


def getL2Norm(
    u: dfx.fem.Function,
    dx: ufl.Measure = None,
    comm: MPI.Intracomm = None,
) -> float:
    """
    Compute the L2 norm of a given Function u.

    Args:
    u (dolfinx.fem.Function): Array to compute the L2 norm.
    dx (ufl.Measure): Measure to use for the integration.
    comm (MPI.Intracomm): MPI communicator.

    Returns:
    float: L2 norm of the u.
    """

    if dx is None:
        dx = ufl.Measure("dx", domain=u.function_space.mesh)
    if comm is None:
        comm = u.function_space.mesh.comm
    L2_2 = dfx.fem.form(ufl.inner(u, u) * dx)
    norm = np.sqrt(comm.allreduce(dfx.fem.assemble_scalar(L2_2), op=MPI.SUM))

    return norm
