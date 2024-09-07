import numpy as np
from mpi4py import MPI as mpi
import dolfinx as dfx
import dolfinx.fem.petsc as petsc
import ufl

import time

# Create the mesh
mesh = dfx.mesh.create_unit_square(
    mpi.COMM_WORLD, 400, 400, cell_type=dfx.mesh.CellType.quadrilateral
)

topology_dim = mesh.topology.dim
boundary_dim = topology_dim - 1


# Define function spaces
W = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim,)))
V = dfx.fem.functionspace(mesh, ("CG", 1))
WW = dfx.fem.functionspace(mesh, ("DG", 1))


# Variational forms
u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
p, q = ufl.TrialFunction(V), ufl.TestFunction(V)

displacement = dfx.fem.Function(W)
displacement_old = dfx.fem.Function(W)

crack_phase = dfx.fem.Function(V)
crack_phase_old = dfx.fem.Function(V)

energy_history = dfx.fem.Function(WW)

displacement.name = "Displacement"
crack_phase.name = "Crack Phase"
energy_history.name = "Energy History"


# Define material parameters
Gc = 2.7
l = 0.005
lamda = 121.1538e3
mu = 80.7692e3


# Define constitutive functions
def getStrain(u):
    return ufl.sym(ufl.grad(u))


def getStress(u):
    return 2.0 * mu * getStrain(u) + lamda * ufl.tr(getStrain(u)) * ufl.Identity(len(u))


def getStrainEnergy(u):
    return 0.5 * (lamda + mu) * (
        0.5 * (ufl.tr(getStrain(u)) + abs(ufl.tr(getStrain(u))))
    ) ** 2 + mu * ufl.inner(ufl.dev(getStrain(u)), ufl.dev(getStrain(u)))


energy_history_expr = dfx.fem.Expression(
    ufl.conditional(
        ufl.gt(getStrainEnergy(displacement), energy_history),
        getStrainEnergy(displacement),
        energy_history,
    ),
    WW.element.interpolation_points(),
)


# Construct the weak form
f = dfx.fem.Constant(mesh, dfx.default_scalar_type((0,) * topology_dim))
displacement_a = (
    ((1.0 - crack_phase_old) ** 2) * ufl.inner(ufl.grad(v), getStress(u))
) * ufl.dx
displacement_L = ufl.dot(f, v) * ufl.dx

crack_phase_a = (
    Gc * l * ufl.inner(ufl.grad(p), ufl.grad(q))
    + ((Gc / l) + 2.0 * energy_history) * ufl.inner(p, q)
) * ufl.dx
crack_phase_L = 2.0 * energy_history * q * ufl.dx


# Boundary conditions
top = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], 1)
)
bot = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], 0)
)


bc_bot = dfx.fem.dirichletbc(
    np.array((0,) * topology_dim, dtype=dfx.default_scalar_type),
    dfx.fem.locate_dofs_topological(W, boundary_dim, bot),
    W,
)

load = dfx.fem.Constant(mesh, dfx.default_scalar_type((0.0,) * topology_dim))
bc_top = dfx.fem.dirichletbc(
    load, dfx.fem.locate_dofs_topological(W, boundary_dim, top), W
)

displacement_bcs = [bc_bot, bc_top]


def is_crack(x):
    return np.logical_and(np.less(np.abs(x[1] - 0.5), 1e-03), np.less_equal(x[0], 0.5))


crack_phase_bcs = [
    dfx.fem.dirichletbc(
        dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0)),
        dfx.fem.locate_dofs_geometrical(V, is_crack),
        V,
    )
]

# Linear problems
displacement_problem = petsc.LinearProblem(
    a=displacement_a,
    L=displacement_L,
    bcs=displacement_bcs,
    u=displacement,
    # petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
crack_phase_problem = petsc.LinearProblem(
    a=crack_phase_a,
    L=crack_phase_L,
    bcs=crack_phase_bcs,
    u=crack_phase,
    # petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)


# Output and iterative scheme
u_r = 0.007
delta_time = 0.1
tolerance = 1e-3
max_iterations = 1000


T = np.concatenate(
    (
        np.arange(0, 0.6, 0.1),
        np.arange(0.6, 0.679, 0.01),
        np.arange(0.68, 0.741, 0.002),
        np.arange(0.75, 1.01, 0.05),
    )
)


def Load(t: float) -> float:
    peak_time = 0.85
    total_time = 1
    return (
        u_r * t / peak_time
        if t <= peak_time
        else u_r * (total_time - t) / (total_time - peak_time)
    )


pvd_file = dfx.io.VTKFile(mesh.comm, "RateIndependent/results.pvd", "w")
pvd_file.write_mesh(mesh)


def getL2error(u, u_old) -> float:
    L2 = dfx.fem.form(ufl.inner(u - u_old, u - u_old) * ufl.dx)
    error_local = dfx.fem.assemble_scalar(L2)
    error = np.sqrt(mpi.COMM_WORLD.allreduce(error_local, op=mpi.SUM))

    return error


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def __str__(self):
        return f"{self.elapsed():.2e}s"


old_time = 0.0
total_timer = Timer()
for t in T:
    delta_time = t - old_time
    old_time = t

    # Update the load
    load.value[1] = Load(t)

    # Solve the problem
    iterations = 0
    residual = tolerance + 1
    timer = Timer()
    while residual > tolerance:
        iterations += 1

        displacement_problem.solve()

        energy_history.interpolate(energy_history_expr)

        crack_phase_problem.solve()

        displacement_residual = getL2error(displacement, displacement_old)
        crack_phase_residual = getL2error(crack_phase, crack_phase_old)
        residual = max(displacement_residual, crack_phase_residual)

        displacement_old.x.array[:] = displacement.x.array
        crack_phase_old.x.array[:] = crack_phase.x.array

        print(
            f"Time: {t:.3e}, Load: {load.value[1]:.3e}, Iteration: {iterations}, Residual: {residual:.3e}"
        )

        if iterations > max_iterations:
            print("Max iterations reached, simulation failed at time", t)
            pvd_file.close()

            import sys

            sys.exit(1)
    else:
        end = time.time()
        pvd_file.write_function(displacement, t)
        pvd_file.write_function(crack_phase, t)
        pvd_file.write_function(energy_history, t)
        print(
            f"Time {t:.3e} completed. Delta time: {delta_time:.3e}, Elapsed: {timer}\n"
        )


print(f"Simulation completed. Total time: {total_timer}")
pvd_file.close()
