### This file is not correct for its mechanical part is still static.
### Result is for reference only.
### The correct one is `DynamicPF.py` in the same folder.

import numpy as np
from mpi4py import MPI as mpi
from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc as petsc

# from dolfinx.nls.petsc import NewtonSolver
import ufl

import time

# Create the mesh
length = 100
mesh = dfx.mesh.create_rectangle(
    mpi.COMM_WORLD,
    [np.array([-length / 2, -length / 2]), np.array([length / 2, length / 2])],
    [400, 400],
    cell_type=dfx.mesh.CellType.quadrilateral,
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

displacement_old.x.array[:] = 0.0
crack_phase_old.x.array[:] = 0.0
energy_history.x.array[:] = 0.0

displacement.name = "Displacement"
crack_phase.name = "Crack Phase"
energy_history.name = "Energy History"


# Define material parameters
def getLame(E, nu):
    return E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)


E = 32e3
nu = 0.2
rho = 2.45e-6
Gc = 3e-3
lc = 0.5
lame, mu = getLame(E, nu)
eta = 1e-5


# Define constitutive functions
def getStrain(u):
    return ufl.sym(ufl.grad(u))


def getStress(u):
    return 2.0 * mu * getStrain(u) + lame * ufl.tr(getStrain(u)) * ufl.Identity(len(u))


def getStrainEnergy(u):
    return 0.5 * (lame + mu) * (
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
    ((1 - crack_phase_old) ** 2) * ufl.inner(ufl.grad(v), getStress(u))
) * ufl.dx
displacement_L = ufl.inner(f, v) * ufl.dx


dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0001))
zero = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))
# crack_phase_b = (
#     2 * (1 - crack_phase_old) * energy_history * q
#     - Gc / lc * crack_phase_old * q
#     - Gc * lc * ufl.inner(ufl.nabla_grad(crack_phase_old), ufl.nabla_grad(q))
# )
# crack_phase_weak_form = (
#     eta * (p - crack_phase_old) * q * ufl.dx
#     - dt * ufl.conditional(ufl.gt(crack_phase_b, zero), crack_phase_b, zero) * ufl.dx
# )
# crack_phase_a = eta * (p - crack_phase_old) * q * ufl.dx
# crack_phase_L = (
#     dt * ufl.conditional(ufl.gt(crack_phase_b, zero), crack_phase_b, zero) * ufl.dx
# )

P = (p + crack_phase_old) * 0.5
crack_phase_b0 = (
    2 * (1 - P) * energy_history * q
    - Gc / lc * P * q
    - Gc * lc * ufl.inner(ufl.nabla_grad(P), ufl.nabla_grad(q))
)
crack_phase_b = ufl.conditional(ufl.gt(crack_phase_b0, zero), crack_phase_b0, zero)
crack_phase_weak_form = (
    eta * (p - crack_phase_old) * q * ufl.dx - dt * crack_phase_b0 * ufl.dx
)
crack_phase_a = dfx.fem.form(ufl.lhs(crack_phase_weak_form))
crack_phase_L = dfx.fem.form(ufl.rhs(crack_phase_weak_form))


# Boundary conditions
top = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], length / 2)
)
bot = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], -length / 2)
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
    return np.logical_and(np.less(np.abs(x[1]), 1e-03), np.less_equal(x[0], 0))


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
)
crack_phase_problem = petsc.LinearProblem(
    a=crack_phase_a,
    L=crack_phase_L,
    bcs=crack_phase_bcs,
    u=crack_phase,
)
# crack_phase_A = petsc.assemble_matrix(crack_phase_a, bcs=crack_phase_bcs)
# crack_phase_A.assemble()
# crack_phase_b = petsc.create_vector(crack_phase_L)

# crack_phase_solver = PETSc.KSP().create(mesh.comm)
# crack_phase_solver.setOperators(crack_phase_A)
# crack_phase_solver.setType(PETSc.KSP.Type.CG)
# crack_phase_pc = crack_phase_solver.getPC()
# crack_phase_pc.setType(PETSc.PC.Type.SOR)


# Output and iterative scheme
u_r = 0.07
delta_t = 0.01
end_t = 1
save_interval = round(0.05 / delta_t)
# delta_t = 0.002
# end_t = 0.1
# save_interval = round(0.01 / delta_t)
T = np.arange(delta_t, end_t + delta_t, delta_t)


def getLoad(t: float) -> float:
    peak_time = 0.85
    total_time = 1
    return (
        u_r * t / peak_time
        if t <= peak_time
        else u_r * (total_time - t) / (total_time - peak_time)
    )


pvd_file = dfx.io.VTKFile(mesh.comm, "RateDependent/results.pvd", "w")
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
for idx, t in enumerate(T):
    delta_time = t - old_time
    dt.value = delta_time
    old_time = t

    # Update the load
    load.value[0] = getLoad(t)

    # Solve the problem
    timer = Timer()

    residual = 1
    iterations = 0
    while residual > 1e-3:
        iterations += 1
        displacement_problem.solve()

        residual = getL2error(displacement, displacement_old)

        displacement_old.x.array[:] = displacement.x.array

    energy_history.interpolate(energy_history_expr)

    crack_phase_problem.solve()
    # with crack_phase_b.localForm() as loc_b:
    #     loc_b.set(0)
    # petsc.assemble_vector(crack_phase_b, crack_phase_L)
    # petsc.apply_lifting(crack_phase_b, [crack_phase_a], [crack_phase_bcs])
    # crack_phase_b.ghostUpdate(
    #     addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    # )
    # petsc.set_bc(crack_phase_b, crack_phase_bcs)
    # crack_phase_solver.solve(crack_phase_b, crack_phase.vector)
    # crack_phase.x.scatter_forward()

    # displacement_residual = getL2error(displacement, displacement_old)
    # crack_phase_residual = getL2error(crack_phase, crack_phase_old)
    # residual = max(displacement_residual, crack_phase_residual)
    # print(f"Residual: {residual:.3e}")

    for i in range(len(crack_phase.x.array)):
        if crack_phase.x.array[i] > 1:
            crack_phase.x.array[i] = 1
        if crack_phase.x.array[i] < crack_phase_old.x.array[i]:
            crack_phase.x.array[i] = crack_phase_old.x.array[i]
        # TODO: Shall be unnecessary, check it
        if crack_phase.x.array[i] < 0:
            crack_phase.x.array[i] = 0

    u_max = mesh.comm.allreduce(np.max(displacement.x.array), op=mpi.MAX)
    u_min = mesh.comm.allreduce(np.min(displacement.x.array), op=mpi.MIN)
    d_max = mesh.comm.allreduce(np.max(crack_phase.x.array), op=mpi.MAX)
    d_min = mesh.comm.allreduce(np.min(crack_phase.x.array), op=mpi.MIN)
    delta_d_max = mesh.comm.allreduce(
        np.max(crack_phase.x.array - crack_phase_old.x.array), op=mpi.MAX
    )
    delta_d_min = mesh.comm.allreduce(
        np.min(crack_phase.x.array - crack_phase_old.x.array), op=mpi.MIN
    )
    print(f"Time: {t:.3e}, Load: {load.value[0]:.3e}")
    print(f"Displacement Residual:{residual:.3e}, Iteration number: {iterations}")
    print(
        f"u max/min: {u_max:.2e}/{u_min:.2e}, d max/min: {d_max:.2e}/{d_min:.2e}, δd max/min: {delta_d_max:.2e}/{delta_d_min:.2e}"
    )
    crack_phase_old.x.array[:] = crack_phase.x.array

    end = time.time()
    if idx % save_interval == 0 or idx == len(T) - 1:
        pvd_file.write_function(displacement, t)
        pvd_file.write_function(crack_phase, t)
        pvd_file.write_function(energy_history, t)
        print(
            f"Time {t:.3e} completed. Delta time: {delta_time:.3e}, Elapsed: {timer}\n"
        )


print(f"Simulation completed. Total time: {total_timer}")
pvd_file.close()
