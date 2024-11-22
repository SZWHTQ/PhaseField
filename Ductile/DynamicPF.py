import os

import ufl.integral

os.environ["OMP_NUM_THREADS"] = "1"

import dolfinx as dfx
import dolfinx.fem.petsc as petsc

import gmsh

from dolfinx.nls.petsc import NewtonSolver
import ufl

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
# from slepc4py import SLEPc # module for solving eigenvalue problems, but no use for my case

try:
    import pyvista as pv
    import pyvistaqt as pvqt

    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualize the solution")
    have_pyvista = False

import time
import sys
from pathlib import Path

import Presets

preset = Presets.johnson_cook
material = preset.material
constitutive = preset.constitutive

out_dir = Path(preset.output_directory)

# MPI.Init()

host = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == host:
    print(f"MPI size: {comm.Get_size()}")
    print(f"Using {constitutive}")

    sys.stdout.flush()

    dfx.log.set_output_file(str(out_dir / "solve.log"))
    dfx.log.set_log_level(dfx.log.LogLevel.INFO)

# %% Create the mesh
# mesh = dfx.mesh.create_rectangle(
#     comm,
#     [np.array([-preset.w / 2, -preset.h / 2]), np.array([preset.w / 2, preset.h / 2])],
#     [preset.mesh_x, preset.mesh_y],
#     cell_type=dfx.mesh.CellType.quadrilateral,
# )

gmsh.initialize()
if rank == host:
    # Define the geometry
    h_e = preset.w / preset.mesh_x
    # crack_length_factor = preset.crack_length / length
    width = h_e * 0.5
    max_element_size = h_e
    min_element_size = h_e
    gmsh.model.occ.addPoint(-preset.w / 2, -preset.h / 2, 0, max_element_size, 1)
    gmsh.model.occ.addPoint(preset.w / 2, -preset.h / 2, 0, max_element_size, 2)
    gmsh.model.occ.addPoint(preset.w / 2, preset.h / 2, 0, max_element_size, 3)
    gmsh.model.occ.addPoint(-preset.w / 2, preset.h / 2, 0, max_element_size, 4)

    gmsh.model.occ.addPoint(-preset.crack_length / 2, width / 2, 0, min_element_size, 5)
    gmsh.model.occ.addPoint(preset.crack_length / 2, width / 2, 0, min_element_size, 6)
    gmsh.model.occ.addPoint(preset.crack_length / 2, -width / 2, 0, min_element_size, 7)
    gmsh.model.occ.addPoint(
        -preset.crack_length / 2, -width / 2, 0, min_element_size, 8
    )

    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)
    gmsh.model.occ.addLine(3, 4, 3)
    gmsh.model.occ.addLine(4, 1, 4)

    gmsh.model.occ.addLine(5, 6, 5)
    # gmsh.model.occ.addLine(6, 7, 6)
    gmsh.model.occ.addCircle(
        preset.crack_length / 2,
        0,
        0,
        r=width / 2,
        tag=6,
        angle1=-np.pi / 2,
        angle2=np.pi / 2,
    )
    gmsh.model.occ.addLine(7, 8, 7)
    # gmsh.model.occ.addLine(8, 5, 8)
    gmsh.model.occ.addCircle(
        -preset.crack_length / 2,
        0,
        0,
        r=width / 2,
        tag=8,
        angle1=np.pi / 2,
        angle2=-np.pi / 2,
    )

    gmsh.model.occ.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.occ.addCurveLoop([5, 6, 7, 8], 2)

    gmsh.model.occ.addPlaneSurface([1, 2], 1)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [1], 1, "Plane")

    # # Define a field to refine the mesh within the specified rectangular area
    # field_id = gmsh.model.mesh.field.add("Box")
    # gmsh.model.mesh.field.setNumber(
    #     field_id, "VIn", min_element_size
    # )  # Smaller element size within the box
    # gmsh.model.mesh.field.setNumber(
    #     field_id, "VOut", max_element_size
    # )  # Larger element size outside the box
    # gmsh.model.mesh.field.setNumber(field_id, "XMin", -0.05)
    # gmsh.model.mesh.field.setNumber(field_id, "XMax", 0.5)
    # gmsh.model.mesh.field.setNumber(field_id, "YMin", -0.1)
    # gmsh.model.mesh.field.setNumber(field_id, "YMax", 0.1)

    # gmsh.model.mesh.field.setAsBackgroundMesh(field_id)

    # Configure the meshing algorithm
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.removeDuplicateNodes()

    gmsh.model.mesh.optimize("Netgen")

    if preset.dim == 3:
        gmsh.model.occ.extrude([(2, 1)], 0, 0, preset.thickness, [1], recombine=True)
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(3, [1], 1, "Model")

        gmsh.model.mesh.generate(3)

    # gmsh.write("mesh.inp")
    # gmsh.write("mesh.msh")

    # gmsh.fltk.run()

mesh, _, _ = dfx.io.gmshio.model_to_mesh(gmsh.model, comm, host, gdim=preset.dim)
gmsh.finalize()

topology_dim = mesh.topology.dim
boundary_dim = topology_dim - 1

if rank == host:
    print(f"Dimension: {topology_dim}")
    assert topology_dim == preset.dim, "Dimension mismatch"

# %% Define function spaces
W = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim, topology_dim)))
V = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim,)))
S = dfx.fem.functionspace(mesh, ("CG", 1))
DW = dfx.fem.functionspace(mesh, ("DG", 1, (topology_dim, topology_dim)))
DV = dfx.fem.functionspace(mesh, ("DG", 1, (topology_dim,)))
DS = dfx.fem.functionspace(mesh, ("DG", 1))

# Trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(S), ufl.TestFunction(S)
d, f = ufl.TrialFunction(S), ufl.TestFunction(S)
t, r = ufl.TrialFunction(S), ufl.TestFunction(S)
a, m = ufl.TrialFunction(DV), ufl.TestFunction(DV)

# Variable functions
# Displacement
displacement_inc = dfx.fem.Function(V)  # du_{k+1}
displacement_inc_old = dfx.fem.Function(V)  # du_k
displacement_inc_old2 = dfx.fem.Function(V)  # du_{k-1}
displacement_inc_old3 = dfx.fem.Function(V)  # du_{k-2}

displacement = dfx.fem.Function(V)  # u_{k+1}
velocity = dfx.fem.Function(V)  # v_{k+1}
acceleration = dfx.fem.Function(V)  # a_{k+1}
acceleration_inc = dfx.fem.Function(V)  # da_{k+1}

strain = dfx.fem.Function(W)  # \epsilon_{k+1}
total_strain = dfx.fem.Function(W)  # \epsilon_{k+1}
stress = dfx.fem.Function(W)  # \sigma_{k+1}
stress_old = dfx.fem.Function(W)  # \sigma_{k}
deviatoric_stress = dfx.fem.Function(W)  # \sigma_{k+1}

eq_stress = dfx.fem.Function(S)  # \sigma_{eq,k+1}

# strain_energy = dfx.fem.Function(S)  # \psi^e_{k+1}

strain_rate = dfx.fem.Function(DS)  # \dot{\alpha}_{k+1}
stress_rate = dfx.fem.Function(W)  # \dot{\sigma}_{k+1}
spin = dfx.fem.Function(W)  # \W_{k+1}
jaumann_rate = dfx.fem.Function(W)

yield_stress_new = dfx.fem.Function(DS)
yield_stress = dfx.fem.Function(DS)

hardening_modulus = dfx.fem.Function(DS)

plastic_strain = dfx.fem.Function(DW)  # \epsilon_{p,k+1}
plastic_strain_inc = dfx.fem.Function(DW)  # \epsilon_{p,k+1}

eq_plastic_strain = dfx.fem.Function(DS)  # \alpha_{k+1}
eq_plastic_strain_inc = dfx.fem.Function(DS)  # d\alpha_{k+1}

strain_energy_positive = dfx.fem.Function(S)  # \psi^e_{k+1}

plastic_work = dfx.fem.Function(DS)  # W^p_{k+1}
plastic_work_inc = dfx.fem.Function(DS)  # dW^p_{k+1}


# Crack phase field
crack_phase = dfx.fem.Function(S)  # d_{k+1}
crack_phase_old = dfx.fem.Function(S)  # d_k

energy_history = dfx.fem.Function(DS)  # H_{k+1}

# Temperature
temperature = dfx.fem.Function(S)  # T_{k+1}
temperature_old = dfx.fem.Function(S)  # T_k

if preset.magnetic:
    magnetic_potential = dfx.fem.Function(DV)  # A_{k+1}

    current = dfx.fem.Function(DV)  # J_{k+1}

# equilibrium_expr = dfx.fem.Expression(

total_strain_expr = dfx.fem.Expression(
    constitutive.getStrain(displacement),
    W.element.interpolation_points(),
)


# %% Define constitutive relations
def fractureDriveForce():
    # strain_energy_positive = constitutive.getStrainEnergyPositive(strain)
    return strain_energy_positive + (1 - material.inelastic_heat) * (
        plastic_work - dfx.fem.Constant(mesh, material.w0)
    )


energy_history_expr = dfx.fem.Expression(
    ufl.conditional(
        ufl.gt(
            fractureDriveForce(),
            energy_history,
        ),
        fractureDriveForce(),
        energy_history,
    ),
    DS.element.interpolation_points(),
)

strain.x.array[:] = 0.0
stress.x.array[:] = 0.0
eq_stress.x.array[:] = 0.0
plastic_strain.x.array[:] = 0.0
eq_plastic_strain.x.array[:] = 0.0
plastic_work.x.array[:] = 0.0

velocity.x.array[:] = 0.0
acceleration.x.array[:] = 0.0

energy_history.x.array[:] = 0.0

temperature.x.array[:] = material.reference_temperature
temperature_old.x.array[:] = material.reference_temperature

# print(f"Element number: {len(mesh.cells)}")

# if rank==host:
#     print(f"Element number: {len(displacement.x.array)}")
#     print(f"Element number: {len(crack_phase.x.array)}")
#     print(f"Element number: {len(strain.x.array)}")
#     print(len(mesh.basix_cell))

W_vis = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim, topology_dim)))
V_vis = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim,)))
S_vis = dfx.fem.functionspace(mesh, ("CG", 1))

displacement_vis = dfx.fem.Function(V_vis)
strain_vis = dfx.fem.Function(W_vis)
stress_vis = dfx.fem.Function(W_vis)
strain_rate_vis = dfx.fem.Function(S_vis)
spin_vis = dfx.fem.Function(W_vis)
jaumann_rate_vis = dfx.fem.Function(W_vis)
eq_stress_vis = dfx.fem.Function(S_vis)
yield_stress_vis = dfx.fem.Function(S_vis)
total_strain_vis = dfx.fem.Function(W_vis)
plastic_strain_vis = dfx.fem.Function(W_vis)
eq_plastic_strain_vis = dfx.fem.Function(S_vis)
strain_energy_positive_vis = dfx.fem.Function(S_vis)
plastic_work_vis = dfx.fem.Function(S_vis)
crack_phase_vis = dfx.fem.Function(S_vis)
energy_history_vis = dfx.fem.Function(S_vis)
temperature_vis = dfx.fem.Function(S_vis)

# strain.name = "Strain"
# von_mises.name = "Equivalent Stress"

displacement_vis.name = "Displacement"
strain_vis.name = "Strain"
stress_vis.name = "Stress"
strain_rate_vis.name = "Strain Rate"
spin_vis.name = "Spin"
jaumann_rate_vis.name = "Jaumann Rate"
eq_stress_vis.name = "Equivalent Stress"
yield_stress_vis.name = "Yield Stress"
total_strain_vis.name = "Total Strain"
plastic_strain_vis.name = "Plastic Strain"
eq_plastic_strain_vis.name = "Equivalent Plastic Strain"
strain_energy_positive_vis.name = "Strain Energy Positive"
plastic_work_vis.name = "Plastic Work"
crack_phase_vis.name = "Crack Phase"
energy_history_vis.name = "Energy History"
temperature_vis.name = "Temperature"


# %% Construct the weak form
def getAcceleration(u, u_old, u_old2, dt, dt_old):
    return (
        2 / (dt_old * dt) * (dt_old * u + dt * u_old2) / (dt + dt_old)
        - 2 / (dt_old * dt) * u_old
    )


def getAccelerationInc(
    du,
    du_old,
    du_old2,
    du_old3,
    dt,
):
    return (2 * du - 5 * du_old + 4 * du_old2 - du_old3) / dt**2


dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))
dt_old = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))

force_inc = dfx.fem.Constant(mesh, dfx.default_scalar_type((0,) * topology_dim))
force = dfx.fem.Constant(mesh, dfx.default_scalar_type((0,) * topology_dim))


def getDisplacementWeakForm(du):
    return (
        (
            material.rho * 2 * ufl.dot(du, v)
            + dt**2
            * ufl.inner(constitutive.getStress(du, crack_phase), ufl.nabla_grad(v))
            - (
                material.rho
                * ufl.dot(
                    5 * displacement_inc_old
                    - 4 * displacement_inc_old2
                    + displacement_inc_old3,
                    v,
                )
                + dt**2 * ufl.dot(force_inc, v)
            )
        )
        # + dt**2
        # * (
        #     material.rho * ufl.dot(acceleration, v)
        #     + ufl.inner(stress, ufl.nabla_grad(v))
        #     - ufl.dot(force, v)
        # )
        * ufl.dx
    )
    # return (
    #     material.rho * ufl.inner(a, v)
    #     + ufl.inner(constitutive.getStress(displacement, crack_phase), ufl.grad(v))
    #     - ufl.inner(f, v)
    # ) * ufl.dx


if constitutive.linear:
    displacement_weak_form = getDisplacementWeakForm(u)
    displacement_a = dfx.fem.form(ufl.lhs(displacement_weak_form))
    displacement_L = dfx.fem.form(ufl.rhs(displacement_weak_form))
else:
    displacement_weak_form = getDisplacementWeakForm(displacement_inc)

P = 0.5 * (d + crack_phase_old)
crack_phase_weak_form = (
    material.eta_f * (d - crack_phase_old) * f
    - dt
    * (
        2 * (1 - d) * energy_history * f
        - material.Gc / material.lf * d * f
        - material.Gc * material.lf * ufl.dot(ufl.nabla_grad(d), ufl.nabla_grad(f))
        + 2
        * material.w0
        * (d * f + material.lf**2 * ufl.dot(ufl.nabla_grad(d), ufl.nabla_grad(f)))
    )
) * ufl.dx
# crack_phase_weak_form = (
#     material.eta_f * (f - crack_phase_old) * g * ufl.dx
#     - dt
#     * (
#         (1 - f) * energy_history * g
#         - f * g
#         - material.lf**2 * ufl.dot(ufl.nabla_grad(f), ufl.nabla_grad(g))
#     )
#     * ufl.dx
# )
crack_phase_a = dfx.fem.form(ufl.lhs(crack_phase_weak_form))
crack_phase_L = dfx.fem.form(ufl.rhs(crack_phase_weak_form))

temperature_weak_form = (
    material.rho * material.specific_heat * (t - temperature_old) * r
    - dt
    * (
        -((1 - crack_phase) ** 2)
        * material.heat_conduction
        * ufl.dot(ufl.nabla_grad(t), ufl.nabla_grad(r))
        + material.inelastic_heat * (plastic_work_inc / dt) * r
    )
) * ufl.dx
temperature_a = dfx.fem.form(ufl.lhs(temperature_weak_form))
temperature_L = dfx.fem.form(ufl.rhs(temperature_weak_form))

if preset.magnetic:
    magnetic_weak_form = (
        1.0 / material.permittivity * ufl.inner(ufl.nabla_grad(a), ufl.nabla_grad(m))
        - (ufl.dot(current, m))
    ) * ufl.dx
    magnetic_potential_a = dfx.fem.form(ufl.lhs(magnetic_weak_form))
    magnetic_potential_L = dfx.fem.form(ufl.rhs(magnetic_weak_form))


# %% Boundary conditions
# Mechanical boundary conditions
top = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], preset.h / 2)
)
bot = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], -preset.h / 2)
)

load_top = dfx.fem.Constant(mesh, (0.0,) * topology_dim)
bc_top = dfx.fem.dirichletbc(
    load_top,
    dfx.fem.locate_dofs_topological(V, boundary_dim, top),
    V,
)
load_bot = dfx.fem.Constant(mesh, (0.0,) * topology_dim)
bc_bot = dfx.fem.dirichletbc(
    load_bot,
    dfx.fem.locate_dofs_topological(V, boundary_dim, bot),
    V,
)

displacement_bcs = [bc_bot, bc_top]


# # Crack phase boundary conditions
# def is_crack(x):
#     y_limit = (preset.h / 2 / preset.mesh_y) * 1.01
#     return np.logical_and(
#         np.less(np.abs(x[1]), y_limit),
#         np.less_equal(np.abs(x[0]), preset.crack_length / 2),
#         # np.less_equal(x[0], -preset.w / 4),
#     )


# # def is_void(x):
# #     r = 5.0
# #     c = [20.0, 20.0]
# #     return np.less_equal(np.sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2), r)


# crack_phase_bcs = [
#     dfx.fem.dirichletbc(
#         dfx.fem.Constant(mesh, 1.0),
#         dfx.fem.locate_dofs_geometrical(S, is_crack),
#         S,
#     ),
#     # dfx.fem.dirichletbc(
#     #     dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0)),
#     #     dfx.fem.locate_dofs_geometrical(S, is_void),
#     #     S,
#     # ),
# ]

# Construct problems
if constitutive.linear:
    displacement_problem = petsc.LinearProblem(
        a=displacement_a,
        L=displacement_L,
        bcs=displacement_bcs,
        u=displacement_inc,
    )
else:
    displacement_problem = petsc.NonlinearProblem(
        F=displacement_weak_form,
        u=displacement_inc,
        bcs=displacement_bcs,
    )

    displacement_solver = NewtonSolver(comm, displacement_problem)

    displacement_solver.atol = 1e-8
    displacement_solver.rtol = 1e-8
    displacement_solver.convergence_criterion = "incremental"
    displacement_solver.report = True

    # ksp = displacement_solver.krylov_solver
    # opts = PETSc.Options()
    # option_prefix = ksp.getOptionsPrefix()
    # opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}pc_type"] = "lu"
    # petsc_sys = PETSc.Sys()  # type: ignore
    # # For factorization prefer MUMPS, then superlu_dist, then default.
    # if petsc_sys.hasExternalPackage("mumps"):
    #     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    # elif petsc_sys.hasExternalPackage("superlu_dist"):
    #     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
    # ksp.setFromOptions()

crack_phase_problem = petsc.LinearProblem(
    a=crack_phase_a,
    L=crack_phase_L,
    # bcs=crack_phase_bcs,
    u=crack_phase,
)

temperature_problem = petsc.LinearProblem(
    a=temperature_a,
    L=temperature_L,
    bcs=[],
    u=temperature,
)

if preset.magnetic:
    magnetic_problem = petsc.LinearProblem(
        a=magnetic_potential_a,
        L=magnetic_potential_L,
        bcs=[],
        u=magnetic_potential,
    )

# %% Output, iterative scheme and tools
if preset.save_interval is None:
    preset.save_interval = preset.num_iterations // 50
# save_interval = 1
delta_t = preset.end_t / preset.num_iterations

T = np.arange(delta_t, preset.end_t + delta_t / 2, delta_t)

# warp_factor = 10 / preset.u_r
warp_factor = preset.warp_factor


def getLoad(time):
    def smooth(xi):
        return xi**3 * (10 - 15 * xi + 6 * xi * xi)

    return preset.u_r * smooth(time / preset.end_t)


def getMaxMin(u: dfx.fem.Function):
    u_max = comm.allreduce(np.max(u), op=MPI.MAX)
    u_min = comm.allreduce(np.min(u), op=MPI.MIN)

    return u_max, u_min


def check_nan(name, array: list):
    nan_indice = np.where(np.isnan(array))
    if len(nan_indice[0]) > 0:
        print(f"NaN found at {nan_indice} in {name}")
        return True
    return


class Timer:
    def __init__(self, name="Timer"):
        self.name = name
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def reset(self):
        self.start = time.time()

    def __str__(self):
        elapsed = self.elapsed()
        if elapsed < 0:
            return "Negative time"
        elif elapsed < 1:
            return f"{elapsed:.2e}s"
        elif elapsed < 60:
            return f"{elapsed:.2f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes:d}m{seconds:.2f}s"
        elif elapsed < 86400:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            return f"{hours:d}h{minutes:d}m{seconds:.2f}s"
        else:
            days = int(elapsed // 86400)
            hours = int((elapsed % 86400) // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            return f"{days:d}d{hours:d}h{minutes:d}m{seconds:.2f}s"

    def pause(self):
        self.pause_time = time.time()

    def resume(self):
        if hasattr(self, "pause_time"):
            self.start += time.time() - self.pause_time
        else:
            print("Timer was not paused")


if preset.out_vtk:
    pvd_file = dfx.io.VTKFile(comm, out_dir / "ParaView/results.pvd", "w")
if preset.out_xdmf:
    xdmf_file = dfx.io.XDMFFile(comm, out_dir / "XDMF/results.xdmf", "w")
    xdmf_file.write_mesh(mesh)

# %% Create visualization
if preset.animation and have_pyvista:
    displacement_vis.interpolate(displacement)
    crack_phase_vis.interpolate(crack_phase)
    # eq_plastic_strain_vis.interpolate(eq_plastic_strain)

    points_num = mesh.geometry.x.shape[0]

    grid = pv.UnstructuredGrid(*dfx.plot.vtk_mesh(mesh))
    # eq_stress = dfx.fem.Expression(
    #     constitutive.getEquivalentStress(displacement, crack_phase),
    #     V_vis.element.interpolation_points(),
    # )
    # crack_phase_vis.interpolate(eq_stress)
    grid["Crack Phase"] = crack_phase_vis.x.array
    # grid["Displacement"] = displacement_vis.x.array
    grid.set_active_scalars("Crack Phase")
    # grid.set_active_scalars("Displacement")

    values = np.zeros((points_num, 3))
    values[:, :topology_dim] = displacement_vis.x.array.reshape(
        points_num, topology_dim
    )
    grid["Displacement"] = values

    warped = grid.warp_by_vector("Displacement", factor=warp_factor)

    if rank == host:
        plotter = pvqt.BackgroundPlotter(
            title="Crack Phase Field",
            auto_update=True,  # , window_size=(1600, 1120)
        )

        plotter.add_mesh(warped, show_edges=False, clim=[0, 1], cmap="coolwarm")
        warps = []
        for i in range(1, comm.Get_size()):
            warp_ = comm.recv(source=i)
            warps.append(warp_)
            plotter.add_mesh(warps[i - 1], clim=[0, 1], cmap="coolwarm")

        plotter.view_xy(False)
        plotter.add_text("Time: 0.0", font_size=12, name="time_label")

        axes = pv.Axes(show_actor=True, actor_scale=10, line_width=5)
        axes.origin = (-7, -5.5, 0)
        plotter.add_actor(axes.actor)
        plotter.app.processEvents()

        if preset.screenshot:
            screenshot_dir = out_dir / "Screenshots"
            if not screenshot_dir.exists():
                screenshot_dir.mkdir(parents=True)
            plotter.screenshot(screenshot_dir / "initial.tiff")
    else:
        comm.send(warped, dest=host)
    comm.Barrier()

# %% Solve the problem
time_old = 0.0
delta_time = 0.0
delta_time_old = 0.0
timer = Timer()
total_timer = Timer()

timers = {
    "displacement_solve": Timer(),
    "energy_history": Timer(),
    "crack_phase_solve": Timer(),
    "temperature_solve": Timer(),
    "magnetic_solve": Timer(),
    "normalize": Timer(),
    "verbose": Timer(),
    "plot": Timer(),
    "update": Timer(),
    "save": Timer(),
}

for _, _t in timers.items():
    _t.reset()
    _t.pause()

load = 0
load_old = 0
# equilibrium = np.zeros(len())
for iteration, current_time in enumerate(T):
    if iteration == 0:
        delta_time_old = current_time - time_old
    else:
        delta_time_old = delta_time
    # delta_time_old = delta_time
    delta_time = current_time - time_old
    time_old = current_time

    dt.value = delta_time
    dt_old.value = delta_time_old

    # print(f"dt: {dt.value:.3e}, dt_old: {dt_old.value:.3e}")

    # Update the load
    load = getLoad(current_time)
    load_top.value[preset.load_direction] = load - load_old
    load_bot.value[preset.load_direction] = load_old - load
    load_old = load

    # Solve the problem
    timers["displacement_solve"].resume()
    # is_equilibrium = np.all(np.less_equal(1e-8 * np.linalg, ))
    if True:
        if constitutive.linear:
            displacement_problem.solve()
        else:
            num_its_disp, converged = displacement_solver.solve(displacement_inc)
            assert converged, "Displacement solver did not converge"

        displacement.x.array[:] += displacement_inc.x.array[:]
        total_strain.interpolate(total_strain_expr)

        if iteration != 0:
            get_yield_stress_expr = dfx.fem.Expression(
                material.getYieldStress(
                    crack_phase, eq_plastic_strain, strain_rate, temperature
                ),
                DS.element.interpolation_points(),
            )
            get_hardening_modulus_expr = dfx.fem.Expression(
                material.getHardeningModulus(
                    crack_phase, eq_plastic_strain, strain_rate, temperature
                ),
                DS.element.interpolation_points(),
            )
            yield_stress.interpolate(get_yield_stress_expr)
            hardening_modulus.interpolate(get_hardening_modulus_expr)
            # check_nan("yield_stress", yield_stress.x.array)
            # check_nan("hardening_modulus", hardening_modulus.x.array)

            deviatoric_stress_expr = dfx.fem.Expression(
                ufl.dev(stress + constitutive.getStress(displacement_inc, crack_phase)),
                W.element.interpolation_points(),
            )
            deviatoric_stress.interpolate(deviatoric_stress_expr)
            eq_stress_expr = dfx.fem.Expression(
                ufl.sqrt(1.5 * ufl.inner(deviatoric_stress, deviatoric_stress)),
                S.element.interpolation_points(),
            )
            eq_stress.interpolate(eq_stress_expr)
            # check_nan("von_mises", von_mises.x.array)

            eq_plastic_strain_inc_expr = dfx.fem.Expression(
                ufl.conditional(
                    ufl.lt(eq_stress, yield_stress),
                    dfx.fem.Constant(mesh, 0.0),
                    (eq_stress - yield_stress)
                    / (3.0 * material.mu + hardening_modulus),
                ),
                DS.element.interpolation_points(),
            )
            eq_plastic_strain_inc.interpolate(eq_plastic_strain_inc_expr)
            eq_plastic_strain.x.array[:] += eq_plastic_strain_inc.x.array[:]
            # check_nan("eq_plastic_strain_inc", eq_plastic_strain_inc.x.array)
            # check_nan("eq_plastic_strain", eq_plastic_strain.x.array)

            plastic_strain_inc_expr = dfx.fem.Expression(
                ufl.conditional(
                    ufl.gt(eq_stress, dfx.fem.Constant(mesh, 0.0)),
                    np.sqrt(1.5) * eq_plastic_strain_inc / eq_stress,
                    dfx.fem.Constant(mesh, 0.0),
                )
                * deviatoric_stress,
                DW.element.interpolation_points(),
            )
            plastic_strain_inc.interpolate(plastic_strain_inc_expr)
            plastic_strain.x.array[:] += plastic_strain_inc.x.array[:]
            # check_nan("plastic_strain_inc", plastic_strain_inc.x.array)
            # check_nan("plastic_strain", plastic_strain.x.array)

            strain_update_expr = dfx.fem.Expression(
                strain + constitutive.getStrain(displacement_inc) - plastic_strain_inc,
                W.element.interpolation_points(),
            )
            # strain_update_expr = dfx.fem.Expression(
            #     total_strain - plastic_strain,
            #     W.element.interpolation_points(),
            # )
            strain.interpolate(strain_update_expr)

            strain_rate.x.array[:] = eq_plastic_strain_inc.x.array[:] / dt.value
            # check_nan("strain_rate", strain_rate.x.array)

            yield_stress_update_expr = dfx.fem.Expression(
                yield_stress + hardening_modulus * eq_plastic_strain_inc,
                DS.element.interpolation_points(),
            )
            yield_stress_new.interpolate(yield_stress_update_expr)
            # check_nan("yield_stress_new", yield_stress_new.x.array)

            # stress_regression_expr = dfx.fem.Expression(
            #     (1 - crack_phase) ** 2
            #     * (
            #         deviatoric_stress
            #         * yield_stress_new
            #         / (yield_stress_new + 3.0 * material.mu * eq_plastic_strain_inc)
            #         + ufl.tr(stress) / 3.0 * ufl.Identity(topology_dim)
            #     ),
            #     W.element.interpolation_points(),
            # )
            # stress.interpolate(stress_regression_expr)

            plastic_work_inc_expr = dfx.fem.Expression(
                0.5
                * (yield_stress_new + yield_stress)
                * eq_plastic_strain_inc
                / (1 - crack_phase) ** 2,
                DS.element.interpolation_points(),
            )
            plastic_work_inc.interpolate(plastic_work_inc_expr)
            plastic_work.x.array[:] += plastic_work_inc.x.array[:]
        # else:
        stress_update_expr = dfx.fem.Expression(
            constitutive.getStressByStrain(strain, crack_phase, dim=topology_dim),
            W.element.interpolation_points(),
        )
        stress_old.x.array[:] = stress.x.array[:]
        stress.interpolate(stress_update_expr)
        stress_rate_expr = dfx.fem.Expression(
            (stress - stress_old) / dt,
            W.element.interpolation_points(),
        )
        stress_rate.interpolate(stress_rate_expr)

    strain_energy_positive.interpolate(constitutive.getStrainEnergyPositive(strain))
    constitutive.eigens_prepared = False
    timers["displacement_solve"].pause()

    timers["update"].resume()
    # displacement.x.array[:] += displacement_inc.x.array[:]

    displacement_inc_old3.x.array[:] = displacement_inc_old2.x.array[:]
    displacement_inc_old2.x.array[:] = displacement_inc_old.x.array[:]
    displacement_inc_old.x.array[:] = displacement_inc.x.array[:]

    velocity.x.array[:] = displacement_inc.x.array[:] / dt.value
    acceleration_inc_expr = dfx.fem.Expression(
        getAccelerationInc(
            displacement_inc,
            displacement_inc_old,
            displacement_inc_old2,
            displacement_inc_old3,
            dt,
        ),
        V.element.interpolation_points(),
    )
    acceleration_inc.interpolate(acceleration_inc_expr)
    acceleration.x.array[:] += acceleration_inc.x.array[:]

    # jaumann_rate = sr.Jaumann()
    spin_expr = dfx.fem.Expression(
        0.5 * (ufl.nabla_grad(velocity) - ufl.nabla_grad(velocity).T),
        W.element.interpolation_points(),
    )
    spin.interpolate(spin_expr)
    jaumann_rate_expr = dfx.fem.Expression(
        stress_rate - ufl.dot(spin, stress) + ufl.dot(stress, spin),
        W.element.interpolation_points(),
    )
    jaumann_rate.interpolate(jaumann_rate_expr)

    objective_stress_rate_update_expr = dfx.fem.Expression(
        stress_old + jaumann_rate * dt,
        W.element.interpolation_points(),
    )
    stress.interpolate(objective_stress_rate_update_expr)

    timers["update"].pause()

    timers["energy_history"].resume()
    energy_history.interpolate(energy_history_expr)
    timers["energy_history"].pause()

    timers["crack_phase_solve"].resume()
    crack_phase_problem.solve()
    timers["crack_phase_solve"].pause()

    timers["temperature_solve"].resume()
    temperature_problem.solve()
    timers["temperature_solve"].pause()

    timers["magnetic_solve"].resume()
    if preset.magnetic:
        magnetic_problem.solve()
    timers["magnetic_solve"].pause()

    timers["normalize"].resume()
    # crack_phase.x.array[:] = np.clip(crack_phase.x.array, crack_phase_old.x.array, 1)
    crack_phase.x.array[:] = np.maximum(crack_phase.x.array, crack_phase_old.x.array)
    timers["normalize"].pause()

    comm.Barrier()

    timers["verbose"].resume()
    if preset.verbose:
        u_tuple = getMaxMin(displacement.x.array)
        du_tuple = getMaxMin(displacement_inc.x.array)
        d_tuple = getMaxMin(crack_phase.x.array)
        dd_tuple = getMaxMin(crack_phase.x.array - crack_phase_old.x.array)
        p_tuple = getMaxMin(eq_plastic_strain.x.array)
        dp_tuple = getMaxMin(eq_plastic_strain_inc.x.array)
        t_tuple = getMaxMin(temperature.x.array)
        dt_tuple = getMaxMin(temperature.x.array - temperature_old.x.array)
        if rank == host:
            print(
                f"Time: {current_time:.3e}, Load: {load:.3e}, its: {iteration}, δt: {dt.value:.3e}"
            )
            if not constitutive.linear:
                print(
                    f"  Nonlinear displacement problem solver converged in {num_its_disp} iterations"
                )
            print(
                f"  u range: {u_tuple[0]:.2e}/{u_tuple[1]:.2e}, δu range: {du_tuple[0]:.2e}/{u_tuple[1]:.2e}"
            )
            print(
                f"  d range: {d_tuple[0]:.2e}/{d_tuple[1]:.2e}, δd range: {dd_tuple[0]:.2e}/{dd_tuple[1]:.2e}"
            )
            print(
                f"  p range: {p_tuple[0]:.2e}/{p_tuple[1]:.2e}, δp range: {dp_tuple[0]:.2e}/{dp_tuple[1]:.2e}"
            )
            print(
                f"  t range: {t_tuple[0]:.2e}/{t_tuple[1]:.2e}, δt range: {dt_tuple[0]:.2e}/{dt_tuple[1]:.2e}"
            )
            sys.stdout.flush()
    timers["verbose"].pause()

    timers["update"].resume()
    crack_phase_old.x.array[:] = crack_phase.x.array[:]
    temperature_old.x.array[:] = temperature.x.array[:]
    timers["update"].pause()

    if preset.animation or preset.out_vtk or preset.out_xdmf:
        displacement_vis.interpolate(displacement)
        strain_vis.interpolate(strain)
        stress_vis.interpolate(stress)
        strain_rate_vis.interpolate(strain_rate)
        spin_vis.interpolate(spin)
        jaumann_rate_vis.interpolate(jaumann_rate)
        eq_stress_vis.interpolate(eq_stress)
        yield_stress_vis.interpolate(yield_stress_new)
        total_strain_vis.interpolate(total_strain)
        plastic_strain_vis.interpolate(plastic_strain)
        eq_plastic_strain_vis.interpolate(eq_plastic_strain)
        strain_energy_positive_vis.interpolate(strain_energy_positive)
        plastic_work_vis.interpolate(plastic_work)
        crack_phase_vis.interpolate(crack_phase)
        energy_history_vis.interpolate(energy_history)
        temperature_vis.interpolate(temperature)

    timers["plot"].resume()
    if preset.animation and have_pyvista:
        warped["Crack Phase"][:] = crack_phase_vis.x.array
        # warped["Displacement"][:] = displacement_vis.x.array

        grid["Displacement"][:, :topology_dim] = displacement_vis.x.array.reshape(
            points_num, topology_dim
        )

        warp_ = grid.warp_by_vector("Displacement", factor=warp_factor)
        warped.points[:, :] = warp_.points

        if rank != host:
            comm.send(warped, dest=host)
        else:
            for i in range(1, comm.Get_size()):
                warp_ = comm.recv(source=i)
                warps[i - 1]["Crack Phase"][:] = warp_["Crack Phase"]
                # warps[i - 1]["Displacement"][:] = warp_["Displacement"]
                warps[i - 1].points[:, :] = warp_.points
            plotter.add_text(
                f"Time: {current_time:.3e}", font_size=12, name="time_label"
            )
            plotter.app.processEvents()

            if preset.screenshot:
                plotter.screenshot(screenshot_dir / f"{iteration+1}.tiff")
        comm.Barrier()
    timers["plot"].pause()

    timers["save"].resume()
    if (
        iteration == 0
        or (iteration + 1) % preset.save_interval == 0
        or (iteration + 1) == len(T)
    ):
        if preset.out_vtk:
            pvd_file.write_function(displacement_vis, current_time)
            pvd_file.write_function(crack_phase_vis, current_time)
            pvd_file.write_function(energy_history_vis, current_time)
            pvd_file.write_function(eq_plastic_strain_vis, current_time)
            pvd_file.write_function(plastic_work_vis, current_time)
        if preset.out_xdmf:
            xdmf_file.write_function(displacement_vis, current_time)
            xdmf_file.write_function(strain_vis, current_time)
            xdmf_file.write_function(stress_vis, current_time)
            xdmf_file.write_function(strain_rate_vis, current_time)
            xdmf_file.write_function(spin_vis, current_time)
            xdmf_file.write_function(jaumann_rate_vis, current_time)
            xdmf_file.write_function(eq_stress_vis, current_time)
            xdmf_file.write_function(yield_stress_vis, current_time)
            xdmf_file.write_function(total_strain_vis, current_time)
            xdmf_file.write_function(plastic_strain_vis, current_time)
            xdmf_file.write_function(eq_plastic_strain_vis, current_time)
            xdmf_file.write_function(strain_energy_positive_vis, current_time)
            xdmf_file.write_function(plastic_work_vis, current_time)
            xdmf_file.write_function(crack_phase_vis, current_time)
            xdmf_file.write_function(energy_history_vis, current_time)
            xdmf_file.write_function(temperature_vis, current_time)
        if rank == host:
            print(
                f"Saved at {current_time:.3e}. Elapsed: {timer}, total elapsed: {total_timer}\n"
            )
            sys.stdout.flush()
            timer.reset()
    timers["save"].pause()

comm.Barrier()

if preset.out_vtk:
    pvd_file.close()
if preset.out_xdmf:
    xdmf_file.close()

if constitutive.linear:
    displacement_solver = displacement_problem.solver
    disp_viewer = PETSc.Viewer().createASCII(
        str(out_dir / "displacement_solver.txt"), "w", comm=comm
    )
    displacement_solver.view(disp_viewer)

crack_phase_solver = crack_phase_problem.solver
crack_viewer = PETSc.Viewer().createASCII(
    str(out_dir / "crack_phase_solver.txt"), "w", comm=comm
)
crack_phase_solver.view(crack_viewer)

if rank == host:
    for name, timer in timers.items():
        print(f"{name}: {timer}")
    print(f"Simulation completed. Total time: {total_timer}\n")
    if preset.animation and have_pyvista:
        plotter.close()

# MPI.Finalize()
