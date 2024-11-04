import os

import ufl.integral

os.environ["OMP_NUM_THREADS"] = "1"

import dolfinx as dfx
import dolfinx.fem.petsc as petsc

from dolfinx.nls.petsc import NewtonSolver
import ufl

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

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

preset = Presets.nl_high_loading_rate
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

# %% Create the mesh
length = 100
mesh = dfx.mesh.create_rectangle(
    comm,
    [np.array([-length / 2, -length / 2]), np.array([length / 2, length / 2])],
    [preset.mesh_x, preset.mesh_y],
    cell_type=dfx.mesh.CellType.quadrilateral,
)

topology_dim = mesh.topology.dim
boundary_dim = topology_dim - 1

# %% Define function spaces
V = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim,)))
S = dfx.fem.functionspace(mesh, ("CG", 1))
DS = dfx.fem.functionspace(mesh, ("DG", 1))

# Trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f, g = ufl.TrialFunction(S), ufl.TestFunction(S)
p, q = ufl.TrialFunction(S), ufl.TestFunction(S)

# Variable functions
# Displacement
displacement = dfx.fem.Function(V)  # u_{k+1}
displacement_old = dfx.fem.Function(V)  # u_k
displacement_old2 = dfx.fem.Function(V)  # u_{k-1}

# Crack phase field
crack_phase = dfx.fem.Function(S)  # d_{k+1}
crack_phase_old = dfx.fem.Function(S)  # d_k

energy_history = dfx.fem.Function(DS)  # H_{k+1}

# Plastic field
equivalent_plastic_strain = dfx.fem.Function(S)  # \alpha_{k+1}
equivalent_plastic_strain_old = dfx.fem.Function(S)  # \alpha_{k+1}
plastic_work = dfx.fem.Function(S)  # W_{k+1}

displacement.name = "Displacement"
crack_phase.name = "Crack_Phase"
energy_history.name = "Energy_History"
equivalent_plastic_strain.name = "Equivalent_Plastic_Strain"

# %% Define constitutive relations
energy_history_expr = dfx.fem.Expression(
    ufl.conditional(
        ufl.gt(
            constitutive.getStrainEnergyPositive(displacement, crack_phase)
            + plastic_work
            - dfx.fem.Constant(mesh, dfx.default_scalar_type(material.wc)),
            energy_history,
        ),
        constitutive.getStrainEnergyPositive(displacement, crack_phase)
        + plastic_work
        - dfx.fem.Constant(mesh, dfx.default_scalar_type(material.wc)),
        energy_history,
    ),
    DS.element.interpolation_points(),
)


scalar_zero = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))
plastic_related_object_expr = dfx.fem.Expression(
    ufl.inner(
        ufl.dev(constitutive.getStress(displacement, scalar_zero)),
        ufl.dev(constitutive.getStress(displacement, scalar_zero)),
    ),
    S.element.interpolation_points(),
)

plastic_work.x.array[:] = 0.0


V_vis = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim,)))
S_vis = dfx.fem.functionspace(mesh, ("CG", 1))

displacement_vis = dfx.fem.Function(V_vis)
crack_phase_vis = dfx.fem.Function(S_vis)
energy_history_vis = dfx.fem.Function(S_vis)
equivalent_plastic_strain_vis = dfx.fem.Function(S_vis)

displacement_vis.name = "Displacement"
crack_phase_vis.name = "Crack Phase"
energy_history_vis.name = "Energy History"
equivalent_plastic_strain_vis.name = "Equivalent Plastic Strain"


# %% Construct the weak form
def getAcceleration(u, u_old, u_old2, dt, dt_old):
    return (
        2 / (dt_old * dt) * (dt_old * u + dt * u_old2) / (dt + dt_old)
        - 2 / (dt_old * dt) * u_old
    )


dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))
dt_old = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))

if constitutive.linear:
    U = 0.5 * (displacement + displacement_old)
    acceleration = getAcceleration(u, displacement_old, displacement_old2, dt, dt_old)
    force = dfx.fem.Constant(mesh, dfx.default_scalar_type((0,) * topology_dim))
    displacement_weak_form = (
        material.rho * ufl.inner(acceleration, v)
        + ufl.inner(constitutive.getStress(u, crack_phase), ufl.grad(v))
        - ufl.inner(force, v)
    ) * ufl.dx
    displacement_a = dfx.fem.form(ufl.lhs(displacement_weak_form))
    displacement_L = dfx.fem.form(ufl.rhs(displacement_weak_form))
else:
    acceleration = getAcceleration(
        displacement, displacement_old, displacement_old2, dt, dt_old
    )
    force = dfx.fem.Constant(mesh, dfx.default_scalar_type((0,) * topology_dim))
    displacement_weak_form = (
        material.rho * ufl.inner(acceleration, v)
        + ufl.inner(constitutive.getStress(displacement, crack_phase), ufl.grad(v))
        - ufl.inner(force, v)
    ) * ufl.dx

P = 0.5 * (f + crack_phase_old)
crack_phase_weak_form = (
    material.eta_p * (f - crack_phase_old) * g * ufl.dx
    - dt
    * (
        2 * (1 - f) * energy_history * g
        - material.Gc0 / material.lf * f * g
        - material.Gc0 * material.lf * ufl.inner(ufl.grad(f), ufl.grad(g))
    )
    * ufl.dx
)
crack_phase_a = dfx.fem.form(ufl.lhs(crack_phase_weak_form))
crack_phase_L = dfx.fem.form(ufl.rhs(crack_phase_weak_form))


def plasticObject(u):
    scalar_zero = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))
    return ufl.inner(
        ufl.dev(constitutive.getStress(u, scalar_zero)),
        ufl.dev(constitutive.getStress(u, scalar_zero)),
    )


equivalent_plastic_strain_weak_form = (
    material.eta_p
    * (equivalent_plastic_strain - equivalent_plastic_strain_old)
    * q
    * ufl.dx
    - dt
    * (
        np.sqrt(3 / 2) * plasticObject(displacement) * q
        - (
            material.hardening(equivalent_plastic_strain) * q
            + material.y0
            * material.lp**2
            * ufl.inner(ufl.grad(equivalent_plastic_strain), ufl.grad(q))
        )
    )
    * ufl.dx
)

# %% Boundary conditions
# Mechanical boundary conditions
top = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], length / 2)
)
bot = dfx.mesh.locate_entities_boundary(
    mesh, boundary_dim, lambda x: np.isclose(x[1], -length / 2)
)

load_top = dfx.fem.Constant(
    mesh,
    dfx.default_scalar_type((0.0,) * topology_dim),
)
bc_top = dfx.fem.dirichletbc(
    load_top, dfx.fem.locate_dofs_topological(V, boundary_dim, top), V
)
load_bot = dfx.fem.Constant(
    mesh,
    dfx.default_scalar_type((0.0,) * topology_dim),
)
bc_bot = dfx.fem.dirichletbc(
    load_bot, dfx.fem.locate_dofs_topological(V, boundary_dim, bot), V
)

displacement_bcs = [bc_bot, bc_top]


# Crack phase boundary conditions
def is_crack(x):
    y_limit = (length / 2 / preset.mesh_y) * 1.01
    return np.logical_and(
        np.less(np.abs(x[1]), y_limit),
        np.less_equal(x[0], -length / 2 + preset.crack_length),
    )


def is_void(x):
    r = 5.0
    c = [20.0, 20.0]
    return np.less_equal(np.sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2), r)


crack_phase_bcs = [
    dfx.fem.dirichletbc(
        dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0)),
        dfx.fem.locate_dofs_geometrical(S, is_crack),
        S,
    ),
    # dfx.fem.dirichletbc(
    #     dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0)),
    #     dfx.fem.locate_dofs_geometrical(S, is_void),
    #     S,
    # ),
]

# plastic_top = dfx.fem.Constant(
#     mesh,
#     dfx.default_scalar_type((0.0,) * topology_dim),
# )
# plastic_bot = dfx.fem.Constant(
#     mesh,
#     dfx.default_scalar_type((0.0,) * topology_dim),
# )

# equivalent_plastic_strain_bcs = [
#     dfx.fem.dirichletbc(
#         plastic_bot, dfx.fem.locate_dofs_topological(V, boundary_dim, bot), V
#     ),
#     dfx.fem.dirichletbc(
#         plastic_top, dfx.fem.locate_dofs_topological(V, boundary_dim, top), V
#     ),
# ]

# Construct linear problems
if constitutive.linear:
    displacement_problem = petsc.LinearProblem(
        a=displacement_a,
        L=displacement_L,
        bcs=displacement_bcs,
        u=displacement,
    )
else:
    displacement_problem = petsc.NonlinearProblem(
        F=displacement_weak_form,
        u=displacement,
        bcs=displacement_bcs,
    )

    displacement_solver = NewtonSolver(comm, displacement_problem)

    displacement_solver.convergence_criterion = "incremental"
    displacement_solver.rtol = 1e-8
    displacement_solver.report = True

    ksp = displacement_solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    petsc_sys = PETSc.Sys()  # type: ignore
    # For factorization prefer MUMPS, then superlu_dist, then default.
    if petsc_sys.hasExternalPackage("mumps"):
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    elif petsc_sys.hasExternalPackage("superlu_dist"):
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
    ksp.setFromOptions()

crack_phase_problem = petsc.LinearProblem(
    a=crack_phase_a,
    L=crack_phase_L,
    bcs=crack_phase_bcs,
    u=crack_phase,
)

equivalent_plastic_strain_problem = petsc.NonlinearProblem(
    F=equivalent_plastic_strain_weak_form,
    u=equivalent_plastic_strain,
    # bcs=[],
)

equivalent_plastic_strain_solver = NewtonSolver(comm, equivalent_plastic_strain_problem)

equivalent_plastic_strain_solver.convergence_criterion = "incremental"
equivalent_plastic_strain_solver.rtol = 1e-8
equivalent_plastic_strain_solver.report = True

ksp = equivalent_plastic_strain_solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
petsc_sys = PETSc.Sys()  # type: ignore
if petsc_sys.hasExternalPackage("mumps"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
elif petsc_sys.hasExternalPackage("superlu_dist"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()


# %% Output, iterative scheme and tools
if preset.save_interval is None:
    preset.save_interval = preset.num_iterations // 20
# save_interval = 1
delta_t = preset.end_t / preset.num_iterations

T = np.arange(delta_t, preset.end_t + delta_t / 2, delta_t)

# warp_factor = 10 / preset.u_r
warp_factor = 0


def getLoad(t):
    return t / preset.end_t * preset.u_r


def getMaxMin(u: dfx.fem.Function, u_old: dfx.fem.Function):
    u_max = comm.allreduce(np.max(u.x.array), op=MPI.MAX)
    u_min = comm.allreduce(np.min(u.x.array), op=MPI.MIN)

    delta_u_max = comm.allreduce(np.max(u.x.array - u_old.x.array), op=MPI.MAX)
    delta_u_min = comm.allreduce(np.min(u.x.array - u_old.x.array), op=MPI.MIN)

    return u_max, u_min, delta_u_max, delta_u_min


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
            title="Dynamic Crack Phase Field",
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
    "normalize": Timer(),
    "verbose": Timer(),
    "plot": Timer(),
    "update": Timer(),
    "save": Timer(),
}

for _, t in timers.items():
    t.reset()
    t.pause()

for idx, t in enumerate(T):
    if idx == 0:
        delta_time_old = t - time_old
    else:
        delta_time_old = delta_time
    # delta_time_old = delta_time
    delta_time = t - time_old
    time_old = t

    dt.value = delta_time
    dt_old.value = delta_time_old

    # print(f"dt: {dt.value:.3e}, dt_old: {dt_old.value:.3e}")

    # Update the load
    load_top.value[preset.load_direction] = getLoad(t)
    load_bot.value[preset.load_direction] = -getLoad(t)

    # Solve the problem
    timers["displacement_solve"].resume()
    if constitutive.linear:
        displacement_problem.solve()
    else:
        num_its_disp, converged = displacement_solver.solve(displacement)
        assert converged, "Displacement solver did not converge"
    timers["displacement_solve"].pause()

    timers["energy_history"].resume()
    energy_history.interpolate(energy_history_expr)
    # stress.interpolate(stress_expr)
    timers["energy_history"].pause()

    timers["crack_phase_solve"].resume()
    crack_phase_problem.solve()
    timers["crack_phase_solve"].pause()

    num_its_peeq, converged = equivalent_plastic_strain_solver.solve(
        equivalent_plastic_strain
    )
    assert converged, "Equivalent plastic strain solver did not converge"

    timers["normalize"].resume()
    crack_phase.x.array[:] = np.clip(crack_phase.x.array, crack_phase_old.x.array, 1)
    equivalent_plastic_strain.x.array[:] = np.maximum(
        equivalent_plastic_strain.x.array, equivalent_plastic_strain_old.x.array
    )
    timers["normalize"].pause()

    comm.Barrier()

    timers["verbose"].resume()
    if preset.verbose:
        u_tuple = getMaxMin(displacement, displacement_old)
        d_tuple = getMaxMin(crack_phase, crack_phase_old)
        p_tuple = getMaxMin(equivalent_plastic_strain, equivalent_plastic_strain_old)
        if rank == host:
            print(f"Time: {t:.3e}, Load: {getLoad(t):.3e}, δt: {dt.value:.3e}")
            if not constitutive.linear:
                print(
                    f"  Nonlinear displacement problem solver converged in {num_its_disp} iterations"
                )
            print(
                f"  Nonlinear equivalent plastic strain solver converged in {num_its_peeq} iterations"
            )
            print(
                f"  u max/min: {u_tuple[0]:.2e}/{u_tuple[1]:.2e}, δu max/min: {u_tuple[2]:.2e}/{u_tuple[3]:.2e}"
            )
            print(
                f"  d max/min: {d_tuple[0]:.2e}/{d_tuple[1]:.2e}, δd max/min: {d_tuple[2]:.2e}/{d_tuple[3]:.2e}"
            )
            print(
                f"  p max/min: {p_tuple[0]:.2e}/{p_tuple[1]:.2e}, δp max/min: {p_tuple[2]:.2e}/{p_tuple[3]:.2e}"
            )
            sys.stdout.flush()
    timers["verbose"].pause()

    if preset.animation or preset.out_vtk or preset.out_xdmf:
        displacement_vis.interpolate(displacement)
        crack_phase_vis.interpolate(crack_phase)
        energy_history_vis.interpolate(energy_history)
        equivalent_plastic_strain_vis.interpolate(equivalent_plastic_strain)

    timers["plot"].resume()
    if preset.animation and have_pyvista:
        # crack_phase_vis.interpolate(eq_stress)
        warped["Crack Phase"][:] = crack_phase_vis.x.array
        # warped["Displacement"][:] = displacement_vis.x.array

        grid["Displacement"][:, :topology_dim] = displacement_vis.x.array.reshape(
            points_num, topology_dim
        )

        warp_ = grid.warp_by_vector("Displacement", factor=warp_factor)
        warped.points[:, :] = warp_.points

        if rank == host:
            for i in range(1, comm.Get_size()):
                warp_ = comm.recv(source=i)
                warps[i - 1]["Crack Phase"][:] = warp_["Crack Phase"]
                # warps[i - 1]["Displacement"][:] = warp_["Displacement"]
                warps[i - 1].points[:, :] = warp_.points
            plotter.add_text(f"Time: {t:.3e}", font_size=12, name="time_label")
            plotter.app.processEvents()

            if preset.screenshot:
                plotter.screenshot(screenshot_dir / f"{idx+1}.tiff")
        else:
            comm.send(warped, dest=host)
        comm.Barrier()
    timers["plot"].pause()

    timers["update"].resume()
    displacement_old2.x.array[:] = displacement_old.x.array
    displacement_old.x.array[:] = displacement.x.array
    crack_phase_old.x.array[:] = crack_phase.x.array
    plastic_work += (
        0.5
        * (
            material.hardening(equivalent_plastic_strain)
            + material.hardening(equivalent_plastic_strain_old)
        )
        * (equivalent_plastic_strain - equivalent_plastic_strain_old)
    )
    equivalent_plastic_strain_old.x.array[:] = equivalent_plastic_strain.x.array
    timers["update"].pause()

    timers["save"].resume()
    if idx == 0 or (idx + 1) % preset.save_interval == 0 or (idx + 1) == len(T):
        if preset.out_vtk:
            pvd_file.write_function(displacement_vis, t)
            pvd_file.write_function(crack_phase_vis, t)
            pvd_file.write_function(energy_history_vis, t)
            pvd_file.write_function(equivalent_plastic_strain_vis, t)
        if preset.out_xdmf:
            xdmf_file.write_function(displacement_vis, t)
            xdmf_file.write_function(crack_phase_vis, t)
            xdmf_file.write_function(energy_history_vis, t)
            xdmf_file.write_function(equivalent_plastic_strain_vis, t)
        if rank == host:
            print(f"Saved at {t:.3e}. Elapsed: {timer}, total elapsed: {total_timer}\n")
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
