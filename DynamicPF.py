import numpy as np
from mpi4py import MPI as mpi
import dolfinx as dfx
import dolfinx.fem.petsc as petsc
import ufl

import time
import os
from pathlib import Path

import Presets as Presets

try:
    import pyvista as pv
    import pyvistaqt as pvqt

    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualize the solution")
    have_pyvista = False

preset = Presets.high_loading_rate
material = preset.material
out_dir = Path(preset.output_directory)

dfx.log.set_output_file(str(out_dir / "solve.log"))

# %% Create the mesh
length = 100
mesh = dfx.mesh.create_rectangle(
    mpi.COMM_WORLD,
    [np.array([-length / 2, -length / 2]), np.array([length / 2, length / 2])],
    [400, 400],
    cell_type=dfx.mesh.CellType.quadrilateral,
)

topology_dim = mesh.topology.dim
boundary_dim = topology_dim - 1


# %% Define function spaces
W = dfx.fem.functionspace(mesh, ("CG", 1, (topology_dim,)))
V = dfx.fem.functionspace(mesh, ("CG", 1))
WW = dfx.fem.functionspace(mesh, ("DG", 1))


# Trial and test functions
u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
p, q = ufl.TrialFunction(V), ufl.TestFunction(V)

# Variable functions
displacement = dfx.fem.Function(W)  # u_{k+1}
displacement_old = dfx.fem.Function(W)  # u_k
displacement_old2 = dfx.fem.Function(W)  # u_{k-1}

crack_phase = dfx.fem.Function(V)  # d_{k+1}
crack_phase_old = dfx.fem.Function(V)  # d_k

energy_history = dfx.fem.Function(WW)  # H_{k+1}

displacement.name = "Displacement"
crack_phase.name = "Crack Phase"
energy_history.name = "Energy History"


# %% Define constitutive relations
def getStrain(u):
    return ufl.sym(ufl.grad(u))


def getStress(u):
    return 2.0 * material.mu * getStrain(u) + material.lame * ufl.tr(
        getStrain(u)
    ) * ufl.Identity(len(u))


def getStrainEnergy(u):
    return 0.5 * (material.lame + material.mu) * (
        0.5 * (ufl.tr(getStrain(u)) + abs(ufl.tr(getStrain(u))))
    ) ** 2 + material.mu * ufl.inner(ufl.dev(getStrain(u)), ufl.dev(getStrain(u)))


# strain_energy_expr = dfx.fem.Expression(
#     getStrainEnergy(displacement), V.element.interpolation_points()
# )

energy_history_expr = dfx.fem.Expression(
    ufl.conditional(
        ufl.gt(getStrainEnergy(displacement), energy_history),
        getStrainEnergy(displacement),
        energy_history,
    ),
    WW.element.interpolation_points(),
)


# %% Construct the weak form
def getAcceleration(u, u_old, u_old2, dt, dt_old):
    return (
        2 / (dt_old * dt) * (dt_old * u + dt * u_old2) / (dt + dt_old)
        - 2 / (dt_old * dt) * u_old
    )


dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))
dt_old = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))


a = getAcceleration(u, displacement_old, displacement_old2, dt, dt_old)
f = dfx.fem.Constant(mesh, dfx.default_scalar_type((0,) * topology_dim))
displacement_weak_form = (
    material.rho * ufl.inner(a, v) * ufl.dx
    + ((1 - crack_phase_old) ** 2) * ufl.inner(getStress(u), ufl.grad(v)) * ufl.dx
    - ufl.inner(f, v) * ufl.dx
)
displacement_a = dfx.fem.form(ufl.lhs(displacement_weak_form))
displacement_L = dfx.fem.form(ufl.rhs(displacement_weak_form))


P = 0.5 * (p + crack_phase_old)
crack_phase_weak_form = (
    material.eta * (p - crack_phase_old) * q * ufl.dx
    - dt
    * (
        2 * (1 - P) * energy_history * q
        - material.Gc / material.lc * crack_phase_old * q
        - material.Gc * material.lc * ufl.inner(ufl.nabla_grad(P), ufl.nabla_grad(q))
    )
    * ufl.dx
)
crack_phase_a = dfx.fem.form(ufl.lhs(crack_phase_weak_form))
crack_phase_L = dfx.fem.form(ufl.rhs(crack_phase_weak_form))


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
    load_top, dfx.fem.locate_dofs_topological(W, boundary_dim, top), W
)
load_bot = dfx.fem.Constant(
    mesh,
    dfx.default_scalar_type((0.0,) * topology_dim),
)
bc_bot = dfx.fem.dirichletbc(
    load_bot, dfx.fem.locate_dofs_topological(W, boundary_dim, bot), W
)

displacement_bcs = [bc_bot, bc_top]


# Crack phase boundary conditions
def is_crack(x):
    return np.logical_and(
        np.less(np.abs(x[1]), 1e-03),
        np.less_equal(x[0], 0),
    )


crack_phase_bcs = [
    dfx.fem.dirichletbc(
        dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0)),
        dfx.fem.locate_dofs_geometrical(V, is_crack),
        V,
    )
]

# Construct linear problems
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


# %% Output, iterative scheme and tools
pvd_file = dfx.io.VTKFile(mesh.comm, out_dir / "ParaView/results.pvd", "w")

save_interval = preset.num_iterations // 20
delta_t = preset.end_t / preset.num_iterations

# warp_factor = 10 / preset.u_r
warp_factor = 1

getLoad = lambda t: (t / preset.end_t) * preset.u_r

T = np.arange(delta_t, preset.end_t + delta_t / 2, delta_t)


def getMaxMin(u, u_old):
    u_max = mesh.comm.allreduce(np.max(u.x.array), op=mpi.MAX)
    u_min = mesh.comm.allreduce(np.min(u.x.array), op=mpi.MIN)

    delta_u_max = mesh.comm.allreduce(np.max(u.x.array - u_old.x.array), op=mpi.MAX)
    delta_u_min = mesh.comm.allreduce(np.min(u.x.array - u_old.x.array), op=mpi.MIN)

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
        if elapsed < 1:
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


# %% Create visualization
if have_pyvista:
    points_num = mesh.geometry.x.shape[0]

    grid = pv.UnstructuredGrid(*dfx.plot.vtk_mesh(mesh))
    grid.point_data["Crack Phase"] = crack_phase.x.array[:]
    grid.set_active_scalars("Crack Phase")

    values = np.zeros((points_num, 3))
    values[:, :topology_dim] = displacement.x.array.reshape(points_num, topology_dim)
    grid.point_data["Displacement"] = values

    warp = grid.warp_by_vector("Displacement", factor=warp_factor)

    plotter = pvqt.BackgroundPlotter(
        title="Dynamic Crack Phase Field",
        auto_update=True,  # , window_size=(1600, 1120)
    )
    plotter.add_mesh(warp, show_edges=False, clim=[0, 1], cmap="coolwarm")
    plotter.view_xy(False)
    plotter.add_text("Time: 0.0", font_size=12, name="time_label")

    axes = pv.Axes(show_actor=True, actor_scale=10, line_width=5)
    axes.origin = (-7, -5.5, 0)
    plotter.add_actor(axes.actor)
    plotter.app.processEvents()

    screenshot_dir = out_dir / "Screenshots"
    if not screenshot_dir.exists():
        os.makedirs(screenshot_dir)
    plotter.screenshot(screenshot_dir / "initial.tiff")

# %% Solve the problem
time_old = 0.0
delta_time = 0.0
delta_time_old = 0.0
timer = Timer()
total_timer = Timer()

timers = {
    "disp_solve": Timer(),
    "energy_history": Timer(),
    "crack_solve": Timer(),
    "normalize": Timer(),
    "verbose": Timer(),
    "plot": Timer(),
    "update": Timer(),
    "save": Timer(),
}

for _, timer in timers.items():
    timer.reset()
    timer.pause()

for idx, t in enumerate(T):
    delta_time_old = delta_time
    delta_time = t - time_old
    time_old = t

    dt.value = delta_time
    dt_old.value = delta_time_old

    # Update the load
    load_top.value[preset.load_direction] = getLoad(t)
    load_bot.value[preset.load_direction] = -getLoad(t)

    # Solve the problem
    timers["disp_solve"].resume()
    displacement_problem.solve()
    timers["disp_solve"].pause()

    timers["energy_history"].resume()
    energy_history.interpolate(energy_history_expr)
    timers["energy_history"].pause()

    timers["crack_solve"].resume()
    crack_phase_problem.solve()
    timers["crack_solve"].pause()

    timers["normalize"].resume()
    crack_phase.x.array[:] = np.clip(crack_phase.x.array, crack_phase_old.x.array, 1)
    timers["normalize"].pause()

    timers["verbose"].resume()
    if preset.verbose:
        u_tuple = getMaxMin(displacement, displacement_old)
        d_tuple = getMaxMin(crack_phase, crack_phase_old)
        print(f"Time: {t:.3e}, Load: {getLoad(t):.3e}, δt: {dt.value:.3e}")
        print(
            f"  u max/min: {u_tuple[0]:.2e}/{u_tuple[1]:.2e}, δu max/min: {u_tuple[2]:.2e}/{u_tuple[3]:.2e}"
        )
        print(
            f"  d max/min: {d_tuple[0]:.2e}/{d_tuple[1]:.2e}, δd max/min: {d_tuple[2]:.2e}/{d_tuple[3]:.2e}"
        )
    timers["verbose"].pause()

    timers["plot"].resume()
    if have_pyvista:
        warp.point_data["Crack Phase"][:] = crack_phase.x.array

        grid.point_data["Displacement"][:, :topology_dim] = (
            displacement.x.array.reshape(points_num, topology_dim)
        )

        warp_ = grid.warp_by_vector("Displacement", factor=warp_factor)
        warp.points[:, :] = warp_.points

        plotter.add_text(f"Time: {t:.3e}", font_size=12, name="time_label")
        plotter.app.processEvents()

        plotter.screenshot(screenshot_dir / f"{idx+1}.tiff")
    timers["plot"].pause()

    timers["update"].resume()
    displacement_old2.x.array[:] = displacement_old.x.array
    displacement_old.x.array[:] = displacement.x.array
    crack_phase_old.x.array[:] = crack_phase.x.array
    timers["update"].pause()

    timers["save"].resume()
    if idx == 0 or (idx + 1) % save_interval == 0 or (idx + 1) == len(T):
        pvd_file.write_function(displacement, t)
        pvd_file.write_function(crack_phase, t)
        pvd_file.write_function(energy_history, t)
        print(f"Saved at {t:.3e}. Elapsed: {timer}, total elapsed: {total_timer}\n")
        timer.reset()
    timers["save"].pause()

print(f"Simulation completed. Total time: {total_timer}\n")
pvd_file.close()

for name, timer in timers.items():
    print(f"{name}: {timer}")
