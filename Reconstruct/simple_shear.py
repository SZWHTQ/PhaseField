import dolfinx as dfx

from mpi4py import MPI

import gmsh
import numpy as np

import tqdm.autonotebook

from pathlib import Path
import os

import Material
import Constitutive
import Problem


os.environ["OMP_NUM_THREADS"] = "1"

result_dir = Path("result/simple_shear/mpi")
if not result_dir.exists():
    result_dir.mkdir(exist_ok=True, parents=True)

is_profiling_enabled = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
host = 0


Ti6Al4V = Material.DuctileFractureMaterial(
    mass_density=4.43e-9,
    lame=68501.40618722378,
    shear_modulus=41984.732824427476,
    viscosity=1e-6,
    initial_yield_stress=1000.0,
    strength_coefficient=1000.0,
    strain_rate_strength_coefficient=0.0,
    hardening_exponent=1.0,
    temperature_exponent=1.0,
    reference_strain_rate=1.0,
    reference_temperature=298.0,
    melting_temperature=1873.0,
    fracture_viscosity=5e-5,
    fracture_characteristic_length=0.1,
    critical_energy_release_rate=45.0,  ## ?
    threshold_energy=150.0,
)

w, h = 10, 4
crack_length = 4
mesh_x = int(4 * w / Ti6Al4V.fracture_characteristic_length)

gmsh.initialize()
if rank == host:
    # Define the geometry
    h_e = w / mesh_x
    width = h_e * 0.5
    max_element_size = h_e
    min_element_size = h_e
    gmsh.model.occ.addPoint(-w / 2, -h / 2, 0, max_element_size, 1)
    gmsh.model.occ.addPoint(w / 2, -h / 2, 0, max_element_size, 2)
    gmsh.model.occ.addPoint(w / 2, h / 2, 0, max_element_size, 3)
    gmsh.model.occ.addPoint(-w / 2, h / 2, 0, max_element_size, 4)

    gmsh.model.occ.addPoint(-crack_length / 2, width / 2, 0, min_element_size, 5)
    gmsh.model.occ.addPoint(crack_length / 2, width / 2, 0, min_element_size, 6)
    gmsh.model.occ.addPoint(crack_length / 2, -width / 2, 0, min_element_size, 7)
    gmsh.model.occ.addPoint(-crack_length / 2, -width / 2, 0, min_element_size, 8)

    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)
    gmsh.model.occ.addLine(3, 4, 3)
    gmsh.model.occ.addLine(4, 1, 4)

    gmsh.model.occ.addLine(5, 6, 5)
    # gmsh.model.occ.addLine(6, 7, 6)
    gmsh.model.occ.addCircle(
        crack_length / 2,
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
        -crack_length / 2,
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

    # if preset.dim == 3:
    #     gmsh.model.occ.extrude([(2, 1)], 0, 0, preset.thickness, [1], recombine=True)
    #     gmsh.model.occ.synchronize()

    #     gmsh.model.addPhysicalGroup(3, [1], 1, "Model")

    #     gmsh.model.mesh.generate(3)

    # gmsh.write("mesh.inp")
    gmsh.write("mesh.msh")

    # gmsh.fltk.run()

mesh, _, _ = dfx.io.gmshio.model_to_mesh(gmsh.model, comm, host, gdim=2)
gmsh.finalize()


constitutive = Constitutive.DuctileFracturePrincipleStrainDecomposition(Ti6Al4V, mesh)

metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
dt = dfx.fem.Constant(mesh, 0.0)
problem = Problem.DuctileFractureProblem(
    constitutive=constitutive, dt=dt, metadata=metadata
)

top = dfx.mesh.locate_entities_boundary(
    mesh, problem.fdim, lambda x: np.isclose(x[1], h / 2)
)
bot = dfx.mesh.locate_entities_boundary(
    mesh, problem.fdim, lambda x: np.isclose(x[1], -h / 2)
)

displacement_load_top = dfx.fem.Constant(mesh, (0.0,) * problem.tdim)
displacement_load_bot = dfx.fem.Constant(mesh, (0.0,) * problem.tdim)
displacement_bcs = [
    dfx.fem.dirichletbc(
        displacement_load_top,
        dfx.fem.locate_dofs_topological(
            problem.isotropic_plastic_problem.V,
            problem.fdim,
            top,
        ),
        problem.isotropic_plastic_problem.V,
    ),
    dfx.fem.dirichletbc(
        displacement_load_bot,
        dfx.fem.locate_dofs_topological(
            problem.isotropic_plastic_problem.V,
            problem.fdim,
            bot,
        ),
        problem.isotropic_plastic_problem.V,
    ),
]

crack_phase_bcs = []

problem.isotropic_plastic_problem.setBCs(displacement_bcs)
problem.ductile_fracture_sub_problem.setBCs(crack_phase_bcs)

u_limit = 0.4
end_time = 4e-4


def getLoad(t):
    def smooth(xi):
        return xi**3 * (10 - 15 * xi + 6 * xi * xi)

    return u_limit * smooth(t / end_time)


problem.result_dir = result_dir
problem.result_filename = "ductile_fracture"
problem.isotropic_plastic_problem.max_iterations = 100
problem.isotropic_plastic_problem.tolerance = 1e-5
problem.isotropic_plastic_problem.iteration_out = True
problem.prepare()

increment_num = 200
delta_t = end_time / increment_num
load_steps = np.arange(delta_t, end_time + delta_t / 2, delta_t)[
    0 : int(increment_num / 2)
]

if rank == host:
    progress = tqdm.autonotebook.tqdm(total=load_steps.size)

t_old = 0.0
load_old = 0.0
for i, t in enumerate(load_steps):
    dt.value = t - t_old
    t_old = t

    # `0` for x-direction, which means simple shear for type II crack
    L = getLoad(t)
    displacement_load_top.value[0] = L - load_old
    displacement_load_bot.value[0] = load_old - L
    load_old = L

    problem.assemble()

    num_it = problem.solve()

    problem.write(t)

    if rank == host:
        progress.update(1)
        progress.set_description(f"{L=:.3e} solved {num_it} its")