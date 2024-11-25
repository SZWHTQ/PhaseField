# %%
import dolfinx as dfx
import ufl
import dolfinx.fem.petsc as petsc

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
import shutil

result_dir = Path("result")
if not result_dir.exists():
    result_dir.mkdir(exist_ok=True, parents=True)

# %%
E = 70e3
nu = 0.3
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
y0 = 250.0
Et = E / 100.0
H = E * Et / (E - Et)

mesh_out = False
iteration_out = True

# %%
Re, Ri = 1.3, 1.0
mesh_size = 0.03

rank = MPI.COMM_WORLD.rank
host = 0

if gmsh.isInitialized():
    gmsh.finalize()
gmsh.initialize()

if rank == host:
    gmsh.model.occ.addPoint(Ri, 0, 0, meshSize=mesh_size, tag=1)
    gmsh.model.occ.addPoint(Re, 0, 0, meshSize=mesh_size, tag=2)
    gmsh.model.occ.addPoint(0, Re, 0, meshSize=mesh_size, tag=3)
    gmsh.model.occ.addPoint(0, Ri, 0, meshSize=mesh_size, tag=4)

    gmsh.model.occ.addLine(1, 2, tag=1)
    gmsh.model.occ.addCircle(0, 0, 0, Re, angle1=0, angle2=np.pi / 2, tag=2)
    gmsh.model.occ.addLine(3, 4, tag=3)
    gmsh.model.occ.addCircle(0, 0, 0, Ri, angle1=0, angle2=np.pi / 2, tag=4)

    gmsh.model.occ.addCurveLoop([1, 2, 3, 4], tag=1)

    gmsh.model.occ.addPlaneSurface([1], tag=1)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [1], tag=1, name="cylinder")

    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.mesh.generate(2)

    gmsh.write((result_dir / "cylinder.msh").as_posix())

    # gmsh.fltk.run()


# %%
comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = dfx.io.gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)

gmsh.finalize()

# with dfx.io.XDMFFile(comm, "mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)

# %%
tdim = mesh.topology.dim
fdim = tdim - 1

W = dfx.fem.functionspace(mesh, ("Lagrange", 1, (tdim * tdim,)))
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (tdim,)))
S = dfx.fem.functionspace(mesh, ("Lagrange", 1))
# Ss = dfx.fem.functionspace(mesh, ("DG", 1))

# %%
stress_vector = dfx.fem.Function(W, name="Stress")
stress_vector_old = dfx.fem.Function(W)
n_elas_vector = dfx.fem.Function(W)

beta = dfx.fem.Function(S)
equivalent_plastic_strain = dfx.fem.Function(S, name="Cumulative plastic strain")
displacement = dfx.fem.Function(V, name="Displacement")
displacement_inc = dfx.fem.Function(V, name="Displacement increment")
ddu = dfx.fem.Function(V, name="Iteration correction")
equivalent_plastic_strain_vis = dfx.fem.Function(S, name="Equivalent plastic strain")

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# %%
# Bottom and left boundary symmetry conditions

# bcs = [
#     # bottom
#     dfx.fem.dirichletbc(
#         dfx.fem.Constant(mesh, 0.0),
#         dfx.fem.locate_dofs_geometrical(V.sub(1), lambda x: np.isclose(x[1], 0.0)),
#         V.sub(1),
#     ),
#     # left
#     dfx.fem.dirichletbc(
#         dfx.fem.Constant(mesh, 0.0),
#         dfx.fem.locate_dofs_geometrical(V.sub(0), lambda x: np.isclose(x[0], 0.0)),
#         V.sub(0),
#     ),
# ]


def getMarker(r):
    def marker(x):
        return np.isclose(np.sqrt(x[0] ** 2 + x[1] ** 2), r)

    return marker


internal_facets = dfx.mesh.locate_entities(mesh, fdim, getMarker(Ri))
external_facets = dfx.mesh.locate_entities(mesh, fdim, getMarker(Re))

bottom_facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
left_facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 0.0))

# mesh.topology.create_connectivity(1, 0)
# for edge in internal_facets:
#     edge_to_node_connectivity = mesh.topology.connectivity(1, 0)
#     # n1, n2 = (
#     #     edge_to_node_connectivity.array[2 * edge],
#     #     edge_to_node_connectivity.array[2 * edge + 1],
#     # )
#     n1, n2 = edge_to_node_connectivity.links(edge)
#     coor1 = mesh.geometry.x[n1]
#     coor2 = mesh.geometry.x[n2]
#     print(f"Facet {edge}:{n1=} {coor1[:2]=}, {np.linalg.norm(coor1)}")
#     print(f"Facet {edge}:{n2=} {coor2[:2]=}, {np.linalg.norm(coor2)}")
#     print()

marked_facets = np.hstack(
    [bottom_facets, external_facets, left_facets, internal_facets]
)
marked_values = np.hstack(
    [
        np.full_like(bottom_facets, 1),
        np.full_like(external_facets, 2),
        np.full_like(left_facets, 3),
        np.full_like(internal_facets, 4),
    ]
)
sorted_facets = np.argsort(marked_facets)
facet_tags = dfx.mesh.meshtags(
    mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
)

mesh.topology.create_connectivity(fdim, tdim)
bcs = [
    dfx.fem.dirichletbc(
        dfx.fem.Constant(mesh, 0.0),
        dfx.fem.locate_dofs_topological(V.sub(1), fdim, facet_tags.find(1)),
        V.sub(1),
    ),
    dfx.fem.dirichletbc(
        dfx.fem.Constant(mesh, 0.0),
        dfx.fem.locate_dofs_topological(V.sub(0), fdim, facet_tags.find(3)),
        V.sub(0),
    ),
]

# %%
q_lim = 2.0 / np.sqrt(3.0) * np.log(Re / Ri) * y0


def getLoad(t):
    return -q_lim * t


def getStrain(u):
    e = ufl.sym(ufl.grad(u))
    return ufl.as_tensor([[e[0, 0], e[0, 1], 0], [e[1, 0], e[1, 1], 0], [0, 0, 0]])


def getElasticStress(strain):
    return lmbda * ufl.tr(strain) * ufl.Identity(3) + 2 * mu * strain


def asThreeDimensionalTensor(expr):
    return ufl.as_tensor(
        [[expr[0], expr[3], 0], [expr[3], expr[1], 0], [0, 0, expr[2]]]
    )


def macaulayBracket(x):
    return 0.5 * (x + abs(x))


def stressProjection(strain_inc, stress_vector_old, equivalent_plastic_strain_old):
    stress_old = asThreeDimensionalTensor(stress_vector_old)

    stress_trial = stress_old + getElasticStress(strain_inc)

    deviatoric_stress = ufl.dev(stress_trial)
    equivalent_stress = ufl.sqrt(1.5 * ufl.inner(deviatoric_stress, deviatoric_stress))

    f_elas = equivalent_stress - y0 - H * equivalent_plastic_strain_old

    equivalent_plastic_strain_inc = macaulayBracket(f_elas) / (3 * mu + H)

    n_elas = deviatoric_stress / equivalent_stress * macaulayBracket(f_elas) / f_elas

    beta = 3 * mu * equivalent_plastic_strain_inc / equivalent_stress

    stress = stress_trial - beta * deviatoric_stress

    # return stress, n_elas, beta, equivalent_plastic_strain_inc
    return (
        ufl.as_vector([stress[0, 0], stress[1, 1], stress[2, 2], stress[0, 1]]),
        ufl.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]),
        beta,
        equivalent_plastic_strain_inc,
    )


# %%
def getTangentStress(u):
    e = getStrain(u)
    N_elas = asThreeDimensionalTensor(n_elas_vector)
    return (
        getElasticStress(e)
        - 3 * mu * (3 * mu / (3 * mu + H) - beta) * ufl.inner(N_elas, e) * N_elas
        - 2 * mu * beta * ufl.dev(e)
    )


# %%
mesh.topology.create_connectivity(fdim, tdim)
if iteration_out:
    with dfx.io.XDMFFile(comm, result_dir / "face_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)

# result_file = dfx.io.XDMFFile(comm, result_dir / "plastic.xdmf", "w")
# result_file.write_mesh(mesh)
# result_file = dfx.io.VTKFile(comm, result_dir / "plastic.pvd", "w")

result_file = dfx.io.VTXWriter(
    comm,
    result_dir / "plastic.bp",
    [displacement, equivalent_plastic_strain_vis, stress_vector, displacement_inc],
    engine="BP4",
)


# %%
metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, metadata=metadata)
T = dfx.fem.Constant(mesh, 0.0)
n = ufl.FacetNormal(mesh)


# def getNormal(x):
#     # values[:] = x[:2] / np.linalg.norm(x[:2], axis=0)
#     distances = np.linalg.norm(x[:2], axis=0)
#     # np.where(np.isclose(distances, Ri), x[:2] / distances, 0, out=values)
#     return np.where(np.isclose(distances, Ri), -x[:2] / distances, 0)


# n = dfx.fem.Function(V)
# n.name = "Normal"
# n.interpolate(getNormal)
# result_file.write_function(n)

# TODO: First working version
F = ufl.inner(getStrain(v), getTangentStress(u)) * dx
F += ufl.inner(getStrain(v), asThreeDimensionalTensor(stress_vector)) * dx
F += -ufl.inner(v, T * n) * ds(4)
a = dfx.fem.form(ufl.lhs(F))
L = dfx.fem.form(ufl.rhs(F))

# TODO: Try with LinearProblem
# F = ufl.inner(getStrain(v), getTangentStress(u)) * dx
# F += ufl.inner(getStrain(v), asThreeDimensionalTensor(stress_vector)) * dx
# F += -ufl.dot(T * n, v) * ds(4)
# a = ufl.lhs(F)
# L = ufl.rhs(F)


def local_project(v, V, u=None):
    """
    Project a UFL expression onto a function space in DOLFINx.

    Parameters:
    v: UFL expression or DOLFINx function to project.
    V: DOLFINx function space where the projection is performed.
    u: Optional existing DOLFINx Function to store the projection result.

    Returns:
    DOLFINx Function containing the projection result if `u` is None.
    """
    dx = ufl.Measure("dx", domain=V.mesh)

    # Define the variational problem
    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)
    a_proj = dfx.fem.form(ufl.inner(dv, v_) * dx)
    b_proj = dfx.fem.form(ufl.inner(v, v_) * dx)

    # Assemble the matrix and vector
    # A = petsc.create_matrix(a_proj)
    A = petsc.assemble_matrix(a_proj)
    A.assemble()
    b = petsc.assemble_vector(b_proj)

    # Create the solution function
    # flag = False
    # if u is None:
    #     u = dfx.fem.Function(V)
    #     flag = True

    # Create the solver
    solver = PETSc.KSP().create(V.mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Solve the linear system
    with b.localForm() as b_local:
        b_local.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # solver.solve(b, u.x.petsc_vec)

    # u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # return u if flag else None
    if u is None:
        u = dfx.fem.Function(V)
        solver.solve(b, u.x.petsc_vec)
        # u.vector.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )
        return u
    else:
        solver.solve(b, u.x.petsc_vec)
        # u.vector.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )


tree = dfx.geometry.bb_tree(mesh, mesh.geometry.dim)
point = np.array([Ri, 0, 0])
cell_candidates = dfx.geometry.compute_collisions_points(tree, point)
colliding_cells = dfx.geometry.compute_colliding_cells(mesh, cell_candidates, point)
cells = colliding_cells.links(0)

# %%
# dfx.log.set_log_level(dfx.log.LogLevel.INFO)
max_iteration = 10000
tolerance = 1e-5
increment_num = 20
load_steps = np.linspace(0, 1.1, increment_num + 1)[1:] ** 0.5
results = np.zeros((increment_num + 1, 2))
for i, t in enumerate(load_steps):
    load = getLoad(t)
    T.value = load

    A = petsc.create_matrix(a)
    b = petsc.create_vector(L)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)

    # solver.setType(PETSc.KSP.Type.BCGS)
    # pc = solver.getPC()
    # pc.setType(PETSc.PC.Type.JACOBI)

    # solver.setType(PETSc.KSP.Type.MINRES)
    # pc = solver.getPC()
    # pc.setType(PETSc.PC.Type.HYPRE)
    # pc.setHYPREType("boomeramg")

    # solver.setType(PETSc.KSP.Type.CG)
    # pc = solver.getPC()
    # pc.setType(PETSc.PC.Type.SOR)

    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)

    # problem = petsc.LinearProblem(
    #     a,
    #     L,
    #     bcs=bcs,
    #     u=ddu,
    #     petsc_options={
    #         "ksp_type": "preonly",
    #         "pc_type": "lu",
    #         "pc_factor_mat_solver_type": "mumps",
    #     },
    # )

    correction_norm = 1
    correction_norm_old = 0
    displacement_inc.x.array[:] = 0.0

    if iteration_out:
        iteration_result = dfx.io.XDMFFile(
            comm, result_dir / f"iteration_{i}.xdmf", "w"
        )
        iteration_result.write_mesh(mesh)

    # print(f"Increment {i}: T={T.value:.3f}, {nRes0=:.2e}")
    if rank == host:
        print(f"Increment {i}: T={T.value:.3f}")
    num_iteration = 0
    while correction_norm > tolerance and num_iteration < max_iteration:
        with b.localForm() as b_loc:
            b_loc.set(0)
        A.zeroEntries()
        petsc.assemble_matrix(A, a, bcs=bcs)
        A.assemble()
        petsc.assemble_vector(b, L)
        # b.scale(-1)
        petsc.apply_lifting(b, [a], [bcs], x0=[displacement_inc.x.petsc_vec])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs, x0=displacement_inc.x.petsc_vec)

        solver.solve(b, ddu.x.petsc_vec)
        ddu.x.scatter_forward()

        # problem.solve()

        displacement_inc.x.array[:] += ddu.x.array[:]
        strain_inc = getStrain(displacement_inc)
        _stress_vector, _n_elas_vector, _beta, _equivalent_plastic_strain_inc = (
            stressProjection(strain_inc, stress_vector_old, equivalent_plastic_strain)
        )
        local_project(_stress_vector, W, stress_vector)
        local_project(_n_elas_vector, W, n_elas_vector)
        local_project(_beta, S, beta)

        if iteration_out:
            iteration_result.write_function(ddu, t=num_iteration)

        correction_norm_old = correction_norm
        correction_norm = ddu.x.petsc_vec.norm(0)

        if rank == host:
            print(
                f"    iteration {num_iteration}: r = {correction_norm:.3e}, dr = {(correction_norm - correction_norm_old):.3e}",
                end="\r",
                flush=True,
            )

        num_iteration += 1
    else:
        if rank == host:
            terminal_size = shutil.get_terminal_size()
            width = terminal_size.columns
            if num_iteration == max_iteration:
                print(" " * width, end="\r", flush=True)
                print(
                    f"   iteration {num_iteration}: r = {correction_norm:.3e}, dr = {abs(correction_norm - correction_norm_old):.3e}"
                )
                raise RuntimeError("Failed to converge")
            else:
                print(" " * width, end="\r", flush=True)
                print(
                    f"    Converged at iteration {num_iteration}: r = {correction_norm:.3e}"
                )
    if iteration_out:
        iteration_result.close()

    displacement.x.array[:] += displacement_inc.x.array[:]
    equivalent_plastic_strain.x.array[:] += local_project(
        _equivalent_plastic_strain_inc, S
    ).x.array[:]
    stress_vector_old.x.array[:] = stress_vector.x.array[:]
    equivalent_plastic_strain_vis.interpolate(equivalent_plastic_strain)

    # result_file.write_function(displacement, t)
    # result_file.write_function(equivalent_plastic_strain, t)
    # result_file.write_function(stress_vector, t)
    # result_file.write_function(displacement_inc, t)

    result_file.write(t)

    disp = displacement.eval(point, cells[:1])
    displacement.evaluate

    results[i + 1, :] = [disp[0], t]

if rank == host:
    np.savetxt(result_dir / "displacement.txt", results)
    plt.plot(results[:, 0], results[:, 1], "-o")
    plt.xlabel("Displacement at the inner boundary")
    plt.ylabel("Load factor")
    plt.savefig(result_dir / "displacement.png")
    plt.show()

result_file.close()
