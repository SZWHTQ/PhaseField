import dolfinx as dfx
import ufl

from mpi4py import MPI

import gmsh
import numpy as np

import matplotlib.pyplot as plt

import tqdm.autonotebook

from pathlib import Path
import os

import Material
import Constitutive
import Problem
import Util

from memory_profiler import profile


os.environ["OMP_NUM_THREADS"] = "1"

result_dir = Path("result/memory_profiling/serial")
if not result_dir.exists():
    result_dir.mkdir(exist_ok=True, parents=True)

is_profiling_enabled = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
host = 0


@profile(stream=open(result_dir / f"memory_profiling/main_{rank}.txt", "w"))
def thick_cylinder():
    # %% Preparation
    mesh_out = True

    if rank == host:
        dfx.log.set_log_level(dfx.log.LogLevel.INFO)
        dfx.log.set_output_file(str(result_dir / "solve.log"))

    # %% Create the material
    lame, shear_modulus = Util.getLamesParameters(E=70e3, nu=0.3)
    material = Material.JohnsonCookMaterial(
        mass_density=None,
        lame=lame,
        shear_modulus=shear_modulus,
        initial_yield_stress=206.0,
        strength_coefficient=505.0,
        strain_rate_strength_coefficient=0.01,
        hardening_exponent=0.42,
        temperature_exponent=1.68,
        reference_strain_rate=5e-4,
        reference_temperature=293.0,
        melting_temperature=1189.0,
    )

    # %% Create the mesh
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()

    Re, Ri = 1.3, 1.0
    mesh_size = 0.01

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

        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

        gmsh.model.mesh.generate(2)

        gmsh.write((result_dir / "cylinder.msh").as_posix())

    mesh, _, _ = dfx.io.gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()

    # %% Create the constitutive model
    constitutive = Constitutive.IsotropicJohnsonCook2DModel(material, mesh)

    # %% Create the problem
    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    problem = Problem.IsotropicPlasticProblem(constitutive, metadata=metadata)

    # %% Create dirichlet boundary conditions
    bottom_facets = dfx.mesh.locate_entities(
        mesh, problem.fdim, lambda x: np.isclose(x[1], 0.0)
    )
    left_facets = dfx.mesh.locate_entities(
        mesh, problem.fdim, lambda x: np.isclose(x[0], 0.0)
    )

    def getMarker(r):
        def marker(x):
            return np.isclose(np.linalg.norm(x, axis=0), r)

        return marker

    internal_facets = dfx.mesh.locate_entities(mesh, problem.fdim, getMarker(Re))
    external_facets = dfx.mesh.locate_entities(mesh, problem.fdim, getMarker(Ri))

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
    sorted_indices = np.argsort(marked_facets)
    facet_tags = dfx.mesh.meshtags(
        mesh, problem.fdim, marked_facets[sorted_indices], marked_values[sorted_indices]
    )

    mesh.topology.create_connectivity(problem.fdim, problem.tdim)
    if mesh_out:
        with dfx.io.XDMFFile(comm, result_dir / "facet_tags.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(facet_tags, mesh.geometry)

    mesh.topology.create_connectivity(problem.fdim, problem.tdim)
    bcs = [
        dfx.fem.dirichletbc(
            dfx.fem.Constant(mesh, 0.0),
            dfx.fem.locate_dofs_topological(
                problem.V.sub(1), problem.fdim, facet_tags.find(1)
            ),
            problem.V.sub(1),
        ),
        dfx.fem.dirichletbc(
            dfx.fem.Constant(mesh, 0.0),
            dfx.fem.locate_dofs_topological(
                problem.V.sub(0), problem.fdim, facet_tags.find(3)
            ),
            problem.V.sub(0),
        ),
    ]
    problem.setBCs(bcs)

    # %% Create the boundary forces
    q_lim = 2 / np.sqrt(3) * np.log(Re / Ri) * (material.initial_yield_stress * 1.25)

    def getLoad(t):
        return q_lim * t

    T = dfx.fem.Constant(mesh, 0.0)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, metadata=metadata)
    problem.F += -T * ufl.inner(problem.v, n) * ds(4)

    # %% Solve the problem
    problem.result_dir = result_dir
    problem.tolerance = 1e-8
    problem.max_iterations = 100
    problem.iteration_out = False
    problem.prepare()

    tree = dfx.geometry.bb_tree(mesh, mesh.geometry.dim)
    point = np.array([Ri, 0, 0])
    cell_candidates = dfx.geometry.compute_collisions_points(tree, point)
    colliding_cells = dfx.geometry.compute_colliding_cells(mesh, cell_candidates, point)
    cells = colliding_cells.links(0)

    increment_num = 250
    load_steps = np.linspace(0, 1.1, increment_num + 1)[1:] ** 0.5
    if rank == host:
        disp_results = np.zeros((increment_num + 1, 2))
    if rank == host:
        progress = tqdm.autonotebook.tqdm(
            total=load_steps.size  # , desc=f"Solving T={T.value:.3f}"
        )
    for i, t in enumerate(load_steps):
        load = getLoad(t)
        T.value = load

        problem.assemble()

        num_it = problem.solve()

        problem.write(t)

        if rank == host:
            # print(f"Increment {i}: T={T.value:.3f}", flush=True)
            progress.update(1)
            progress.set_description(f"T={T.value:.3f} solved in {num_it} iterations")

        disp_list = None
        if len(cells) > 0:
            disp_list = problem.displacement.eval(point, cells[:1])
        disp_list = comm.gather(disp_list, root=host)
        if rank == host:
            for disp in disp_list:
                if disp is not None:
                    disp_results[i + 1, :] = [disp[0], t]
                    break

    del mesh
    del material
    del constitutive
    del problem

    if rank == host:
        progress.close()

        np.savetxt(result_dir / "displacement.txt", disp_results)
        plt.plot(disp_results[:, 0], disp_results[:, 1], "-o")
        plt.xlabel("Displacement at the inner boundary")
        plt.ylabel("Load factor")
        plt.xlim([0, 9e-3])
        plt.xticks(np.linspace(0, 9e-3, 6))
        plt.ylim([0, 1.1])
        plt.grid()
        plt.savefig(result_dir / "displacement.png")
        # plt.show()


if __name__ == "__main__":
    if is_profiling_enabled:
        from line_profiler import LineProfiler

        profiler = LineProfiler()
        profiler_wrapper = profiler(thick_cylinder)
        profiler_wrapper()
        profiler_result = result_dir / f"profiling_{rank}.txt"
        with open(profiler_result, "w") as stream:
            profiler.print_stats(stream=stream)
    else:
        thick_cylinder()
