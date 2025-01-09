import dolfinx as dfx
import dolfinx.fem.petsc as petsc
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tqdm

import Material
from Solid import Constitutive
import Util

from pathlib import Path

import shutil

# from memory_profiler import profile

# result_dir = Path("result/simple_shear.memory_test/mpi")
# if not result_dir.exists():
#     result_dir.mkdir(exist_ok=True, parents=True)
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# host = 0


class Problem:
    def __init__(
        self,
        mesh: dfx.mesh.Mesh,
        element_type: str,
        degree: int,
        dx: ufl.Measure,
        metadata: dict,
    ):
        self._mesh = mesh

        self._comm: MPI.Intracomm = self._mesh.comm

        if element_type is None:
            self._element_type = "Lagrange"
        else:
            self._element_type = element_type

        self._degree = degree

        self._bcs = None

        if dx is None:
            self.dx = ufl.Measure("dx", domain=self._mesh, metadata=metadata)

        self.tdim = mesh.topology.dim
        self.fdim = self.tdim - 1

        self.F = None

        self.result_dir = Path("result")
        self.result_filename = "problem"

        self._W_vis = dfx.fem.functionspace(
            self._mesh, ("Lagrange", 1, (self.tdim, self.tdim))
        )
        self._V_vis = dfx.fem.functionspace(self._mesh, ("Lagrange", 1, (self.tdim,)))
        self._S_vis = dfx.fem.functionspace(self._mesh, ("Lagrange", 1))

    def setBCs(self, bcs):
        self._bcs = bcs

    def prepare(self):
        # Judge if the result directory exists
        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True)

        # Judge if the filename ends with ".bp"
        if self.result_filename.endswith(".bp"):
            self.result_filename = self.result_filename[:-3]

    def assemble(self):
        # self._a, self._L = ufl.system(self.F)
        self._a = dfx.fem.form(ufl.lhs(self.F))
        self._L = dfx.fem.form(ufl.rhs(self.F))

        self._A = petsc.create_matrix(self._a)
        self._b = petsc.create_vector(self._L)

        self._solver = PETSc.KSP().create(self._comm)
        self._solver.setOperators(self._A)

        ### Memory leak here, #TODO: Fix this in the future
        # self._solver.setType(PETSc.KSP.Type.PREONLY)
        # pc = self._solver.getPC()
        # pc.setType(PETSc.PC.Type.LU)

        self._solver.setType(PETSc.KSP.Type.MINRES)
        pc = self._solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")

        # self._solver.setType(PETSc.KSP.Type.CG)

        # self._solver.setType(PETSc.KSP.Type.GMRES)
        # pc = self._solver.getPC()
        # pc.setType(PETSc.PC.Type.NONE)


class SolidProblem(Problem):
    def __init__(
        self,
        constitutive: Constitutive.Constitutive,
        element_type: str = "Lagrange",
        degree: int = 1,
        dx: ufl.Measure = None,
        metadata: dict = None,
    ):
        self._constitutive = constitutive
        self._material = self._constitutive._material
        super().__init__(
            mesh=self._constitutive._mesh,
            element_type=element_type,
            degree=degree,
            dx=dx,
            metadata=metadata,
        )

        self.V = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (self.tdim,))
        )

        self.displacement = dfx.fem.Function(self.V)

        self._displacement_vis = dfx.fem.Function(self._V_vis, name="Displacement")


class IsotropicPlasticProblem(SolidProblem):
    def __init__(
        self,
        constitutive: Constitutive.IsotropicJohnsonCook,
        dt: dfx.fem.Constant,
        element_type: str = "Lagrange",
        degree: int = 1,
        dx: ufl.Measure = None,
        metadata: dict = None,
    ):
        super().__init__(
            constitutive=constitutive,
            element_type=element_type,
            degree=degree,
            dx=dx,
            metadata=metadata,
        )
        self.dt = dt

        assert issubclass(type(self._material), Material.JohnsonCookMaterial)
        assert issubclass(type(self._constitutive), Constitutive.IsotropicJohnsonCook)

        self.displacement_inc = dfx.fem.Function(self.V)
        # Iteration correction for displacement in Newton-Raphson method
        self.iteration_correction = dfx.fem.Function(self.V)

        self.displacement_inc_old = dfx.fem.Function(self.V)
        self.displacement_inc_old2 = dfx.fem.Function(self.V)
        self.displacement_inc_old3 = dfx.fem.Function(self.V)

        self.velocity = dfx.fem.Function(self.V, )
        self.velocity_old = dfx.fem.Function(self.V)
        self.velocity_inc = dfx.fem.Function(self.V)

        self.acceleration = dfx.fem.Function(self.V, )
        self.acceleration_old = dfx.fem.Function(self.V)
        self.acceleration_inc = dfx.fem.Function(self.V)

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        con = self._constitutive

        a = self._getAcceleration(
            self.u,
            self.displacement_inc_old,
            self.displacement_inc_old2,
            self.dt,
        )
        v = self._getVelocity(self.u, self.displacement_inc_old, self.dt)

        self.F = self._material.mass_density * ufl.dot(self.v, a) * self.dx
        self.F += (
            ufl.inner(
                con.getStrain(self.v),
                con.getElasticStress(con.getStrain(self.u)),
            )
            * self.dx
        )
        self.F += self._material.viscosity * ufl.dot(self.v, v) * self.dx
        self.F += (
            self._material.mass_density * ufl.dot(self.v, self.acceleration) * self.dx
        )
        self.F += (
            ufl.inner(
                con.getStrain(self.v),
                con.stress,
            )
            * self.dx
        )
        self.F += self._material.viscosity * ufl.dot(self.v, self.velocity) * self.dx

        ## Config
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.iteration_out = False
        # Can't work properly right now, #TODO: Fix this in the future
        self.convergence_plot = False
        self.result_dir = Path("result")
        self.result_filename = "plastic"
        self.log_filename = "iteration.log"
        self.verbose = False

        self._solve_call_time = 0
        self._time = 0
        self._log_info_line_end_with = "\r"

        self._equivalent_plastic_strain_vis = dfx.fem.Function(
            self._S_vis, name="Equivalent plastic strain"
        )
        self._stress_vis = dfx.fem.Function(self._W_vis, name="Stress")
        self._equivalent_stress_vis = dfx.fem.Function(
            self._S_vis, name="Equivalent stress"
        )
        self._yield_stress = dfx.fem.Function(self._S_vis, name="Yield stress")
        self._hardening = dfx.fem.Function(self._S_vis, name="Hardening")

        self._elastic_strain_vis = dfx.fem.Function(self._W_vis, name="Elastic strain")
        self._elastic_strain_energy_positive_vis = dfx.fem.Function(
            self._S_vis, name="Elastic strain energy positive"
        )
        self._plastic_work_vis = dfx.fem.Function(self._S_vis, name="Plastic work")

        self._acceleration_vis = dfx.fem.Function(self._V_vis, name="Acceleration")
        self._velocity_vis = dfx.fem.Function(self._V_vis, name="Velocity")

    def prepare(self):
        super().prepare()

        self.result_file = dfx.io.VTXWriter(
            self._comm,
            self.result_dir / (self.result_filename + ".bp"),
            [
                self._displacement_vis,
                self._equivalent_plastic_strain_vis,
                self._stress_vis,
                self._equivalent_stress_vis,
                self._yield_stress,
                self._hardening,
            ],
            engine="BP4",
        )
        self.result_file.write(0)

        if (
            self._mesh.comm.rank == 0
        ):  # Shall be changed to the host rank, think about a better way
            # TODO: Change write and log to a standalone class
            self.log_file = open(self.result_dir / self.log_filename, "w")

    def solve(self):
        con: Constitutive.IsotropicJohnsonCook = self._constitutive

        num_iteration = 0
        correction_norm = 0
        converged = False
        host = 0
        rank = self._comm.Get_rank()

        self.displacement_inc.x.array[:] = 0.0

        Util.localProject(
            con.getYieldStress(),
            con.yield_stress,
        )
        Util.localProject(
            con.getHardening(),
            con.hardening,
        )

        if self.iteration_out:
            iteration_result = dfx.io.XDMFFile(
                self._comm,
                self.result_dir / f"iteration_{self._solve_call_time}.xdmf",
                "w",
            )
            iteration_result.write_mesh(self._mesh)

        if rank == host:
            self.log_file.write(f"Increment {self._solve_call_time}\n")
            if self.convergence_plot:
                correction_norm_history = []
                fig, ax = plt.subplots()
                ax.set_title("Convergence Plot")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Residual (log scale)")
                ax.set_yscale("log")
                ax.grid(True)

                # Use Agg backend for non-interactive plotting
                canvas = FigureCanvas(fig)

        while (not converged) and num_iteration < self.max_iterations:
            with self._b.localForm() as b_loc:
                b_loc.set(0.0)
            self._A.zeroEntries()
            petsc.assemble_matrix(self._A, self._a, bcs=self._bcs)
            self._A.assemble()

            petsc.assemble_vector(self._b, self._L)
            petsc.apply_lifting(self._b, [self._a], [self._bcs])
            self._b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            petsc.set_bc(self._b, self._bcs)

            self._solver.solve(self._b, self.iteration_correction.x.petsc_vec)
            self.iteration_correction.x.scatter_forward()

            self.displacement_inc.x.array[:] += self.iteration_correction.x.array[:]
            strain_inc = self._constitutive.getStrain(self.displacement_inc)
            (
                _stress_vector,
                _n_elastic_vector,
                _beta,
                _equivalent_plastic_strain_inc,
            ) = con.stressProjection(strain_inc)

            Util.localProject(_stress_vector, con.stress)
            Util.localProject(_n_elastic_vector, con.n_elastic)
            Util.localProject(_beta, con.beta)

            if self.iteration_out:
                iteration_result.write_function(
                    self.iteration_correction, t=num_iteration
                )

            # self._comm.Barrier()
            correction_norm_old = correction_norm
            correction_norm = Util.getL2Norm(self.iteration_correction)
            converged = correction_norm < self.tolerance
            converged = self._comm.allreduce(converged, op=MPI.LAND)

            if rank == host:
                if self.verbose:
                    terminal_size = shutil.get_terminal_size()
                    width = terminal_size.columns
                    # print(" " * width, end=self._log_info_line_end_with, flush=True)
                    print(
                        f"    iteration {num_iteration}: r = {correction_norm:.3e}, dr = {(correction_norm - correction_norm_old):.3e}",
                        # end=self._log_info_line_end_with,
                        flush=True,
                    )
                self.log_file.write(
                    f"iteration {num_iteration}: r = {correction_norm:.3e}, dr = {(correction_norm - correction_norm_old):.3e}\n"
                )
                # self.log_file.flush()
                if self.convergence_plot:
                    ax.plot(
                        correction_norm_history,
                        color="green",
                        label="Residual" if num_iteration == 0 else "",
                    )
                    ax.legend(loc="upper right")
                    canvas.draw()
                    # plt.pause(0.01)

            num_iteration += 1
            self._comm.Barrier()
        else:
            if self.iteration_out:
                iteration_result.close()
            if num_iteration == self.max_iterations:
                if rank == host:
                    if self.verbose:
                        terminal_size = shutil.get_terminal_size()
                        width = terminal_size.columns
                        print(" " * width, end=self._log_info_line_end_with, flush=True)
                        print(
                            f"   iteration {num_iteration}: r = {correction_norm:.3e}, dr = {(correction_norm - correction_norm_old):.3e}",
                        )
                    self.log_file.write(
                        f"iteration {num_iteration}: r = {correction_norm:.3e}, dr = {(correction_norm - correction_norm_old):.3e}\n"
                    )
                raise RuntimeError("Failed to converge")
            else:
                if rank == host:
                    if self.verbose:
                        # print(" " * width, end=self._log_info_line_end_with, flush=True)
                        print(
                            f"  Converged at iteration {num_iteration-1}: r = {correction_norm:.3e}"
                        )
                    self.log_file.write(
                        f"Converged at iteration {num_iteration-1}: r = {correction_norm:.3e}\n\n"
                    )
                    self.log_file.flush()

        if self.convergence_plot:
            correction_norm_history = []
            plt.close(fig)

        con.stress_old.x.array[:] = con.stress.x.array[:]
        con.equivalent_plastic_strain.x.array[:] += Util.macaulayBracket(
            Util.localProject(_equivalent_plastic_strain_inc, V=con.S).x.array[:]
        )
        deviatoric_stress = ufl.dev(con.stress)
        Util.localProject(
            ufl.sqrt(
                1.5
                * ufl.inner(
                    deviatoric_stress,
                    deviatoric_stress,
                )
            ),
            con.equivalent_stress,
        )
        self._post_solve()

        self._solve_call_time += 1

        return num_iteration

    def _post_solve(self):
        ## Useless for memory leak problem
        import gc

        opts = PETSc.Options()
        opts.clear()
        opts.destroy()
        del opts
        gc.collect()

    def write(self, time: float = 0):
        self._displacement_vis.interpolate(self.displacement)
        self._equivalent_plastic_strain_vis.interpolate(
            self._constitutive.equivalent_plastic_strain
        )
        self._stress_vis.interpolate(self._constitutive.stress_vector)
        self._equivalent_stress_vis.interpolate(self._constitutive.equivalent_stress)
        self._yield_stress.interpolate(self._constitutive.yield_stress)
        self._hardening.interpolate(self._constitutive.hardening)

        self.result_file.write(time)

    # def _getAcceleration(self, du, du_1, du_2, du_3, dt):
    #     return (2 * du + -5 * du_1 + 4 * du_2 - du_3) / dt**2

    def _getAcceleration(self, du, du_1, du_2, dt):
        return (du - 2 * du_1 + du_2) / dt**2

    def _getVelocity(self, du, du_1, dt):
        return (du - du_1) / dt

    def __exit__(self, exc_type, exc_value, traceback):
        self.result_file.close()


class DuctileFractureSubProblem(Problem):
    def __init__(
        self,
        # material: Material.DuctileFractureMaterial,
        # mesh: dfx.mesh.Mesh,
        constitutive: Constitutive.DuctileFracture,
        dt: dfx.fem.Constant,
        element_type: str = "Lagrange",
        degree: int = 1,
        dx: ufl.Measure = None,
        metadata: dict = None,
    ):
        self._constitutive = constitutive
        self._material = self._constitutive._material
        self.dt = dt
        super().__init__(
            mesh=self._constitutive._mesh,
            element_type=element_type,
            degree=degree,
            dx=dx,
            metadata=metadata,
        )

        assert issubclass(type(self._material), Material.DuctileFractureMaterial)
        assert issubclass(type(self._constitutive), Constitutive.DuctileFracture)

        self.S = dfx.fem.functionspace(self._mesh, (self._element_type, self._degree))

        self.crack_phase = dfx.fem.Function(self.S, name="Crack phase")
        self.crack_phase_old = dfx.fem.Function(self.S)

        self.u = ufl.TrialFunction(self.S)
        self.v = ufl.TestFunction(self.S)

        m = self._material
        self.F = (
            m.fracture_viscosity * (self.u - self.crack_phase_old) * self.v * self.dx
        )
        self.F += (
            -self.dt
            * (2 * (1 - self.u) * self._constitutive.crack_driven_force * self.v)
            * self.dx
        )
        self.F += (
            -self.dt
            * (
                -m.critical_energy_release_rate
                / m.fracture_characteristic_length
                * self.u
                * self.v
            )
            * self.dx
        )
        self.F += (
            -self.dt
            * (
                -m.critical_energy_release_rate
                * m.fracture_characteristic_length
                * ufl.dot(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v))
            )
            * self.dx
        )
        self.F += (
            -self.dt
            * (
                2
                * m.threshold_energy
                * (
                    self.u * self.v
                    + m.fracture_characteristic_length**2
                    * ufl.dot(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v))
                )
            )
            * self.dx
        )

        ## Config
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.iteration_out = False
        self.result_dir = Path("result")
        self.result_filename = "plastic"
        self.log_filename = "crack_phase_iteration.log"
        self.verbose = False

        self._solve_call_time = 0
        self._time = 0
        self._log_info_line_end_with = "\r"

        self._crack_phase_vis = dfx.fem.Function(self._S_vis, name="Crack phase")
        self._crack_driven_force_vis = dfx.fem.Function(
            self._S_vis, name="Crack driven force"
        )

    def prepare(self):
        super().prepare()

        self.result_file = dfx.io.VTXWriter(
            self._comm,
            self.result_dir / (self.result_filename + ".bp"),
            [self._crack_phase_vis, self._crack_driven_force_vis],
            engine="BP4",
        )

        if (
            self._mesh.comm.rank == 0
        ):  # Shall be changed to the host rank, think about a better way
            # TODO: Change write and log to a standalone class
            self.log_file = open(self.result_dir / self.log_filename, "w")

    def assemble(self):
        self._problem = petsc.LinearProblem(
            a=dfx.fem.form(ufl.lhs(self.F)),
            L=dfx.fem.form(ufl.rhs(self.F)),
            bcs=self._bcs,
            u=self.crack_phase,
            # petsc_options={
            #     "ksp_type": "preonly",
            #     "pc_type": "lu",
            #     "pc_factor_mat_solver_type": "mumps",
            # },
            petsc_options={
                "ksp_type": "minres",
                "pc_type": "hypre",
            },
        )

    def solve(self):
        con = self._constitutive

        # For crack phase field sub problem

        # Util.localProject(
        #     con.getElasticStrainEnergyPositive(),
        #     con.elastic_strain_energy_positive,
        # )
        con.elastic_strain_energy_positive.interpolate(
            dfx.fem.Expression(
                con.getElasticStrainEnergyPositive(),
                con.elastic_strain_energy_positive.function_space.element.interpolation_points(),
            )
        )

        # Util.localProject(
        #     con.getCrackDrivenForce(),
        #     con.crack_driven_force,
        # )
        con.crack_driven_force.interpolate(
            dfx.fem.Expression(
                con.getCrackDrivenForce(),
                con.crack_driven_force.function_space.element.interpolation_points(),
            )
        )

        self._problem.solve()

        self.crack_phase.x.array[:] = np.maximum(
            self.crack_phase.x.array, self.crack_phase_old.x.array
        )

    def write(self, time: float = 0):
        self._crack_phase_vis.interpolate(self.crack_phase)
        self._crack_driven_force_vis.interpolate(self._constitutive.crack_driven_force)

        self.result_file.write(time)

    def __exit__(self, exc_type, exc_value, traceback):
        self.result_file.close()


class DuctileFractureProblem(Problem):
    def __init__(
        self,
        constitutive: Constitutive.DuctileFracture,
        dt: dfx.fem.Constant,
        element_type: str = "Lagrange",
        degree: int = 1,
        dx: ufl.Measure = None,
        metadata: dict = None,
    ):
        super().__init__(
            mesh=constitutive._mesh,
            element_type=element_type,
            degree=degree,
            dx=dx,
            metadata=metadata,
        )

        self._constitutive = constitutive

        assert issubclass(type(self._constitutive), Constitutive.DuctileFracture)

        self.isotropic_plastic_problem = IsotropicPlasticProblem(
            constitutive=constitutive,
            dt=dt,
            element_type=element_type,
            degree=degree,
            dx=dx,
            metadata=metadata,
        )
        self.ductile_fracture_sub_problem = DuctileFractureSubProblem(
            constitutive=constitutive,
            dt=dt,
            element_type=element_type,
            degree=degree,
            dx=dx,
            metadata=metadata,
        )

    def prepare(self):
        super().prepare()
        # assert isinstance(
        #     self._constitutive, Constitutive.DuctileFracturePrincipleStrainDecomposition
        # )
        function_list = [
            self.isotropic_plastic_problem._displacement_vis,
            self.isotropic_plastic_problem._equivalent_plastic_strain_vis,
            self.isotropic_plastic_problem._stress_vis,
            self.isotropic_plastic_problem._equivalent_stress_vis,
            self.isotropic_plastic_problem._yield_stress,
            self.isotropic_plastic_problem._hardening,
            self.isotropic_plastic_problem._elastic_strain_vis,
            self.isotropic_plastic_problem._elastic_strain_energy_positive_vis,
            self.isotropic_plastic_problem._plastic_work_vis,
            self.isotropic_plastic_problem._acceleration_vis,
            self.isotropic_plastic_problem._velocity_vis,
            self.ductile_fracture_sub_problem._crack_phase_vis,
            self.ductile_fracture_sub_problem._crack_driven_force_vis,
            # self._constitutive._principle_strain_vis,
            # self._constitutive.plastic_strain_vector,
        ]
        if isinstance(
            self._constitutive, Constitutive.DuctileFracturePrincipleStrainDecomposition
        ):
            function_list.append(self._constitutive._principle_strain_vis)

        self.result_file = dfx.io.VTXWriter(
            self._comm,
            self.result_dir / (self.result_filename + ".bp"),
            function_list,
            engine="BP4",
        )
        self.result_file.write(0)

        if (
            self._mesh.comm.rank == 0
        ):  # Shall be changed to the host rank, think about a better way
            # TODO: Change write and log to a standalone class
            self.log_file = open(
                self.result_dir / self.isotropic_plastic_problem.log_filename, "w"
            )

            self._convergence_progress_bar = tqdm.tqdm(
                total=100,
                # desc="Convergence",
                position=0,
                leave=True,
                bar_format="{desc}{percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {postfix}]",
            )

        # self.ductile_fracture_sub_problem.result_dir = self.result_dir

        # self.ductile_fracture_sub_problem.prepare()

    def assemble(self):
        self.isotropic_plastic_problem.assemble()
        self.ductile_fracture_sub_problem.assemble()

    # @profile(stream=open(result_dir / f"memory/solve_{rank}.txt", "w"))
    def solve(self):
        con = self._constitutive
        assert isinstance(
            con, Constitutive.DuctileFracture
        ), (
            "Constitutive is not DuctileFracture"
        )  # Type check just for syntax highlighting and autocompletion

        num_iteration = 0
        correction_norm = 0
        converged = False
        host = 0
        rank = self._comm.Get_rank()
        pp = self.isotropic_plastic_problem
        pp._time += pp.dt.value

        pp.displacement_inc.x.array[:] = 0.0

        if pp.iteration_out:
            iteration_result = dfx.io.XDMFFile(
                self._comm,
                self.result_dir / f"iteration_{pp._solve_call_time}.xdmf",
                "w",
            )
            iteration_result.write_mesh(self._mesh)

        if rank == host:
            inc_info = (
                Util.getTimeStr()
                + f"Increment {pp._solve_call_time}, time {pp._time:.3e}"
            )
            if pp.verbose:
                print(inc_info)
            self.log_file.write(inc_info + "\n")
            if pp.convergence_plot:
                correction_norm_history = []
                fig, ax = plt.subplots()
                ax.set_title("Convergence Plot")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Residual (log scale)")
                ax.set_yscale("log")
                ax.grid(True)

                # Use Agg backend for non-interactive plotting
                canvas = FigureCanvas(fig)

            self._convergence_progress_bar.reset()
            self._convergence_progress_bar.set_description(
                f"Increment {pp._solve_call_time}"
            )
            # self._convergence_progress_bar.bar_format =

            timer = Util.Timer()
            total_timer = Util.Timer()

        # @profile(
        #     stream=open(
        #         result_dir / f"memory/it_solve_{pp._solve_call_time}_{rank}.txt", "a"
        #     )
        # )
        def it_solve():
            if rank == host:
                timer.reset()
            # Util.localProject(
            #     con.getYieldStress(),
            #     con.yield_stress,
            # )
            # Util.localProject(
            #     con.getHardening(),
            #     con.hardening,
            # )
            con.yield_stress.interpolate(
                dfx.fem.Expression(
                    con.getYieldStress(),
                    con.S.element.interpolation_points(),
                )
            )
            con.hardening.interpolate(
                dfx.fem.Expression(
                    con.getHardening(),
                    con.S.element.interpolation_points(),
                )
            )

            with pp._b.localForm() as b_loc:
                b_loc.set(0.0)
            pp._A.zeroEntries()
            petsc.assemble_matrix(
                pp._A,
                pp._a,
                bcs=pp._bcs,
            )
            pp._A.assemble()

            petsc.assemble_vector(pp._b, pp._L)
            petsc.apply_lifting(
                pp._b,
                [pp._a],
                [pp._bcs],
                x0=[pp.displacement_inc.x.petsc_vec],
            )
            pp._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(
                b=pp._b,
                bcs=pp._bcs,
                x0=pp.displacement_inc.x.petsc_vec,
            )

            pp._solver.solve(
                pp._b,
                pp.iteration_correction.x.petsc_vec,
            )
            pp.iteration_correction.x.scatter_forward()

            pp.displacement_inc.x.array[:] += pp.iteration_correction.x.array[:]
            strain_inc = self._constitutive.getStrain(pp.displacement_inc)
            (
                _stress_vector,
                # _elastic_strain_vector,
                _equivalent_plastic_strain_inc,
                _plastic_work_inc,
            ) = con.stressProjection(strain_inc)

            # Util.localProject(_stress_vector, con.stress_vector)
            con.stress.interpolate(
                dfx.fem.Expression(
                    _stress_vector,
                    con.W.element.interpolation_points(),
                )
            )

            # Util.localProject(
            #     pp._getAcceleration(
            #         pp.displacement_inc,
            #         pp.displacement_inc_old,
            #         pp.displacement_inc_old2,
            #         pp.dt,
            #     ),
            #     pp.acceleration_inc,
            # )
            pp.acceleration_inc.interpolate(
                dfx.fem.Expression(
                    pp._getAcceleration(
                        pp.displacement_inc,
                        pp.displacement_inc_old,
                        pp.displacement_inc_old2,
                        pp.dt,
                    ),
                    pp.V.element.interpolation_points(),
                )
            )
            pp.acceleration.x.array[:] = (
                pp.acceleration_old.x.array[:] + pp.acceleration_inc.x.array[:]
            )

            # Util.localProject(
            #     pp._getVelocity(
            #         pp.displacement_inc,
            #         pp.displacement_inc_old,
            #         pp.dt,
            #     ),
            #     pp.velocity_inc,
            # )
            pp.velocity_inc.interpolate(
                dfx.fem.Expression(
                    pp._getVelocity(
                        pp.displacement_inc,
                        pp.displacement_inc_old,
                        pp.dt,
                    ),
                    pp.V.element.interpolation_points(),
                )
            )
            pp.velocity.x.array[:] = (
                pp.velocity_old.x.array[:] + pp.velocity_inc.x.array[:]
            )

            # Util.localProject(_elastic_strain_vector, con.elastic_strain_vector)
            # con.elastic_strain_vector.interpolate(
            #     dfx.fem.Expression(
            #         _elastic_strain_vector,
            #         con.W.element.interpolation_points(),
            #     )
            # )

            # Util.localProject(
            #     Util.macaulayBracket(_equivalent_plastic_strain_inc),
            #     con.equivalent_plastic_strain_inc,
            # )
            con.equivalent_plastic_strain_inc.interpolate(
                dfx.fem.Expression(
                    Util.macaulayBracket(_equivalent_plastic_strain_inc),
                    con.S.element.interpolation_points(),
                )
            )
            con.equivalent_plastic_strain.x.array[:] = (
                con.equivalent_plastic_strain_old.x.array[:]
                + con.equivalent_plastic_strain_inc.x.array[:]
            )

            # Util.localProject(
            #     Util.macaulayBracket(_plastic_work_inc)
            #     / (1 - self.ductile_fracture_sub_problem.crack_phase) ** 2,
            #     con.plastic_work_inc,
            # )
            con.plastic_work_inc.interpolate(
                dfx.fem.Expression(
                    Util.macaulayBracket(_plastic_work_inc)
                    / (1 - self.ductile_fracture_sub_problem.crack_phase) ** 2,
                    con.DS.element.interpolation_points(),
                )
            )
            con.plastic_work.x.array[:] = (
                con.plastic_work_old.x.array[:] + con.plastic_work_inc.x.array[:]
            )

            self.ductile_fracture_sub_problem.solve()

            con._crack_phase.x.array[:] = (
                self.ductile_fracture_sub_problem.crack_phase.x.array[:]
            )

        while (not converged) and num_iteration < pp.max_iterations:
            it_solve()
            if pp.iteration_out:
                iteration_result.write_function(
                    pp.iteration_correction, t=num_iteration
                )
                pp.acceleration_inc.name = "Acceleration increment"
                iteration_result.write_function(pp.acceleration_inc, t=num_iteration)
                iteration_result.write_function(pp.acceleration, t=num_iteration)
                iteration_result.write_function(pp.displacement_inc, t=num_iteration)

            # self._comm.Barrier()
            correction_norm_old = correction_norm
            correction_norm = Util.getL2Norm(pp.iteration_correction)
            correction_norm: float = self._comm.allreduce(correction_norm, op=MPI.MIN)
            converged = correction_norm < pp.tolerance
            # converged = self._comm.allreduce(converged, op=MPI.LAND)

            if rank == host:
                it_time = timer.elapsed()

                it_info = (
                    Util.getTimeStr()
                    + f"iteration {num_iteration}: "
                    + f"r = {correction_norm:.3e}, "
                    + f"dr = {(correction_norm - correction_norm_old):.3e}, "
                    + f"elapsed: {it_time:.2f}s"
                )
                if pp.verbose:
                    terminal_size = shutil.get_terminal_size()
                    width = terminal_size.columns
                    # print(" " * width, end=self._log_info_line_end_with, flush=True)
                    print(
                        it_info,
                        # end=self._log_info_line_end_with,
                        flush=True,
                    )
                self.log_file.write(it_info + "\n")
                self.log_file.flush()
                if pp.convergence_plot:
                    ax.plot(
                        correction_norm_history,
                        color="green",
                        label="Residual" if num_iteration == 0 else "",
                    )
                    ax.legend(loc="upper right")
                    canvas.draw()
                    # plt.pause(0.01)

                rate_str = (
                    f"{it_time:.2f}s/it" if it_time > 1 else f"{1/it_time:.2f}it/s"
                )

                self._convergence_progress_bar.set_postfix(
                    {
                        "\b\b": "\b" + rate_str,
                        "it": num_iteration,
                        "res": f"{correction_norm:.3e}",
                    }
                )
                # self._convergence_progress_bar.update(pp.tolerance / correction_norm)
                n = pp.tolerance / correction_norm * 100 if not converged else 100.0
                n -= self._convergence_progress_bar.n
                self._convergence_progress_bar.update(n)
                # self._convergence_progress_bar.

            num_iteration += 1
            self._comm.Barrier()
        else:
            if num_iteration == pp.max_iterations:
                if rank == host:
                    info = (
                        Util.getTimeStr()
                        + f"iteration {num_iteration}: "
                        + f"r = {correction_norm:.3e}, "
                        + f"dr = {(correction_norm - correction_norm_old):.3e}"
                    )
                    if pp.verbose:
                        terminal_size = shutil.get_terminal_size()
                        width = terminal_size.columns
                        print(
                            " " * width,
                            end=pp._log_info_line_end_with,
                            flush=True,
                        )
                        print(info)
                    self.log_file.write(info + "\n")
                    self.log_file.flush()
                raise RuntimeError("Failed to converge")

        if pp.iteration_out:
            iteration_result.close()
        if rank == host:
            info = (
                Util.getTimeStr()
                + f"Converged at iteration {num_iteration-1}: "
                # + f"r = {correction_norm:.3e}, "
                + f"elapsed: {total_timer}, "
                + f"rate: {num_iteration / total_timer.elapsed():.2f}it/s\n"
            )
            if pp.verbose:
                # print(" " * width, end=self._log_info_line_end_with, flush=True)
                print(info)
            self.log_file.write(info + "\n")
            self.log_file.flush()

        if pp.convergence_plot:
            correction_norm_history = []
            plt.close(fig)

        # Update solution
        pp._solve_call_time += 1

        pp.displacement.x.array[:] += pp.displacement_inc.x.array[:]
        con.stress_old.x.array[:] = con.stress.x.array[:]

        pp.displacement_inc_old3.x.array[:] = pp.displacement_inc_old2.x.array[:]
        pp.displacement_inc_old2.x.array[:] = pp.displacement_inc_old.x.array[:]
        pp.displacement_inc_old.x.array[:] = pp.displacement_inc.x.array[:]

        pp.acceleration_old.x.array[:] = pp.acceleration.x.array[:]
        pp.velocity_old.x.array[:] = pp.velocity.x.array[:]

        con.elastic_strain_old.x.array[:] = con.elastic_strain.x.array[:]
        con.equivalent_plastic_strain_old.x.array[:] = (
            con.equivalent_plastic_strain.x.array[:]
        )
        con.plastic_work_old.x.array[:] = con.plastic_work.x.array[:]

        deviatoric_stress = ufl.dev(con.stress)
        # Util.localProject(
        #     ufl.sqrt(
        #         1.5
        #         * ufl.inner(
        #             deviatoric_stress,
        #             deviatoric_stress,
        #         )
        #     ),
        #     con.equivalent_stress,
        # )
        con.equivalent_stress.interpolate(
            dfx.fem.Expression(
                ufl.sqrt(
                    1.5
                    * ufl.inner(
                        deviatoric_stress,
                        deviatoric_stress,
                    )
                ),
                con.S.element.interpolation_points(),
            )
        )

        self.ductile_fracture_sub_problem.crack_phase_old.x.array[:] = (
            self.ductile_fracture_sub_problem.crack_phase.x.array[:]
        )

        self._post_solve()

        return num_iteration

    def _post_solve(self):
        import gc

        opts = PETSc.Options()
        opts.clear()
        opts.destroy()
        del opts

        gc.collect()

    def write(self, t: float = 0):
        con = self._constitutive
        # assert isinstance(
        #     con, Constitutive.DuctileFracturePrincipleStrainDecomposition
        # ), (
        #     "Constitutive is not DuctileFracturePrincipleStrainDecomposition"
        # )  # Type check just for syntax highlighting and autocompletion
        self.isotropic_plastic_problem._displacement_vis.interpolate(
            self.isotropic_plastic_problem.displacement
        )
        self.isotropic_plastic_problem._equivalent_plastic_strain_vis.interpolate(
            con.equivalent_plastic_strain
        )
        self.isotropic_plastic_problem._stress_vis.interpolate(con.stress)
        self.isotropic_plastic_problem._equivalent_stress_vis.interpolate(
            con.equivalent_stress
        )
        self.isotropic_plastic_problem._yield_stress.interpolate(
            dfx.fem.Expression(
                con.yield_stress + con.hardening * con.equivalent_plastic_strain_inc,
                con.S.element.interpolation_points(),
            )
        )
        self.isotropic_plastic_problem._hardening.interpolate(con.hardening)
        self.isotropic_plastic_problem._elastic_strain_vis.interpolate(
            con.elastic_strain
        )
        self.isotropic_plastic_problem._elastic_strain_energy_positive_vis.interpolate(
            con.elastic_strain_energy_positive
        )
        self.isotropic_plastic_problem._plastic_work_vis.interpolate(con.plastic_work)
        self.isotropic_plastic_problem._acceleration_vis.interpolate(
            self.isotropic_plastic_problem.acceleration
        )
        self.isotropic_plastic_problem._velocity_vis.interpolate(
            self.isotropic_plastic_problem.velocity
        )
        self.ductile_fracture_sub_problem._crack_phase_vis.interpolate(
            self.ductile_fracture_sub_problem.crack_phase
        )
        self.ductile_fracture_sub_problem._crack_driven_force_vis.interpolate(
            con.crack_driven_force
        )

        if isinstance(con, Constitutive.DuctileFracturePrincipleStrainDecomposition):
            con._principle_strain_vis.x.array[:] = con.principle_strain.flatten()

        self.result_file.write(t)

    def __exit__(self, exc_type, exc_value, traceback):
        self.result_file.close()

        if self._mesh.comm.rank == 0:
            self._convergence_progress_bar.close()
