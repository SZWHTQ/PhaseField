import dolfinx as dfx
import dolfinx.fem.petsc as petsc
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import Material
import Constitutive
import Util

from pathlib import Path

import shutil


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


class Displacement(Problem):
    def __init__(
        self,
        constitutive: Constitutive.Constitutive,
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
        self._material = self._constitutive._material

        self.V = dfx.fem.functionspace(
            self._mesh, (self._element_type, self._degree, (self.tdim,))
        )

        self.displacement = dfx.fem.Function(self.V, name="Displacement")


class IsotropicPlasticity(Displacement):
    def __init__(
        self,
        constitutive: Constitutive.IsotropicJohnsonCook,
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

        assert isinstance(self._material, Material.JohnsonCook)
        assert isinstance(self._constitutive, Constitutive.IsotropicJohnsonCook)

        self.displacement_inc = dfx.fem.Function(self.V, name="Displacement increment")
        self.iteration_correction = dfx.fem.Function(
            self.V, name="Iteration correction"
        )  # Iteration correction for displacement in Newton-Raphson method

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        self.F = (
            ufl.inner(
                self._constitutive.getStrain(self.v),
                self._constitutive.getStress(self.u),
            )
            * self.dx
        )
        self.F += (
            ufl.inner(
                self._constitutive.getStrain(self.v),
                self._constitutive.asThreeDimensionalTensor(
                    self._constitutive.stress_vector
                ),
            )
            * self.dx
        )

        ## Config
        self.max_iteration = 100
        self.tolerance = 1e-6
        self.iteration_out = False
        # Can't work properly right now, #TODO: Fix this in the future
        self.convergence_plot = False
        self.result_dir = Path("result")
        self.result_filename = "plastic"
        self.log_filename = "iteration.log"
        self.verbose = False

        self._solve_call_time = 0
        self._log_info_line_end_with = "\r"

        W_vis = dfx.fem.functionspace(
            self._mesh, ("Lagrange", 1, (self.tdim * self.tdim,))
        )
        V_vis = dfx.fem.functionspace(self._mesh, ("Lagrange", 1, (self.tdim,)))
        S_vis = dfx.fem.functionspace(self._mesh, ("Lagrange", 1))
        self._displacement_vis = dfx.fem.Function(V_vis, name="Displacement")
        self._equivalent_plastic_strain_vis = dfx.fem.Function(
            S_vis, name="Equivalent plastic strain"
        )
        self._stress_vector_vis = dfx.fem.Function(W_vis, name="Stress")
        self._equivalent_stress_vis = dfx.fem.Function(S_vis, name="Equivalent stress")
        self._yield_stress = dfx.fem.Function(S_vis, name="Yield stress")
        self._hardening = dfx.fem.Function(S_vis, name="Hardening")

    def prepare(self):
        # Judge if the result directory exists
        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True)

        # Judge if the filename ends with ".bp"
        if self.result_filename.endswith(".bp"):
            self.result_filename = self.result_filename[:-3]

        self.result_file = dfx.io.VTXWriter(
            self._comm,
            self.result_dir / (self.result_filename + ".bp"),
            [
                self._displacement_vis,
                self._equivalent_plastic_strain_vis,
                self._stress_vector_vis,
                self._equivalent_stress_vis,
                self._yield_stress,
                self._hardening,
            ],
            engine="BP4",
        )
        self.result_file.write(0)

        if (
            self._mesh.comm.rank == 0
        ):  # TODO: Shall be changed to the host rank, think about a better way
            self.log_file = open(self.result_dir / self.log_filename, "w")

    def setBCs(self, bcs):
        self._bcs = bcs

    def assemble(self):
        # self._a, self._L = ufl.system(self.F)
        self._a = dfx.fem.form(ufl.lhs(self.F))
        self._L = dfx.fem.form(ufl.rhs(self.F))

        self._A = petsc.create_matrix(self._a)
        self._b = petsc.create_vector(self._L)

        self._solver = PETSc.KSP().create(self._comm)
        self._solver.setOperators(self._A)

        self._solver.setType(PETSc.KSP.Type.PREONLY)
        pc = self._solver.getPC()
        pc.setType(PETSc.PC.Type.LU)

    def solve(self):
        con = self._constitutive
        assert isinstance(
            con, Constitutive.IsotropicJohnsonCook
        ), (
            "Constitutive is not IsotropicJohnsonCook"
        )  # Type check just for syntax highlighting and autocompletion

        num_iteration = 0
        correction_norm = 0
        converged = False
        host = 0
        rank = self._comm.Get_rank()
        self._solve_call_time += 1

        self.displacement_inc.x.array[:] = 0.0

        Util.localProject(
            con._getYieldStress(),
            con.S,
            con.yield_stress,
        )
        Util.localProject(
            con._getHardening(),
            con.S,
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

        while (not converged) and num_iteration < self.max_iteration:
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
            _stress_vector, _n_elastic_vector, _beta, _equivalent_plastic_strain_inc = (
                con.stressProjection(strain_inc)
            )

            Util.localProject(Util.macaulayBracket(_stress_vector), con.W, con.stress_vector)
            Util.localProject(_n_elastic_vector, con.W, con.n_elastic_vector)
            Util.localProject(_beta, con.S, con.beta)

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
            if num_iteration == self.max_iteration:
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

        self.displacement.x.array[:] += self.displacement_inc.x.array[:]
        con.stress_vector_old.x.array[:] = con.stress_vector.x.array[:]
        con.equivalent_plastic_strain.x.array[:] += np.abs(
            Util.localProject(_equivalent_plastic_strain_inc, con.S).x.array[:]
        )
        deviatoric_stress = ufl.dev(con.asThreeDimensionalTensor(con.stress_vector))
        Util.localProject(
            ufl.sqrt(
                1.5
                * ufl.inner(
                    deviatoric_stress,
                    deviatoric_stress,
                )
            ),
            con.S,
            con.equivalent_stress,
        )

        return num_iteration

    def write(self, time: float = 0):
        assert isinstance(
            self._constitutive, Constitutive.IsotropicJohnsonCook
        ), (
            "Constitutive is not IsotropicJohnsonCook"
        )  # Type check just for syntax highlighting and autocompletion
        self._displacement_vis.interpolate(self.displacement)
        self._equivalent_plastic_strain_vis.interpolate(
            self._constitutive.equivalent_plastic_strain
        )
        self._stress_vector_vis.interpolate(self._constitutive.stress_vector)
        self._equivalent_stress_vis.interpolate(self._constitutive.equivalent_stress)
        self._yield_stress.interpolate(self._constitutive.yield_stress)
        self._hardening.interpolate(self._constitutive.hardening)

        self.result_file.write(time)

    def __exit__(self, exc_type, exc_value, traceback):
        self.result_file.close()
