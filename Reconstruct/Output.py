import dolfinx as dfx

from pathlib import Path


class VtxWriter:
    def __init__(
        self,
        mesh: dfx.mesh.Mesh,
        filename: str | Path,
        function_list: list[dfx.fem.Function] = None,
    ) -> None:
        self._mesh = mesh
        self._filename = filename

        if function_list is None:
            self._function_list = []
        else:
            self._function_list = function_list

        tdim = self._mesh.topology.dim
        self._S = dfx.fem.functionspace(mesh, "Lagrange", 1)
        self._V = dfx.fem.functionspace(mesh, ("Lagrange", 1, tdim))
        # self._W = dfx.fem.functionspace(mesh, ("Lagrange", 1, tdim * 2)) #TODO: Find out how to implement this
        self._V.element()

    #TODO: Implement this
    def addFunction(self, function: dfx.fem.Function) -> None:
        self._function_list.append(dfx.fem.Function())

    def addFunctions(self, functions: list[dfx.fem.Function]) -> None:
        self._function_list.extend(functions)
