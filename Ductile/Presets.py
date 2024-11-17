import pathlib
# import warnings

import Material
import ConstitutiveRelation as cr


class Preset:
    def __init__(self, name="Default", material=None):
        self.__name = name

        self.output_directory = pathlib.Path("result/Output")

        self.dim = 2

        self.w = 100
        self.h = 100
        self.thickness = 1

        self.mesh_x = 256
        self.mesh_y = 256

        self.load_direction = 1  # 1 for y-axis uniaxial tension, 0 for pure shear
        self.crack_length = 50
        self.u_r = 0.4
        self.end_t = 4e-3
        self.num_iterations = 200
        self.save_interval = None
        self.warp_factor = 0

        self.verbose = True
        self.out_vtk = False
        self.out_xdmf = True
        self.animation = True
        self.screenshot = True

        if material is None:
            self.material = Material.JohnsonCook()
        else:
            # if isinstance(material, Material.Ductile):
            #     self.material = material
            # else:
            #     # warnings.warn("Material should be of type Ductile")
            #     raise TypeError("Material should be of type Ductile")
            self.material = material

        self.constitutive = cr.ElastoPlastic_AmorMarigo2009(self.material)

    def __str__(self):
        return self.__name


default = Preset()

# %% Miehe 2016, gradient plasticity theory, single edge notch shear
name = "Miehe_2016_Shear"
# C. Miehe used for single edge notch shear and bar torsion test
mild_steel = Material.Ductile()
# E = 2e5 MPa, nu = 0.3
mild_steel.lame = 115.384615385e6
mild_steel.mu = 76.923076923e6
mild_steel.eta_f = 0.1
mild_steel.lf = 0.008
mild_steel.wc = 13e3
mild_steel.zeta = 1
mild_steel.y0 = 450e3
mild_steel.y_inf = 600e3
mild_steel.omega_p = 16.96
mild_steel.h = 130e3  # 200e3 for torsion
mild_steel.eta_p = 0.1
mild_steel.lp = 0.016

miehe_2016_shear = Preset(name, mild_steel)
miehe_2016_shear.output_directory = pathlib.Path("result") / name
miehe_2016_shear.w = 1
miehe_2016_shear.mesh_x = 250
miehe_2016_shear.mesh_y = 250
miehe_2016_shear.load_direction = 0
miehe_2016_shear.crack_length = 0.5
miehe_2016_shear.u_r = 60e-3
miehe_2016_shear.end_t = 1e-3
# 10s, just for testing, in that Miehe was on the static case
miehe_2016_shear.num_iterations = 300
# miehe_2016_shear.save_interval = 1 # For debugging

# %% Johnson-Cook test preset, nothing to compare
name = "JohnsonCook"
Ti6Al4V = Material.JohnsonCook()
Ti6Al4V.rho = 4.43e-9
# Ti6Al4V.specific_heat = 560000000
Ti6Al4V.heat_conduction = 6.6
Ti6Al4V.inelastic_heat = 0.9
Ti6Al4V.lame = 68501.40618722378  # 63461538.46153846
Ti6Al4V.mu = 41984.732824427476  # 42307692.307692304
Ti6Al4V.A = 1000.0  # 1e6
Ti6Al4V.B = 1000.0  # 1e6
Ti6Al4V.C = 0.0
Ti6Al4V.n = 1.0
Ti6Al4V.m = 1.0
Ti6Al4V.melting_temperature = 1873.0
Ti6Al4V.w0 = 150.0
Ti6Al4V.Gc = 45.0  # ???
Ti6Al4V.lf = 0.1
Ti6Al4V.eta_f = 5e-5
johnson_cook = Preset(name, Ti6Al4V)
johnson_cook.output_directory = pathlib.Path("result") / name
johnson_cook.dim = 2
johnson_cook.w = 10
johnson_cook.h = 4
johnson_cook.mesh_x = int(4 * johnson_cook.w / Ti6Al4V.lf)  # 400
johnson_cook.mesh_y = int(4 * johnson_cook.h / Ti6Al4V.lf)  # 160
johnson_cook.crack_length = 4

# Shear
johnson_cook.load_direction = 0
johnson_cook.u_r = 0.4
johnson_cook.end_t = 4e-4

# # Tension
# johnson_cook.load_direction = 1
# johnson_cook.u_r = 0.4
# johnson_cook.end_t = 4e-4

johnson_cook.num_iterations = 500
# johnson_cook.save_interval = 5
# johnson_cook.screenshot = False
johnson_cook.warp_factor = 0

# %% Johnson-Cook 3D test preset, nothing to compare
name = "JohnsonCook_3D"
johnson_cook_3d = Preset(name, Ti6Al4V)
johnson_cook_3d.output_directory = pathlib.Path("result") / name
johnson_cook_3d.dim = 3
johnson_cook_3d.w = 10
johnson_cook_3d.h = 4
johnson_cook_3d.thickness = 1
johnson_cook_3d.mesh_x = int(4 * johnson_cook_3d.w / Ti6Al4V.lf)  # 400
johnson_cook_3d.mesh_y = int(4 * johnson_cook_3d.h / Ti6Al4V.lf)  # 160
johnson_cook_3d.crack_length = 4

# Shear
johnson_cook_3d.load_direction = 0
johnson_cook_3d.u_r = 0.4
johnson_cook_3d.end_t = 4e-4

# # Tension
# johnson_cook_3d.load_direction = 1
# johnson_cook_3d.u_r = 0.4
# johnson_cook_3d.end_t = 4e-4

johnson_cook_3d.num_iterations = 2000
johnson_cook_3d.save_interval = 10
johnson_cook_3d.screenshot = False
johnson_cook_3d.warp_factor = 0
