import pathlib
# import warnings

import Material
import ConstitutiveRelation as cr


class Preset:
    def __init__(self, name="Default", material=None):
        self.__name = name

        self.output_directory = pathlib.Path("result/Output")

        self.L = 100

        self.mesh_x = 256
        self.mesh_y = 256

        self.load_direction = 1  # 1 for y-axis uniaxial tension, 0 for pure shear
        self.crack_length = 50
        self.u_r = 0.4
        self.end_t = 4e-3
        self.num_iterations = 200
        self.save_interval = None

        self.verbose = True
        self.out_vtk = False
        self.out_xdmf = True
        self.animation = True
        self.screenshot = True

        if material is None:
            self.material = Material.Ductile()
        else:
            if isinstance(material, Material.Ductile):
                self.material = material
            else:
                # warnings.warn("Material should be of type Ductile")
                raise TypeError("Material should be of type Ductile")

        self.constitutive = cr.Elastic_BourdinFrancfort2008(self.material)

    def __str__(self):
        return self.__name


default = Preset()

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
miehe_2016_shear.L = 1
miehe_2016_shear.mesh_x = 250
miehe_2016_shear.mesh_y = 250
miehe_2016_shear.load_direction = 1
miehe_2016_shear.crack_length = 0.5
miehe_2016_shear.u_r = 6e-3
miehe_2016_shear.end_t = 1
# 10s, just for testing, in that Miehe was on the static case
miehe_2016_shear.num_iterations = 500
