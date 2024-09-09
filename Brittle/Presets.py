import pathlib

import Material
import ConstitutiveRelation as cr


class Preset:
    def __init__(self, name="Default", material=Material.Brittle()):
        self.__name = name

        self.mesh_x = 400
        self.mesh_y = 400

        self.output_directory = pathlib.Path("result/Output")
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

        self.material = material

        self.constitutive = cr.ConstitutiveRelation()

    def __str__(self):
        return self.__name


default = Preset()

# Linear presets
name = "HighLoadingRate"
mat = Material.Brittle()
mat.lc = 0.5
high_loading_rate = Preset(name, mat)
high_loading_rate.output_directory = pathlib.Path("result") / name
high_loading_rate.u_r = 0.4
high_loading_rate.end_t = 4e-3
high_loading_rate.num_iterations = 500
high_loading_rate.crack_length = 25
high_loading_rate.constitutive = cr.Elastic_BourdinFrancfort2008(mat)

name = "LowLoadingRate"
low_loading_rate = Preset("LowLoadingRate")
low_loading_rate.output_directory = pathlib.Path("result") / name
low_loading_rate.u_r = 7.5e-2
low_loading_rate.end_t = 7.5e-3
low_loading_rate.num_iterations = 500
low_loading_rate.constitutive = cr.Elastic_BourdinFrancfort2008(
    low_loading_rate.material
)

name = "PureShear"
pure_shear = Preset(name)
pure_shear.output_directory = pathlib.Path("result") / name
pure_shear.load_direction = 0
pure_shear.u_r = 0.125
pure_shear.end_t = 9e-3
pure_shear.num_iterations = 500
pure_shear.constitutive = cr.Elastic_BourdinFrancfort2008(pure_shear.material)


# Nonlinear presets
name = "HighLoadingRate"
mat = Material.Brittle()
mat.lc = 0.5
nl_high_loading_rate = Preset(name, mat)
nl_high_loading_rate.output_directory = pathlib.Path("result/Nonlinear") / name
nl_high_loading_rate.u_r = 0.4
nl_high_loading_rate.end_t = 4e-3
nl_high_loading_rate.num_iterations = 500
nl_high_loading_rate.crack_length = 25
nl_high_loading_rate.constitutive = cr.Elastic_AmorMarigo2009(mat)

name = "LowLoadingRate"
nl_low_loading_rate = Preset("LowLoadingRate")
nl_low_loading_rate.output_directory = pathlib.Path("result/Nonlinear") / name
nl_low_loading_rate.u_r = 7.5e-2
nl_low_loading_rate.end_t = 7.5e-3
nl_low_loading_rate.num_iterations = 500
nl_low_loading_rate.constitutive = cr.Elastic_AmorMarigo2009(
    nl_low_loading_rate.material
)

name = "PureShear"
nl_pure_shear = Preset(name)
nl_pure_shear.output_directory = pathlib.Path("result/Nonlinear") / name
nl_pure_shear.load_direction = 0
nl_pure_shear.u_r = 0.125
nl_pure_shear.end_t = 9e-3
nl_pure_shear.num_iterations = 500
nl_pure_shear.constitutive = cr.Elastic_AmorMarigo2009(nl_pure_shear.material)

# V. Ziaei-Rad, Y.Shen / Comput. Methods Appl. Mech. Engrg.
ziaei_rad_high_loading_rate = Preset("HighLoadingRate")
ziaei_rad_high_loading_rate.output_directory = pathlib.Path("result/ZiaeiRad") / str(
    ziaei_rad_high_loading_rate
)
ziaei_rad_high_loading_rate.mesh_x = 151
ziaei_rad_high_loading_rate.mesh_y = 151
ziaei_rad_high_loading_rate.u_r = 0.2
ziaei_rad_high_loading_rate.end_t = 2e-3
ziaei_rad_high_loading_rate.crack_length = 25
ziaei_rad_high_loading_rate.num_iterations = 200
ziaei_rad_high_loading_rate.material.lc = 1
ziaei_rad_high_loading_rate.constitutive = cr.Elastic_AmorMarigo2009(
    ziaei_rad_high_loading_rate.material
)

ziaei_rad_low_loading_rate = Preset("LowLoadingRate")
ziaei_rad_low_loading_rate.output_directory = pathlib.Path("result/ZiaeiRad") / str(
    ziaei_rad_low_loading_rate
)
ziaei_rad_low_loading_rate.mesh_x = 151
ziaei_rad_low_loading_rate.mesh_y = 151
ziaei_rad_low_loading_rate.u_r = 7e-2
ziaei_rad_low_loading_rate.end_t = 7e-3
ziaei_rad_low_loading_rate.num_iterations = 200
ziaei_rad_low_loading_rate.material.lc = 1
ziaei_rad_low_loading_rate.constitutive = cr.Elastic_AmorMarigo2009(
    ziaei_rad_low_loading_rate.material
)

ziaei_rad_pure_shear = Preset("PureShear")
ziaei_rad_pure_shear.output_directory = pathlib.Path("result/ZiaeiRad") / str(
    ziaei_rad_pure_shear
)
ziaei_rad_pure_shear.mesh_x = 268
ziaei_rad_pure_shear.mesh_y = 268
ziaei_rad_pure_shear.load_direction = 0
ziaei_rad_pure_shear.u_r = 0.125
ziaei_rad_pure_shear.end_t = 9e-3
ziaei_rad_pure_shear.num_iterations = 200
ziaei_rad_pure_shear.material.lc = 1
ziaei_rad_pure_shear.constitutive = cr.Elastic_AmorMarigo2009(
    ziaei_rad_pure_shear.material
)


name = "SpeedTest"
speed_test = Preset(name)
speed_test.output_directory = pathlib.Path("result") / name
speed_test.u_r = 0.02
speed_test.end_t = 2e-4
speed_test.num_iterations = 20
speed_test.save_interval = 20
speed_test.verbose = False
speed_test.out_vtk = False
speed_test.out_xdmf = False
speed_test.animation = False
speed_test.screenshot = False
