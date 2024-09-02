import pathlib

from Material import Brittle


class Preset:
    def __init__(self, name="Default", material=Brittle()):
        self.__name = name

        self.output_directory = pathlib.Path("result/Output")
        self.load_direction = 1  # 1 for y-axis uniaxial tension, 0 for pure shear
        self.u_r = 0.4
        self.end_t = 4e-3
        self.num_iterations = 200
        self.verbose = True
        self.out_vtk = False
        self.out_xdmf = True
        self.animation = True
        self.screenshot = True

        self.material = material

    def __str__(self):
        return self.__name


default = Preset()

name = "HighLoadingRate"
mat = Brittle()
mat.lc = 0.5
high_loading_rate = Preset(name, mat)
high_loading_rate.output_directory = pathlib.Path("test") / name
high_loading_rate.u_r = 0.2
high_loading_rate.end_t = 2e-3
high_loading_rate.num_iterations = 500

name = "LowLoadingRate"
low_loading_rate = Preset("LowLoadingRate")
low_loading_rate.output_directory = pathlib.Path("result") / name
low_loading_rate.u_r = 7.5e-2
low_loading_rate.end_t = 7.5e-3
low_loading_rate.num_iterations = 500

name = "PureShear"
pure_shear = Preset(name)
pure_shear.output_directory = pathlib.Path("result") / name
pure_shear.load_direction = 0
pure_shear.u_r = 0.125
pure_shear.end_t = 9e-3
pure_shear.num_iterations = 500
