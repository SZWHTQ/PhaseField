import pathlib


class Material:

    def __init__(self):
        # Mechanical properties
        self.rho = 7e-6  # Mass Density
        self.lame = 1.2e5  # Lamé coefficient
        self.mu = 8e4  # Shear modulus

        # Crack phase properties
        self.Gc = 2.7  # Critical energy release rate
        self.lc = 0.5  # Characteristic length
        self.eta = 1e-3  # Crack phase viscosity parameter


class Preset:

    def __init__(self, name="Default", material=Material()):
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
mat = Material()
mat.lc = 0.5
high_loading_rate = Preset(name, mat)
high_loading_rate.output_directory = pathlib.Path("result") / name
high_loading_rate.u_r = 0.4
high_loading_rate.end_t = 4e-3
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
