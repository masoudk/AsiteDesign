from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = "C_Modules",
  cmdclass = {"build_ext": build_ext},
  ext_modules = [
    Extension("Energy", ["Energy.pyx"], extra_compile_args = ["-O3", "-fopenmp"], extra_link_args=['-fopenmp']),
    #Extension("Gradient", ["Gradient.pyx"], extra_compile_args = ["-O3"]),
  ]
)
