from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

ext = Extension("_julia", 
                ["_julia.pyx", "_julia_ext.c"],
                extra_compile_args=["-fopenmp"])

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext],
    include_dirs = [np.get_include()],
)
