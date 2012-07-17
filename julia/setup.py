#-----------------------------------------------------------------------------
# Copyright (c) 2012, Enthought, Inc.
# All rights reserved.  See LICENSE.txt for details.
# 
# Author: Kurt W. Smith
# Date: 26 March 2012
#-----------------------------------------------------------------------------

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

extra_args = []
# Comment/Uncomment the following line to disable/enable OpenMP for GCC-ish
# compilers.
# extra_args = ["-fopenmp"]

exts = [Extension("julia_cython", 
                  ["julia_cython.pyx", "_julia_ext.c"],
                  extra_compile_args=extra_args),
        Extension("julia_cython_solution",
                  ["julia_cython_solution.pyx", "_julia_ext.c"],
                  extra_compile_args=extra_args),
        ]


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = exts,
    include_dirs = [np.get_include()],
)
