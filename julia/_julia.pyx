# --- Python std lib imports -------------------------------------------------
from time import time
import numpy as np

# --- Cython cimports --------------------------------------------------------
cimport cython
cimport numpy as np

np.import_array()

#-----------------------------------------------------------------------------
# External declarations
#-----------------------------------------------------------------------------
cdef extern from "_julia_ext.h" nogil:
    unsigned int ext_julia_kernel "julia_kernel"(double complex, double complex, 
                                                 double, double)
    unsigned int *ext_compute_julia "compute_julia"(double complex, unsigned int,
                                               double, double)

#-----------------------------------------------------------------------------
# Cython functions
#-----------------------------------------------------------------------------
cdef inline double cabs_sq(double complex z) nogil:
    return z.real * z.real + z.imag * z.imag

cpdef unsigned int _julia_kernel(double complex z, 
                                 double complex c,
                                 double lim,
                                 double cutoff=1e6) nogil:
    cdef unsigned int count = 0
    cdef double lim_sq = lim * lim
    while cabs_sq(z) < lim_sq and count < cutoff:
        z = z * z + c
        count += 1
    return count

def _compute_julia_no_opt(double complex c,
                         unsigned int N,
                         double bound=1.5,
                         double lim=1000.):

    julia = np.empty((N, N), dtype=np.uint32)
    grid = np.linspace(-bound, bound, N)
    t0 = time()
    for i in range(N):
        x = grid[i]
        for j in range(N):
            y = grid[j]
            julia[i,j] = _julia_kernel(x+y*1j, c, lim)
    return julia, time() - t0

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_julia_opt(double complex c,
                       unsigned int N,
                       double bound=1.5,
                       double lim=1000.):

    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] julia 
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] grid

    julia = np.empty((N, N), dtype=np.uint32)
    grid = np.linspace(-bound, bound, N)
    t0 = time()
    for i in range(N):
        x = grid[i]
        for j in range(N):
            y = grid[j]
            julia[i,j] = _julia_kernel(x+y*1j, c, lim)
    return julia, time() - t0

def _compute_julia_ext(double complex c,
                       unsigned int N,
                       double bound=1.5,
                       double lim=1000.):
    t0 = time()
    cdef unsigned int *julia = ext_compute_julia(c, N, bound, lim)
    cdef np.npy_intp dims[2]
    dims[0] = N; dims[1] = N
    arr = np.PyArray_SimpleNewFromData(2, dims, np.NPY_UINT, <void*>julia)
    return arr, time() - t0
