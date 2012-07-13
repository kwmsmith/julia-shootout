#-----------------------------------------------------------------------------
# Copyright (c) 2012, Enthought, Inc.
# All rights reserved.  See LICENSE.txt for details.
# 
# Author: Kurt W. Smith
# Date: 26 March 2012
#-----------------------------------------------------------------------------

# --- Python std lib imports -------------------------------------------------
from time import time
import numpy as np

# --- Cython cimports --------------------------------------------------------
cimport cython
from libc.stdlib cimport free
from cpython.cobject cimport PyCObject_FromVoidPtr
cimport numpy as cnp

# --- Local Ctypedefs --------------------------------------------------------
# Necessary to declare these typedefs here (not in the cdef extern block).
# NOTE: Make sure these stay in sync with the _julia_ext.h header typedefs!!

ctypedef double complex cpx_t
ctypedef double         real_t

#-----------------------------------------------------------------------------
# External declarations
#-----------------------------------------------------------------------------
cdef extern from "_julia_ext.h" nogil:

    unsigned int ext_julia_kernel "julia_kernel"(cpx_t, cpx_t, 
                                                 real_t, real_t)
    unsigned int *ext_compute_julia "compute_julia"(cpx_t, unsigned int,
                                               real_t, real_t)

# Necessary to call `np.import_array()` before calling functions from the NumPy
# C-API.
cnp.import_array()

#-----------------------------------------------------------------------------
# Cython functions
#-----------------------------------------------------------------------------
def compute_julia_no_opt(cpx_t c,
                         unsigned int N,
                         real_t bound=1.5,
                         real_t lim=1000.):
    ''' 
    Cythonized version of a pure Python implementation of the compute_julia()
    function.  It uses numpy arrays, but does not use any extra syntax to speed
    things up beyond simple type declarations.

    '''
    cdef int i, j
    cdef real_t x, y
    julia = np.empty((N, N), dtype=np.uint32)
    grid = np.linspace(-bound, bound, N)
    t0 = time()
    for i in range(N):
        x = grid[i]
        for j in range(N):
            y = grid[j]
            julia[i,j] = kernel(x+y*1j, c, lim)
    return julia, time() - t0

cdef inline real_t cabs_sq(cpx_t z) nogil:
    ''' Helper inline function, computes the square of the abs. value of the
    complex number `z`.
    '''
    return z.real * z.real + z.imag * z.imag

cpdef unsigned int kernel(cpx_t z, 
                                 cpx_t c,
                                 real_t lim,
                                 real_t cutoff=1e6) nogil:
    ''' Cython implementation of the kernel computation.

    This is implemented so that no C-API calls are made inside the function
    body.  Even still, there is some overhead as compared with a pure C
    implementation.
    '''
    cdef unsigned int count = 0
    cdef real_t lim_sq = lim * lim
    while cabs_sq(z) < lim_sq and count < cutoff:
        z = z * z + c
        count += 1
    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_julia_opt(cpx_t c,
                       unsigned int N,
                       real_t bound=1.5,
                       real_t lim=1000.):
    '''
    Cython `compute_julia()` implementation with Numpy array buffer
    declarations and appropriate compiler directives.  The body of this
    function is nearly identical to the `compute_julia_no_opt()` function.

    '''

    cdef cnp.ndarray[cnp.uint32_t, ndim=2, mode='c'] julia 
    cdef cnp.ndarray[real_t, ndim=1, mode='c'] grid
    cdef unsigned int i, j
    cdef real_t x, y

    julia = np.empty((N, N), dtype=np.uint32)
    grid = np.linspace(-bound, bound, N)
    t0 = time()
    for i in range(N):
        x = grid[i]
        for j in range(N):
            y = grid[j]
            julia[i,j] = kernel(x+y*1j, c, lim)
    return julia, time() - t0

def compute_julia_ext(cpx_t c,
                       unsigned int N,
                       real_t bound=1.5,
                       real_t lim=1000.):
    '''
    Call an externally implemented version of `compute_julia()` and wrap the
    resulting C array in a NumPy array.

    '''
    t0 = time()
    cdef unsigned int *julia = ext_compute_julia(c, N, bound, lim)
    cdef cnp.npy_intp dims[2]
    dims[0] = N; dims[1] = N
    arr = new_array_owns_data_cython(2, dims, cnp.NPY_UINT, <void*>julia)
    return arr, time() - t0


cdef void local_free(void *data):
    ''' Wraps `free()` from C's stdlib with some output to indicate that it's
    been called.
    '''
    free(data)

cdef object new_array_owns_data(int nd,
                                cnp.npy_intp *dims,
                                int typenum,
                                void *data):
    ''' Creates a Numpy array with data from the `data` buffer.  Sets the array
    base appropriately using `PyCObject_FromVoidPtr()` to ensure that the data
    gets cleaned up when the Numpy array object is garbage collected.

    '''
    arr = cnp.PyArray_SimpleNewFromData(nd, dims, typenum, data)
    cnp.set_array_base(arr, PyCObject_FromVoidPtr(data, local_free))
    return arr

cdef class _dealloc_shim:
    ''' Deallocation shim class that exists simply to free() the _data pointer.
    '''
    cdef void *_data

    def __cinit__(self):
        self._data = NULL

    def __dealloc__(self):
        if self._data:
            free(self._data)
        self._data = NULL

cdef object new_array_owns_data_cython(int nd,
                                       cnp.npy_intp *dims,
                                       int typenum,
                                       void *data):
    ''' Same as `new_array_owns_data()`, but uses a `_dealloc_shim` instance
    rather than `PyCObject_FromVoidPtr()`.  This solution is usable from all
    Python versions (Python 2 and 3), whereas PyCObject_FromVoidPtr() is only
    valid in Python 2.

    '''
    arr = cnp.PyArray_SimpleNewFromData(nd, dims, typenum, data)
    cdef _dealloc_shim dd = _dealloc_shim()
    dd._data = data
    cnp.set_array_base(arr, dd)
    return arr
