#-----------------------------------------------------------------------------
# Copyright (c) 2012, Enthought, Inc.
# All rights reserved.  See LICENSE.txt for details.
# 
# Author: Kurt W. Smith
# Date: 26 March 2012
#-----------------------------------------------------------------------------

'''
julia.py

Compute and plot the Julia set.

This provides a self-contained---if somewhat contrived---example for comparing
the runtimes between pure Python, Numpy, Cython, and Cython-wrapped C versions
of the julia set calculation.

It is meant to be run from the command line; run

    $ python julia.py -h

for details.

'''

# --- Python / Numpy imports -------------------------------------------------
import numpy as np
import pylab as pl
from time import time

# --- Local imports ----------------------------------------------------------
# These are distinguished by being prefixed with an underscore, the idea being
# that these functions are part of an extension module and aren't meant to be
# used directly from external facing Python code...
from _julia import (_julia_kernel, _compute_julia_no_opt, 
                    _compute_julia_opt, _compute_julia_ext)

def julia_kernel(z, c, lim, cutoff=1e6):
    ''' Computes the number, `n`, of iterations necessary such that 
    |z_n| > `lim`, where `z_n = z_{n-1}**2 + c`.
    '''
    count = 0
    while abs(z) < lim and count < cutoff:
        z = z * z + c
        count += 1
    return count


def compute_julia_python(c, N, bound=2, lim=1000., kernel=julia_kernel):
    ''' Pure Python calculation of the Julia set for a given `c`.  No NumPy
    array operations are used.
    '''
    julia = np.empty((N, N), dtype=np.uint32)
    grid = np.linspace(-bound, bound, N)
    c = complex(c)
    t0 = time()
    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            julia[i,j] = kernel(x+y*1j, c, lim)
    return julia, time() - t0


def compute_julia_numpy(c, N, bound=2, lim=1000.):
    ''' Pure Python calculation of the Julia set for a given `c` using NumPy
    array operations.
    '''
    orig_err = np.seterr()
    np.seterr(over='ignore', invalid='ignore')
    julia = np.zeros((N, N), dtype=np.uint32)
    X, Y = np.ogrid[-bound:bound:N*1j, -bound:bound:N*1j]
    iterations = X + Y * 1j
    count = 0
    esc_mask = np.zeros_like(julia, dtype=bool)
    t0 = time()
    while not np.all(esc_mask):
        new_mask = ~esc_mask & (np.abs(iterations) >= lim)
        julia[new_mask] = count
        esc_mask |= new_mask
        count += 1
        iterations = iterations**2 + c
    np.seterr(**orig_err)
    return julia, time() - t0


def printer(label, runtime, speedup):
    ''' Given a label, the total runtime in seconds, and a speedup value,
    prints things nicely to stdout.
    '''
    from sys import stdout
    print "{}:".format(label.strip())
    fs =  "    {:.<15s} {: >6.2g}"
    print fs.format("runtime (s)", runtime)
    print fs.format("speedup", speedup)
    print
    stdout.flush()

# some good c values:
# (-0.1 + 0.651j)
# (-0.4 + 0.6j) 
# (0.285 + 0.01j)

def plot_julia(kwargs, compute_julia):
    ''' Given parameters dict in `kwargs` and a function to compute the Julia
    set (`compute_julia`), plots the resulting Julia set with appropriately
    labeled axes.
    '''
    kwargs = kwargs.copy()

    def _plotter(kwargs):
        bound = kwargs['bound']
        julia, _ = compute_julia(**kwargs)
        julia = np.log(julia)
        pl.imshow(julia, 
                  interpolation='nearest',
                  extent=(-bound, bound)*2)
        pl.colorbar()
        title = r"Julia set for $C={0.real:5.3f}+{0.imag:5.3f}i$ $[{1}\times{1}]$"
        pl.title(title.format(kwargs['c'], kwargs['N']))
        pl.xlabel("$Re(z)$")
        pl.ylabel("$Im(z)$")

    pl.figure(figsize=(14, 12))

    cvals = [0.285+0.01j, -0.1+0.651j, -0.4+0.6j, -0.8+0.156j]
    subplots = ['221',    '222',       '223',     '224'      ]

    for c, sp in zip(cvals, subplots):
        kwargs.update(c=c)
        pl.subplot(sp)
        _plotter(kwargs)

    pl.show()


def compare_runtimes(kwargs):
    ''' Given a parameter dict `kwargs`, runs different implementations of the
    Julia set computation and compares the runtimes of each.
    '''

    ref_julia, python_time = compute_julia_python(**kwargs)
    printer("Python only", python_time, 1.0)

    _, numpy_time = compute_julia_numpy(**kwargs)
    assert np.allclose(ref_julia, _)
    printer("Python only + Numpy expressions", numpy_time,
            python_time / numpy_time)

    _, cython_kernel_time = compute_julia_python(kernel=_julia_kernel, **kwargs)
    assert np.allclose(ref_julia, _)
    printer("Python + cythonized kernel", cython_kernel_time, 
            python_time / cython_kernel_time)

    _, cython_no_opt_time = _compute_julia_no_opt(**kwargs)
    assert np.allclose(ref_julia, _)
    printer("All Cython, no optimizations", cython_no_opt_time, 
            python_time / cython_no_opt_time)

    _, cython_opt_time = _compute_julia_opt(**kwargs)
    assert np.allclose(ref_julia, _)
    printer("All Cython, Numpy optimizations", cython_opt_time,
            python_time / cython_opt_time)

    _, ext_opt_time = _compute_julia_ext(**kwargs)
    assert np.allclose(ref_julia, _)
    printer("All C version, wrapped with Cython", ext_opt_time,
            python_time / ext_opt_time)

def main(args):
    ''' The main entry point; branches on whether `args.action` is "plot" or
    "compare".
    '''
    bound = 1.5
    kwargs = dict(c=(0.285 + 0.01j),
                  N=args.N,
                  bound=bound)

    if args.action == 'plot':
        plot_julia(kwargs, _compute_julia_ext)
    elif args.action == 'compare':
        compare_runtimes(kwargs)

description = """ Explore the performance characteristics of Cython and Numpy
when computing the Julia set."""

help_arg_n = """ The number of grid points in each dimension; larger for more
resolution.  (default 100)) """

help_arg_a = """ Either *plot* an approximation of a Julia set with resolution
N (default), or *compare* the runtimes for different implementations.) """

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description=description)

    parser.add_argument('-N', type=int, default=100, help=help_arg_n)
    parser.add_argument('-a', '--action', type=str, 
                        default='plot', 
                        choices=('plot', 'compare'),
                        help=help_arg_a)

    args = parser.parse_args()
    main(args)
