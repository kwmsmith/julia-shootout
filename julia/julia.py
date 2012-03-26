import numpy as np
import pylab as pl
from time import time
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
    mask = (iterations >= lim)
    t0 = time()
    while not np.all(mask):
        new_mask = ~mask & (np.abs(iterations) >= lim)
        julia[new_mask] = count
        mask |= new_mask
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
        pl.title(r"Julia set for $C={0.real:5.3f}+{0.imag:5.3f}i$ "
        r"$[{1}\times{1}]$".format(kwargs['c'], kwargs['N']))
        pl.xlabel("$Re(z)$")
        pl.ylabel("$Im(z)$")

    pl.figure(figsize=(14, 12))

    pl.subplot('221')
    _plotter(kwargs)

    kwargs.update(c=-0.1 + 0.651j)
    pl.subplot('222')
    _plotter(kwargs)

    kwargs.update(c=-0.4 + 0.6j)
    pl.subplot('223')
    _plotter(kwargs)

    kwargs.update(c=-0.8 + 0.156j)
    pl.subplot('224')
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

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description=("Explore the performance characteristics "
                                         "of Cython and Numpy when computing "
                                         "the Julia set."))

    parser.add_argument('-N', type=int, default=100,
            help=("The number of grid points in each dimension;"
            "larger for more resolution."))
    parser.add_argument('-a', '--action', type=str, 
            default='plot', choices=('plot', 'compare'),
            help=("Either *plot* an approximation of a Julia set "
            "with resolution N (default), or *compare* the runtimes "
            "for different implementations."))

    args = parser.parse_args()
    main(args)
