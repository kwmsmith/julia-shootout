import numpy as np
import pylab as pl
from time import time
from _julia import (_julia_kernel, _compute_julia_no_opt, 
                    _compute_julia_opt, _compute_julia_ext)

def julia_kernel(z, c, lim, cutoff=1e6):
    count = 0
    while abs(z) < lim and count < cutoff:
        z = z * z + c
        count += 1
    return count

def compute_julia(c, N, bound=2, lim=1000., kernel=julia_kernel):
    ''' '''
    julia = np.empty((N, N), dtype=np.uint32)
    grid = np.linspace(-bound, bound, N)
    c = complex(c)
    t0 = time()
    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            julia[i,j] = kernel(x+y*1j, c, lim)
    return julia, time() - t0


def compute_julia_numpy(c, N, bound=2, lim=1000.):
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
    return julia, time() - t0

def printer(label, runtime, speedup):
    print "{}:".format(label.strip())
    fs =  "    {: >10s}: {: >6.2f}"
    print fs.format("runtime", runtime)
    print fs.format("speedup", speedup)
    print

# some good c values:
# (-0.1 + 0.651j)
# (-0.4 + 0.6j) 
# (0.285 + 0.01j)

def main():
    ''' '''
    bound = 1.5
    kwargs = dict(c=(0.285 + 0.01j),
                  N=500,
                  bound=bound)

    ref_julia, python_time = compute_julia(**kwargs)
    printer("Python only", python_time, 1.0)

    _, numpy_time = compute_julia_numpy(**kwargs)
    assert np.allclose(ref_julia, _)
    printer("Pure Numpy (no Cython)", numpy_time,
            python_time / numpy_time)

    _, cython_kernel_time = compute_julia(kernel=_julia_kernel, **kwargs)
    assert np.allclose(ref_julia, _)
    printer("Cythonized kernel", cython_kernel_time, 
            python_time / cython_kernel_time)

    _, cython_no_opt_time = _compute_julia_no_opt(**kwargs)
    assert np.allclose(ref_julia, _)
    printer("All Cython, no optimizations ", cython_no_opt_time, 
            python_time / cython_no_opt_time)

    _, cython_opt_time = _compute_julia_opt(**kwargs)
    assert np.allclose(ref_julia, _)
    printer("All Cython, Numpy optimizations ", cython_opt_time,
            python_time / cython_opt_time)

    _, ext_opt_time = _compute_julia_ext(**kwargs)
    # FIXME: wrap the double * array correctly...
    # assert np.allclose(ref_julia, _)
    printer("External library version", ext_opt_time,
            python_time / ext_opt_time)

    kwargs.update(N=2500)

    julia, _ = _compute_julia_opt(**kwargs)

    julia = np.log(julia)
    pl.imshow(julia, 
              interpolation='nearest',
              extent=(-bound, bound)*2)
    pl.colorbar()

if __name__ == '__main__':
    main()
    pl.show()
