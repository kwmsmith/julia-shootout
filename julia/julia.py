import numpy as np
import pylab as pl

def julia_kernel(z, c, lim, cutoff=1e6):
    count = 0
    while abs(z) < lim and count < cutoff:
        z = z * z + c
        count += 1
    return count

def compute_julia(c, N, bound=2, lim=1000.):
    ''' '''
    julia = np.empty((N, N), dtype=np.int32)
    grid = np.linspace(-bound, bound, N)
    c = complex(c)
    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            julia[i,j] = julia_kernel(x+y*1j, c, lim)
    return julia

# some good c values:
# (-0.1 + 0.651j)
# (-0.4 + 0.6j) 
# (0.285 + 0.01j)

def main():
    ''' '''
    bound=1.5
    julia = compute_julia(c=(0.285 + 0.01j),
                          N=500,
                          bound=bound)
    julia = np.log(julia)
    pl.imshow(julia, 
              interpolation='bicubic',
              extent=(-bound, bound)*2)
    pl.colorbar()

if __name__ == '__main__':
    main()
    pl.show()
