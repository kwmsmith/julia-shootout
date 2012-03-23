#include "_julia_ext.h"
#include <stdlib.h>

#define CABS_SQ(z) (creal(z) * creal(z) + cimag(z) * cimag(z))

unsigned int
julia_kernel(double complex z,
             double complex c,
             double lim,
             double cutoff)
{
    unsigned int count = 0;
    double lim_sq = lim * lim;

    while(CABS_SQ(z) < lim_sq && count < cutoff) {
        z = z * z + c;
        count++;
    }
    return count;
}

unsigned int *
compute_julia(double complex c,
              unsigned int N,
              double bound,
              double lim)
{
    int i, j, idx;
    double step, x, y;

    unsigned int *julia = NULL;
    double *grid = NULL;

    julia = (unsigned int*)malloc(N * N * sizeof(unsigned int));
    if(!julia) {
        return NULL;
    }

    grid = (double*)malloc(N * sizeof(double));
    if(!grid) {
        goto fail_grid;
    }

    step = (2.0 * bound) / (N-1);
    for(i=0; i < N; i++) {
        grid[i] = -bound + i * step;
    }

    for(i=0; i < N; i++) {
        x = grid[i];
        for(j=0; j < N; j++) {
            y = grid[j];
            idx = j + N * i;
            julia[idx] = julia_kernel(x + y * I, c, lim, 1e6);
        }
    }

    goto success;

fail_grid:
    if(julia)
        free(julia);
success:
    if(grid)
        free(grid);
    return julia;
}
