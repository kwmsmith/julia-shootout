/*-----------------------------------------------------------------------------
 * Copyright (c) 2012, Enthought, Inc.
 * All rights reserved.  See LICENSE.txt for details.
 *  
 * Author: Kurt W. Smith
 * Date: 26 March 2012
 --------------------------------------------------------------------------*/

#ifndef __JULIA_EXT_H__
#define __JULIA_EXT_H__

#include <complex.h>

unsigned int julia_kernel(double complex, double complex, double, double);

unsigned int *compute_julia(double complex, unsigned int, double, double);

#endif
