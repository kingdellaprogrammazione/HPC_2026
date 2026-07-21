#ifndef OPENMP_HELPER_h
#define OPENMP_HELPER_h

#include "damped_wave/general_functions/misc.h"

#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// write the matrix inside the file? this can be done only by one thread due to the problems of I/O
// collective access. moreover no sense since the normal hdd, ssd. Investigate on the possible
// benefits if the HPC implements a parallel filesystem (so no normal hdd). Evaluate slowness of
// this write operation vs calculation time.

void simulate_wave(double gamma, double c, double dt, double dx, int M, int N, int i0, int j0,
                   int intensity, char *relative_path_sim_folder);

#endif OPENMP_HELPER_h