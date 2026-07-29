#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include "damped_wave/general_functions/misc.h"

#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Part 3 of the project.
 *
 * Same simulation as the OpenMP version: the wave is still advanced on the CPU
 * with OpenMP, on the greyscale matrix, exactly as before. What changes is the
 * output stage: instead of writing a greyscale PGM, every frame is sent to the
 * GPU, turned into RGB triplets by a CUDA kernel and written as a colour PPM.
 *
 * Same signature as simulate_wave() in openmp_helper.h, so the two can be
 * driven by the same main and compared directly. */
void simulate_wave_cuda(double gamma, double c, double dt, double dx, int M, int N, int i0, int j0,
                        int intensity, char *relative_path_sim_folder);

#endif
