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
                   int intensity) {

    double *old = (double *)malloc(M * M * sizeof(double));
    double *current = (double *)malloc(M * M * sizeof(double));
    double *new = (double *)malloc(M * M * sizeof(double));
    // write_snapshot_serial expects int* (it copies into an unsigned char
    // buffer internally before writing to the PGM file), so keep this as
    // int even though only the low byte is actually used.
    int *color_value = (int *)malloc(M * M * sizeof(int));

    // Clarify the cartesian axes

    // Initialize the matrix in parallel, using a simple domain partitioning, like a rectangle row
    // major one. Here no unbalance registered, since each single square needs to be initialised in
    // the exact same way.

    // Now define a region where the initial pulse will exist.
    int gaussian_pulse_dimension = (int)(M * 0.1);
    int half_side = (int)(gaussian_pulse_dimension * 0.5);

    // min/max only depend on intensity: they are constant across the whole
    // simulation, so compute them once outside the time loop instead of
    // redoing it N times. Computed here (before frame 0) so it can also be
    // used to normalize the very first frame consistently.
    int min_val = -abs(intensity);
    int max_val = abs(intensity);
    int range = 2 * max_val;
    double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {

            // Evaluate only once
            double start_impulse_i_j = initialize_gaussian(i, j, half_side, i0, j0, intensity);

            old[i * M + j] = start_impulse_i_j;
            current[i * M + j] = start_impulse_i_j;

            // Use previously calculated (once) values to rescale and cast to int.
            color_value[i * M + j] =
                rescale_discretize_intensity(start_impulse_i_j, &min_val, &inv_range);
        }
    }

    // No initialization of new, it is a waste.

    double damp = gamma * dt * 0.5;     /* γΔt/2         */
    double factor = 1.0 / (1.0 + damp); /* 1/(1+γΔt/2)   */
    double c2dt2 = c * c * dt * dt;     /* c²Δt²         */
    double inv_dx2 = 1.0 / (dx * dx);

    // Do a save of old on the file using the function
    //  remember path are defined fromn the root Makefile
    write_snapshot_serial(color_value, M, 0, "./damped_wave/openmp/sim/");

    // Here the iteration steps, each one produces a frame
    for (int iter = 1; iter < N; iter++) {
        // I don't think this is exploiting cache locality, (moreover adapt for cache dimension).
        // check if it is dividing the blocks for rows or for cols. Investigate for better caching
        // and domain divisions.
#pragma omp parallel for schedule(static)
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < M - 1; j++) {
                // Leapfrog update with isotropic 9-point Laplacian, implemented in misc.c
                new[i * M + j] =
                    wave_update_9_pts(old, current, i, j, M, factor, damp, c2dt2, inv_dx2);
            }
        }

        // Rescale to unsigned char, clamping to [0,255] to avoid silent
        // wrap-around if the wave amplitude ever exceeds the assumed range.
#pragma omp parallel for schedule(static)
        for (int i = 0; i < M * M; ++i) {
            color_value[i] = rescale_discretize_intensity(new[i], &min_val, &inv_range);
        }

        // Here convert the double heigth of the wave in ints between 0,255, scaling valleys to
        // black, and peaks to white.

        write_snapshot_serial(color_value, M, iter, "./damped_wave/openmp/sim/");
        // Exchange pointers. We need this since if I only point old to new then when i will write
        // new it will overwrite.
        double *temp = old;
        old = current;
        current = new;
        new = temp;
    }

    // After the rotation above, all three original buffers are still
    // reachable through old/current/new (just relabeled): free all of them.
    free(old);
    free(current);
    free(new);
    free(color_value);
}
