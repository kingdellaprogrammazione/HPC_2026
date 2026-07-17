#include "../general_functions/misc.h"
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

// Initialize the matrix in parallel, using a simple domain partitioning, like a rectangle row major
// one. Here no unbalance registered, since each single square needs to be initialised in the exact
// same way.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (i != i0 || j != j0) {
                old[i * M + j] = 0.0;
                current[i * M + j] = 0.0;
                color_value[i * M + j] = 0;
            } else {
                old[i * M + j] = intensity;
                current[i * M + j] = intensity;
                color_value[i * M + j] = intensity;
            }
        }
    }

    // No initialization of new, it is a waste.

    double damp = gamma * dt * 0.5;     /* γΔt/2         */
    double factor = 1.0 / (1.0 + damp); /* 1/(1+γΔt/2)   */
    double c2dt2 = c * c * dt * dt;     /* c²Δt²         */
    double inv_dx2 = 1.0 / (dx * dx);

    // min/max only depend on intensity: they are constant across the whole
    // simulation, so compute them once outside the time loop instead of
    // redoing it N times. Computed here (before frame 0) so it can also be
    // used to normalize the very first frame consistently.
    double min_val = -abs(intensity);
    double max_val = abs(intensity);
    double range = 2 * max_val;
    double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

    // Normalize the initial impulse the same way every other frame will be
    // normalized. Without this, a negative intensity (a trough) gets cast
    // directly to unsigned char and wraps around to a near-white value,
    // while the same physical height would be mapped to near-black (0) in
    // every subsequent frame -- a visible flash on frame 0.
    if (intensity != 0) {
        double scaled = (intensity - min_val) * inv_range;
        if (scaled < 0.0)
            scaled = 0.0;
        else if (scaled > 255.0)
            scaled = 255.0;
        color_value[i0 * M + j0] = (int)scaled;
    }

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
            double scaled = (new[i] - min_val) * inv_range;
            if (scaled < 0.0)
                scaled = 0.0;
            else if (scaled > 255.0)
                scaled = 255.0;
            color_value[i] = (int)scaled;
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

// Maybe find a way to specify thread number wanted??
// Add checks for the allowed ranges between the inputs
int main(int argc, char *argv[]) {
    if (argc != 10) {
        fprintf(stderr, "Usage: %s M N dx dt c gamma i0 j0 intensity\n", argv[0]);
        fprintf(stderr, "Example: %s 200 200 0.01 0.00001 343 35.5 100 100 54\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    double dx = atof(argv[3]);
    double dt = atof(argv[4]);
    double c = atof(argv[5]);
    double gamma = atof(argv[6]);
    int i0 = atoi(argv[7]);
    int j0 = atoi(argv[8]);
    int intensity = atoi(argv[9]);

    simulate_wave(gamma, c, dt, dx, M, N, i0, j0, intensity);
    return 0;
}