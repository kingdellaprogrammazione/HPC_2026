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

    // Do a save of old on the file using the function
    //  remember path are defined fromn the root Makefile
    write_snapshot_serial(color_value, M, 0, "./damped_wave/openmp/sim/");

    double damp = gamma * dt * 0.5;     /* γΔt/2         */
    double factor = 1.0 / (1.0 + damp); /* 1/(1+γΔt/2)   */
    double c2dt2 = c * c * dt * dt;     /* c²Δt²         */
    double inv_dx2 = 1.0 / (dx * dx);

    // Here the iteration steps, each one produces a frame
    for (int iter = 1; iter < N; iter++) {
        // I don't think this is exploiting cache locality, (moreover adapt for cache dimension).
        // check if it is dividing the blocks for rows or for cols. Investigate for better caching
        // and domain divisions.
#pragma omp parallel for schedule(static)
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < M - 1; j++) {
                // Here call the function that give us the evolution of the wave. This below is only
                // a stupid calculus
                // TODO: implement the correct logic in a new function
                new[i * M + j] =
                    wave_update_9_pts(old, current, i, j, M, factor, damp, c2dt2, inv_dx2);
            }
        }

        // Here we need to gather the min and the max , QUESTION: MAYBE WE SHOULD CALCULATE THE
        // MINMAX ON ALL THE TIME SERIES? IDEA: SINCE THE WAVE IS DAMPED, WE CAN USE ONLY THE FIRST
        // MAX AS THE AMPLITUDE DECREASE GOING FORWARD!!!!!!!!!!! WE  can use the same absolute
        // range
        double min_val = -abs(intensity);
        double max_val = abs(intensity);

        // Then write the color_value

        // Rescale to unsigned char
        // substitute division by reciprocal multiplication !!
        // Check approximation scheme

        double range = 2 * max_val;
        double inv_range;
        if (range > 0.0)
            inv_range = 255.0 / range; // combine the two multiplications
        else
            inv_range = 0.0; // fallback, handled below

#pragma omp parallel for schedule(static)

        for (size_t i = 0; i < M * M; ++i) {
            color_value[i] = (unsigned char)((new[i] - min_val) * inv_range);
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

    free(old);
    free(new);
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
