#include "damped_wave/openmp/src/hilbert.h"
#include "damped_wave/general_functions/misc.h"

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// write the matrix inside the file? this can be done only by one thread due to the problems of I/O
// collective access. moreover no sense since the normal hdd, ssd. Investigate on the possible
// benefits if the HPC implements a parallel filesystem (so no normal hdd). Evaluate slowness of
// this write operation vs calculation time.

double wave_update_9_pts_hilbert(const double *prev, const double *curr, const uint32_t *H, int i,
                                 int j, int M, double factor, double damp, double c2dt2,
                                 double inv_dx2) {

    /* 9-point isotropic Laplacian */
    double lap = (
                     // corners (weight 1)
                     curr[H[(i - 1) * M + (j - 1)]] + curr[H[(i - 1) * M + (j + 1)]] +
                     curr[H[(i + 1) * M + (j - 1)]] + curr[H[(i + 1) * M + (j + 1)]] +

                     // edge neighbors (weight 4)
                     4.0 * curr[H[(i - 1) * M + j]] + 4.0 * curr[H[(i + 1) * M + j]] +
                     4.0 * curr[H[i * M + (j - 1)]] + 4.0 * curr[H[i * M + (j + 1)]] +

                     // center (weight -20)
                     -20.0 * curr[H[i * M + j]]) *
                 (inv_dx2 / 6.0);

    /* Damped wave equation update (unchanged) */
    return factor * (2.0 * curr[H[i * M + j]] + (damp - 1.0) * prev[H[i * M + j]] + c2dt2 * lap);
}

void simulate_wave_hilbert(double gamma, double c, double dt, double dx, int M, int N, int i0,
                           int j0, int intensity, char *relative_path_sim_folder) {

    int bits = 0;
    while ((1u << bits) < (uint32_t)M)
        bits++;

    // initialize the hilbert mapping
    uint32_t *H = malloc(M * M * sizeof(uint32_t));
    uint32_t *X = malloc(M * M * sizeof(uint32_t));
    uint32_t *Y = malloc(M * M * sizeof(uint32_t));

#pragma omp parallel for schedule(static)
    for (int x = 0; x < M; x++) {
        for (int y = 0; y < M; y++) {
            uint32_t h = hilbert_encode(bits, x, y);
            H[x * M + y] = h;
            X[h] = x;
            Y[h] = y;
        }
    }

    double *old = (double *)malloc(M * M * sizeof(double));
    double *current = (double *)malloc(M * M * sizeof(double));
    double *new = (double *)calloc(M * M, sizeof(double));
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

            old[H[i * M + j]] = start_impulse_i_j;
            current[H[i * M + j]] = start_impulse_i_j;

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
    write_snapshot_serial(color_value, M, 0, relative_path_sim_folder);

    // Here the iteration steps, each one produces a frame
    for (int iter = 1; iter < N; iter++) {
        // I don't think this is exploiting cache locality, (moreover adapt for cache dimension).
        // check if it is dividing the blocks for rows or for cols. Investigate for better caching
        // and domain divisions.
#pragma omp parallel for schedule(static)
        for (int h = 0; h < M * M; h++) {

            int i = X[h];
            int j = Y[h];

            if (i == 0 || i == M - 1 || j == 0 || j == M - 1)
                continue;

            new[h] =
                wave_update_9_pts_hilbert(old, current, H, i, j, M, factor, damp, c2dt2, inv_dx2);
        }

        // Rescale to unsigned char, clamping to [0,255] to avoid silent
        // wrap-around if the wave amplitude ever exceeds the assumed range.
#pragma omp parallel for schedule(static)
        for (int h = 0; h < M * M; h++) {

            int i = X[h];
            int j = Y[h];

            color_value[i * M + j] = rescale_discretize_intensity(new[h], &min_val, &inv_range);
        }

        // Here convert the double heigth of the wave in ints between 0,255, scaling valleys to
        // black, and peaks to white.

        write_snapshot_serial(color_value, M, iter, relative_path_sim_folder);
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

/* Hilbert curve encode
 *
 * bits: order of the Hilbert curve
 *       grid size = 2^bits x 2^bits
 *
 * x,y:  integer coordinates
 *
 * returns:
 *       Hilbert distance d
 */
uint32_t hilbert_encode(int bits, uint32_t x, uint32_t y) {
    uint32_t d = 0;
    uint32_t tmp;

    for (uint32_t s = (1u << (bits - 1)); s > 0; s >>= 1) {
        uint32_t rx = 0;
        uint32_t ry = 0;

        if (x & s)
            rx = 1;

        if (y & s)
            ry = 1;

        d += s * s * ((3 * rx) ^ ry);

        /*
         * Hilbert rotation
         */
        if (ry == 0) {
            if (rx == 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }

            tmp = x;
            x = y;
            y = tmp;
        }
    }

    return d;
}

/*
 * Hilbert curve decode
 *
 * bits: order of the Hilbert curve
 *
 * d: Hilbert distance
 *
 * returns x,y coordinates through pointers
 */
void hilbert_decode(int bits, uint32_t d, uint32_t *x, uint32_t *y) {
    uint32_t xx = 0;
    uint32_t yy = 0;
    uint32_t tmp;

    uint32_t n = 1u << bits;

    for (uint32_t s = 1; s < n; s <<= 1) {
        uint32_t rx;
        uint32_t ry;

        rx = 1 & (d / 2);
        ry = 1 & (d ^ rx);

        /*
         * Hilbert rotation
         */
        if (ry == 0) {
            if (rx == 1) {
                xx = s - 1 - xx;
                yy = s - 1 - yy;
            }

            tmp = xx;
            xx = yy;
            yy = tmp;
        }

        xx += s * rx;
        yy += s * ry;

        d /= 4;
    }

    *x = xx;
    *y = yy;
}
