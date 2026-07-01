#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO remember to check if filename chars are sufficient, to explain size_t and unsigned_char use.
void write_snapshot_serial(int *frame_array, int side_dimension, int step, char *sim_folder_path) {
    // Allocate only the necessary
    char filename[16];

    snprintf(filename, sizeof(filename), "frame_%05d.pgm", step);

    char full_path[256];
    // Join here the folder with the name.
    snprintf(full_path, sizeof(full_path), "%s/%s", sim_folder_path, filename);

    FILE *f = fopen(full_path, "wb");
    if (!f) {
        perror("fopen error");
        return;
    }

    // Print metadata at file head
    fprintf(f, "P5\n%d %d\n255\n", side_dimension, side_dimension);

    // Prevents undefined behaviour with size_t
    size_t n = (size_t)side_dimension * side_dimension;

    // Using unsigned_char to occupy exactly 1 byte per character
    unsigned char *buffer = malloc(n);
    if (!buffer) {
        perror("malloc");
        fclose(f);
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        buffer[i] = (unsigned char)frame_array[i];
    }

    fwrite(buffer, sizeof(unsigned char), n, f);
    free(buffer);
    fclose(f);
}

// This function evaluates for a given (i,j) cartesian position the evolved wave amplitude
double wave_update(const double *prev, const double *curr, int i, int j, int M, double factor,
                   double damp, double c2dt2, double inv_dx2) {
    /* Discrete Laplacian — 5-point stencil */
    double lap = (curr[(i + 1) * M + j] + curr[(i - 1) * M + j] + curr[i * M + (j + 1)] +
                  curr[i * M + (j - 1)] - 4.0 * curr[i * M + j]) *
                 inv_dx2;

    /* Damped wave equation update */
    return factor * (2.0 * curr[i * M + j] + (damp - 1.0) * prev[i * M + j] + c2dt2 * lap);
}

// This function evaluates for a given (i,j) cartesian position the evolved wave amplitude
double wave_update_9_pts(const double *prev, const double *curr, int i, int j, int M, double factor,
                         double damp, double c2dt2, double inv_dx2) {

    /* 9-point isotropic Laplacian */
    double lap = (
                     // corners (weight 1)
                     curr[(i - 1) * M + (j - 1)] + curr[(i - 1) * M + (j + 1)] +
                     curr[(i + 1) * M + (j - 1)] + curr[(i + 1) * M + (j + 1)] +

                     // edge neighbors (weight 4)
                     4.0 * curr[(i - 1) * M + j] + 4.0 * curr[(i + 1) * M + j] +
                     4.0 * curr[i * M + (j - 1)] + 4.0 * curr[i * M + (j + 1)] +

                     // center (weight -20)
                     -20.0 * curr[i * M + j]) *
                 (inv_dx2 / 6.0);

    /* Damped wave equation update (unchanged) */
    return factor * (2.0 * curr[i * M + j] + (damp - 1.0) * prev[i * M + j] + c2dt2 * lap);
}
