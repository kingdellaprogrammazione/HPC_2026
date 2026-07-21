#include "damped_wave/general_functions/misc.h"
#include "external/tinyexpr/tinyexpr.h"
#include <float.h>
#include <math.h>
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

double gaussian_function(int i, int j, int mean_i, int mean_j, double one_over_sigma,
                         double max_intensity) {
    return max_intensity * exp(-0.5 * ((i - mean_i) * (i - mean_i) + (j - mean_j) * (j - mean_j)) *
                               one_over_sigma * one_over_sigma);
}

double initialize_gaussian(int i, int j, int half_pulse_side, int i_0, int j_0,
                           double max_intensity) {

    // Impose a decay of the gaussian to 13% of its maximum at the borders of the pulse zone
    double three_sigma = half_pulse_side;
    double sigma = three_sigma * 0.33;
    double one_over_sigma = 1 / sigma;

    // clang-format off
    if ((i > i_0 - half_pulse_side && i < i_0 + half_pulse_side) &&
        (j > j_0 - half_pulse_side && j < j_0 + half_pulse_side)) {
        // clang-format on
        return gaussian_function(i, j, i_0, j_0, one_over_sigma, max_intensity);
    } else {
        return 0.0;
    }
}

double initialize_delta(int i, int j, int i_0, int j_0, double intensity) {

    // clang-format off
    if (i == i_0 && j == j_0){
        return intensity;
    } else {
        return 0.0;
    }
}

int rescale_discretize_intensity(double actual_intensity, int *min_intensity, double *inv_range) {
    double scaled = (actual_intensity - *min_intensity) * *inv_range;
    if (scaled < 0.0)
        scaled = 0.0;
    else if (scaled > 255.0)
        scaled = 255.0;
    return (int)scaled;
}

int eval_grid_expression(const char *expr, Params *p) {
    double M = p->M;
    double N = p->N;

    te_variable vars[] = {{"M", &M}, {"N", &N}};

    te_expr *e = te_compile(expr, vars, 2, NULL);

    if (!e) {
        printf("Error parsing expression: %s\n", expr);
        return 0;
    }

    int result = (int)te_eval(e);

    te_free(e);

    return result;
}

int read_params(const char *filename, Params *p) {
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return 1;
    }

    char line[256];
    char key[64];
    char expr[128];

    while (fgets(line, sizeof(line), file)) {

        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n')
            continue;

        if (sscanf(line, "%63s = %127s", key, expr) != 2)
            continue;

        if (strcmp(key, "dx") == 0)
            p->dx = atof(expr);
        else if (strcmp(key, "dt") == 0)
            p->dt = atof(expr);
        else if (strcmp(key, "c") == 0)
            p->c = atof(expr);
        else if (strcmp(key, "gamma") == 0)
            p->gamma = atof(expr);
        else if (strcmp(key, "i0") == 0)
            p->i0 = eval_grid_expression(expr, p);
        else if (strcmp(key, "j0") == 0)
            p->j0 = eval_grid_expression(expr, p);
        else if (strcmp(key, "intensity") == 0)
            p->intensity = eval_grid_expression(expr, p);
        else if (strcmp(key, "frame_start") == 0)
            p->frame_start = eval_grid_expression(expr, p);
        else
            fprintf(stderr, "Warning: unknown parameter %s\n", key);
    }

    fclose(file);
    return 0;
}
