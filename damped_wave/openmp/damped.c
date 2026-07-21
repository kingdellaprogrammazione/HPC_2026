#include "damped_wave/general_functions/misc.h"

#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "external/tinyexpr/tinyexpr.h"

typedef struct {
    int M;
    int N;
    double dx;
    double dt;
    double c;
    double gamma;
    int i0;
    int j0;
    int intensity;
    int time_start;
} Params;

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
        else if (strcmp(key, "time_start") == 0)
            p->time_start = eval_grid_expression(expr, p);
        else
            fprintf(stderr, "Warning: unknown parameter %s\n", key);
    }

    fclose(file);
    return 0;
}

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

// Maybe find a way to specify thread number wanted??
// Add checks for the allowed ranges between the inputs

int main(int argc, char *argv[]) {

    Params first_wave_params;

    // Read M and N from command line
    if (argc != 3) {
        printf("Usage: %s M N\n", argv[0]);
        return 1;
    }

    first_wave_params.M = atoi(argv[1]);
    first_wave_params.N = atoi(argv[2]);

    // Read remaining parameters from file
    if (read_params("damped_wave/parameters/first_wave.txt", &first_wave_params) != 0) {
        printf("Error while reading the param file!\n");
        return 1;
    }

    printf("M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d\n", first_wave_params.M,
           first_wave_params.N, first_wave_params.dx, first_wave_params.dt, first_wave_params.c,
           first_wave_params.gamma, first_wave_params.i0, first_wave_params.j0,
           first_wave_params.intensity);

    simulate_wave(first_wave_params.gamma, first_wave_params.c, first_wave_params.dt,
                  first_wave_params.dx, first_wave_params.M, first_wave_params.N,
                  first_wave_params.i0, first_wave_params.j0, first_wave_params.intensity);

    return 0;
}
