#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include "damped_wave/general_functions/misc.h"
#include "damped_wave/general_functions/params.h"

#include <float.h>
#include <stdlib.h>

void independent_simulation(int rank, Params *wave_params, Params *wave_params_2, double *old,
                            double *current, double *new, int *color_value) {

    // Read remaining parameters from file
    if (read_params("damped_wave/parameters/first_wave.txt", wave_params) != 0) {
        printf("Error while reading the first param file!\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    double damp = wave_params->gamma * wave_params->dt * 0.5; /* γΔt/2         */
    double factor = 1.0 / (1.0 + damp);                       /* 1/(1+γΔt/2)   */
    double c2dt2 = wave_params->c * wave_params->c * wave_params->dt * wave_params->dt; /* c²Δt² */
    double inv_dx2 = 1.0 / (wave_params->dx * wave_params->dx);

    int gaussian_pulse_dimension = (int)(wave_params->M * 0.1);
    int half_side = (int)(gaussian_pulse_dimension * 0.5);

    // This node does only the normal simulation
    if (rank == 0) {

#pragma omp parallel
        {
#pragma omp master
            {
                printf("Rank %d: %d OpenMP threads\n", rank, omp_get_num_threads());
            }
        }

        printf("First wave params: M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d "
               "start_frame=%d\n",
               wave_params->M, wave_params->N, wave_params->dx, wave_params->dt, wave_params->c,
               wave_params->gamma, wave_params->i0, wave_params->j0, wave_params->intensity,
               wave_params->frame_start);

        // ------------------------------------------------------------------------------------------------------
        // initialize 1st frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        old = (double *)malloc(wave_params->M * wave_params->M * sizeof(double));
        current = (double *)malloc(wave_params->M * wave_params->M * sizeof(double));
        new = (double *)calloc(wave_params->M * wave_params->M, sizeof(double));
        // write_snapshot_serial expects int* (it copies into an unsigned char
        // buffer internally before writing to the PGM file), so keep this as
        // int even though only the low byte is actually used.
        color_value = (int *)malloc(wave_params->M * wave_params->M * sizeof(int));

        // Initialize the matrix in parallel, using a simple domain partitioning, like a rectangle
        // row major one. Here no unbalance registered, since each single square needs to be
        // initialised in the exact same way.

        // Now define a region where the initial pulse will exist.
        printf("[Rank %d] Creating initial pulse\n", rank);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params->M; i++) {
            for (int j = 0; j < wave_params->M; j++) {

                // Evaluate only once
                double start_impulse_i_j = initialize_gaussian(
                    i, j, half_side, wave_params->i0, wave_params->j0, wave_params->intensity);

                old[i * wave_params->M + j] = start_impulse_i_j;
                current[i * wave_params->M + j] = start_impulse_i_j;
            }
        }

        int min_val = -abs(wave_params->intensity);
        int max_val = abs(wave_params->intensity);
        int range = 2 * max_val;
        double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params->M; i++) {
            for (int j = 0; j < wave_params->M; j++) {
                color_value[i * wave_params->M + j] = rescale_discretize_intensity(
                    current[i * wave_params->M + j], &min_val, &inv_range);
            }
        }

        // Do a save of old on the file using the function
        //  remember path are defined fromn the root Makefile, don't use a last /
        write_snapshot_serial(color_value, wave_params->M, 0, "damped_wave/MPI/sim1");
        // TEST
        // DISABLING
        printf("[Rank %d] Wrote first frame\n", rank);

        // ------------------------------------------------------------------------------------------------------
        // Evaluate next frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        printf("[Rank %d] Starting frame loop\n", rank);
        // Here the iteration steps, each one produces a frame
        for (int iter = 1; iter < wave_params->N; iter++) {
#pragma omp parallel for schedule(static)
            for (int i = 1; i < wave_params->M - 1; i++) {
                for (int j = 1; j < wave_params->M - 1; j++) {
                    // Leapfrog update with isotropic 9-point Laplacian, implemented in misc.c
                    new[i * wave_params->M + j] = wave_update_9_pts(
                        old, current, i, j, wave_params->M, factor, damp, c2dt2, inv_dx2);
                }
            }

            // ------------------------------------------------------------------------------------------------------
            // Normalize next frame
            // ------------------------------------------------------------------------------------------------------

            // Rescale to unsigned char, clamping to [0,255] to avoid silent
            // wrap-around if the wave amplitude ever exceeds the assumed range.
#pragma omp parallel for schedule(static)
            for (int i = 0; i < wave_params->M * wave_params->M; ++i) {
                color_value[i] = rescale_discretize_intensity(new[i], &min_val, &inv_range);
            }

            write_snapshot_serial(color_value, wave_params->M, iter, "damped_wave/MPI/sim1");
            // TEST DISABLING

            // Exchange pointers. We need this since if I only point old to new then when i will
            // write new it will overwrite.
            double *temp = old;
            old = current;
            current = new;
            new = temp;
        }
        printf("[Rank %d] Ended frame loop\n", rank);

        // After the rotation above, all three original buffers are still
        // reachable through old/current/new (just relabeled): free all of them.
        free(old);
        free(current);
        free(new);
        free(color_value);
    }

    // This node does the interference with two simultaneous pulses
    if (rank == 1) {

#pragma omp parallel
        {
#pragma omp master
            {
                printf("Rank %d: %d OpenMP threads\n", rank, omp_get_num_threads());
            }
        }

        // Read remaining parameters from file
        if (read_params("damped_wave/parameters/second_wave.txt", wave_params_2) != 0) {
            printf("Error while reading the second param file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Second wave params: M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d "
               "start_frame=%d\n",
               wave_params_2->M, wave_params_2->N, wave_params_2->dx, wave_params_2->dt,
               wave_params_2->c, wave_params_2->gamma, wave_params_2->i0, wave_params_2->j0,
               wave_params_2->intensity, wave_params_2->frame_start);

        old = (double *)malloc(wave_params->M * wave_params->M * sizeof(double));

        current = (double *)malloc(wave_params->M * wave_params->M * sizeof(double));

        new = (double *)calloc(wave_params->M * wave_params->M, sizeof(double));

        // write_snapshot_serial expects int* (it copies into an unsigned char
        // buffer internally before writing to the PGM file), so keep this as
        // int even though only the low byte is actually used.
        color_value = (int *)malloc(wave_params->M * wave_params->M * sizeof(int));

        // ------------------------------------------------------------------------------------------------------
        // initialize 1st frame without normalizing
        // ------------------------------------------------------------------------------------------------------
        printf("[Rank %d] Creating initial pulse\n", rank);

        // Evaluate once and for all the simulation the max intensity reachable in the worst
        // case of interfering waves. Since the simulation is damped, this is the max for
        // all instants
        int min_val = -abs(wave_params->intensity) - abs(wave_params_2->intensity);
        int max_val = abs(wave_params->intensity) + abs(wave_params_2->intensity);
        int range = 2 * max_val;
        double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int i = 0; i < wave_params->M; i++) {
                for (int j = 0; j < wave_params->M; j++) {

                    // Evaluate only once
                    double start_impulse_i_j = initialize_gaussian(
                        i, j, half_side, wave_params->i0, wave_params->j0, wave_params->intensity);

                    double start_impulse_i_j_2 =
                        initialize_gaussian(i, j, half_side, wave_params_2->i0, wave_params_2->j0,
                                            wave_params_2->intensity);

                    // they are sufficiently apart to be summed

                    double total_impulse_i_j = start_impulse_i_j + start_impulse_i_j_2;

                    old[i * wave_params->M + j] = total_impulse_i_j;
                    current[i * wave_params->M + j] = total_impulse_i_j;
                }
            }

#pragma omp for schedule(static)
            for (int i = 0; i < wave_params->M; i++) {
                for (int j = 0; j < wave_params->M; j++) {
                    color_value[i * wave_params->M + j] = rescale_discretize_intensity(
                        current[i * wave_params->M + j], &min_val, &inv_range);
                }
            }
        }

        // Do a save of old on the file using the function
        // remember path are defined fromn the root Makefile
        write_snapshot_serial(color_value, wave_params->M, 0, "damped_wave/MPI/sim2");
        // TEST
        // DISABLING

        // ------------------------------------------------------------------------------------------------------
        // Evaluate next frames without normalizing
        // ------------------------------------------------------------------------------------------------------

        // Here the iteration steps, each one produces a frame
        for (int iter = 1; iter < wave_params->N; iter++) {
#pragma omp parallel
            {
#pragma omp for schedule(static)
                for (int i = 1; i < wave_params->M - 1; i++) {
                    for (int j = 1; j < wave_params->M - 1; j++) {
                        // Leapfrog update with isotropic 9-point Laplacian, implemented in misc.c
                        new[i * wave_params->M + j] = wave_update_9_pts(
                            old, current, i, j, wave_params->M, factor, damp, c2dt2, inv_dx2);
                    }
                }

                // Scale using the calculated parameters at the beginning
#pragma omp for schedule(static)
                for (int i = 0; i < wave_params->M * wave_params->M; ++i) {
                    color_value[i] = rescale_discretize_intensity(current[i], &min_val, &inv_range);
                }

                // Exchange pointers. We need this since if I only point old to new then when i will
                // write new it will overwrite.
#pragma omp single
                {
                    write_snapshot_serial(color_value, wave_params->M, iter,
                                          "damped_wave/MPI/sim2");
                    // TEST DISABLING
                    double *temp = old;
                    old = current;
                    current = new;
                    new = temp;
                }
            }
        }

        // After the rotation above, all three original buffers are still
        // reachable through old/current/new (just relabeled): free all of them.
        free(old);
        free(current);
        free(new);
        free(color_value);
    }

    // This sums the delayed wave
    if (rank == 2) {

#pragma omp parallel
        {
#pragma omp master
            {
                printf("Rank %d: %d OpenMP threads\n", rank, omp_get_num_threads());
            }
        }

        // Read remaining parameters from file
        if (read_params("damped_wave/parameters/third_wave.txt", wave_params_2) != 0) {
            printf("Error while reading the third param file!\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        printf("Third wave params: M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d "
               "start_frame=%d\n",
               wave_params_2->M, wave_params_2->N, wave_params_2->dx, wave_params_2->dt,
               wave_params_2->c, wave_params_2->gamma, wave_params_2->i0, wave_params_2->j0,
               wave_params_2->intensity, wave_params_2->frame_start);

        int starting_frame = wave_params_2->frame_start;

        // ------------------------------------------------------------------------------------------------------
        // initialize 1st frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        old = (double *)malloc(wave_params->M * wave_params->M * sizeof(double));
        current = (double *)malloc(wave_params->M * wave_params->M * sizeof(double));

        // use calloc so elements are set to 0
        new = calloc(wave_params->M * wave_params->M, sizeof(double));

        // write_snapshot_serial expects int* (it copies into an unsigned char
        // buffer internally before writing to the PGM file), so keep this as
        // int even though only the low byte is actually used.
        color_value = (int *)malloc(wave_params->M * wave_params->M * sizeof(int));

        // ------------------------------------------------------------------------------------------------------
        // Sum frames the colors (1st frame)
        // ------------------------------------------------------------------------------------------------------
        // no current = current_rank_0 since it leads to a double free and a segfault
        // ------------------------------------------------------------------------------------------------------
        // Now normalize the colors (1st frame)
        // ------------------------------------------------------------------------------------------------------

        int min_val = -abs(wave_params->intensity) - abs(wave_params_2->intensity);
        int max_val = abs(wave_params->intensity) + abs(wave_params_2->intensity);
        int range = 2 * max_val;
        double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

#pragma omp for schedule(static)
        for (int i = 0; i < wave_params->M; i++) {
            for (int j = 0; j < wave_params->M; j++) {

                // Evaluate only once
                double start_impulse_i_j = initialize_gaussian(
                    i, j, half_side, wave_params->i0, wave_params->j0, wave_params->intensity);

                old[i * wave_params->M + j] = start_impulse_i_j;
                current[i * wave_params->M + j] = start_impulse_i_j;
            }
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params->M; i++) {
            for (int j = 0; j < wave_params->M; j++) {
                color_value[i * wave_params->M + j] = rescale_discretize_intensity(
                    current[i * wave_params->M + j], &min_val, &inv_range);
            }
        }

        // Do a save of old on the file using the function
        //  remember path are defined fromn the root Makefile
        write_snapshot_serial(color_value, wave_params->M, 0, "damped_wave/MPI/sim3");
        // TEST
        // DISABLING

        // ------------------------------------------------------------------------------------------------------
        // Evaluate next frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        // Prepare the pointers
        printf("[Rank %d] Creating initial pulse\n", rank);

        // Here the iteration steps, each one produces a frame

        for (int iter = 1; iter < wave_params->N; iter++) {
#pragma omp parallel
            {
                if (iter == starting_frame) {
#pragma omp for schedule(static)
                    for (int i = 1; i < wave_params->M - 1; i++) {
                        for (int j = 1; j < wave_params->M - 1; j++) {
                            old[i * wave_params->M + j] +=
                                initialize_gaussian(i, j, half_side, wave_params_2->i0,
                                                    wave_params_2->j0, wave_params_2->intensity);
                            current[i * wave_params->M + j] +=
                                initialize_gaussian(i, j, half_side, wave_params_2->i0,
                                                    wave_params_2->j0, wave_params_2->intensity);
                        }
                    }
                }
#pragma omp for schedule(static)
                for (int i = 1; i < wave_params->M - 1; i++) {
                    for (int j = 1; j < wave_params->M - 1; j++) {
                        // Leapfrog update with isotropic 9-point Laplacian, implemented in
                        // misc.c

                        new[i * wave_params->M + j] = wave_update_9_pts(
                            old, current, i, j, wave_params->M, factor, damp, c2dt2, inv_dx2);
                    }
                }

                // Rescale to unsigned char, clamping to [0,255] to avoid silent
                // wrap-around if the wave amplitude ever exceeds the assumed range.
#pragma omp for schedule(static)
                for (int i = 0; i < wave_params->M * wave_params->M; ++i) {
                    color_value[i] = rescale_discretize_intensity(current[i], &min_val, &inv_range);
                }

                // then when i will write new it will overwrite.
#pragma omp single
                {
                    write_snapshot_serial(color_value, wave_params->M, iter,
                                          "damped_wave/MPI/sim3");
                    // TEST DISABLING Exchange pointers. We need this since if I only point old to
                    // new

                    double *temp = old;
                    old = current;
                    current = new;
                    new = temp;
                }
            }
        }

        // After the rotation above, all three original buffers are still
        // reachable through old/current/new (just relabeled): free all of them.
        free(old);
        free(current);
        free(new);
        free(color_value);
    }
}