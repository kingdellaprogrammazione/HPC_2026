#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include "damped_wave/general_functions/misc.h"
#include "damped_wave/general_functions/params.h"
#include "damped_wave/openmp/src/openmp_helper.h"

#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

    // Read M and N from command line
    if (argc != 3) {
        printf("Usage: %s M N (M grid dimension, N time steps).\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Do not create all and then sum everything at the end, too many useless r/w. Instead only
    // calculate and immediately exchange info and write down to the folder.

    // Define common parameters, all the simulations has the same grid dimension, the same number
    // of time steps, the same velocity and damping coefficient, and the same grid step and time
    // step
    Params wave_params;

    wave_params.M = atoi(argv[1]);
    wave_params.N = atoi(argv[2]);

    // Define the 3 computation array frames pointers
    double *old = NULL;
    double *current = NULL;
    double *new = NULL;

    // Define the 1 output array frame pointers
    int *color_value = NULL;

    // This node does only the normal simulation
    if (rank == 0) {

        // Read remaining parameters from file
        if (read_params_specific("damped_wave/parameters/first_wave.txt", &wave_params) != 0) {
            printf("Error while reading the first param file!\n");
            MPI_Abort(MPI_COMM_WORLD);
        }

        printf("First wave params: M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d "
               "start_frame=%d\n",
               wave_params.M, wave_params.N, wave_params.dx, wave_params.dt, wave_params.c,
               wave_params.gamma, wave_params.i0, wave_params.j0, wave_params.intensity,
               wave_params.frame_start);

        double damp = wave_params.gamma * wave_params.dt * 0.5; /* γΔt/2         */
        double factor = 1.0 / (1.0 + damp);                     /* 1/(1+γΔt/2)   */
        double c2dt2 = wave_params.c * wave_params.c * wave_params.dt * wave_params.dt; /* c²Δt² */
        double inv_dx2 = 1.0 / (wave_params.dx * wave_params.dx);

        int gaussian_pulse_dimension = (int)(wave_params.M * 0.1);
        int half_side = (int)(gaussian_pulse_dimension * 0.5);

        // ------------------------------------------------------------------------------------------------------
        // initialize 1st frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        old = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));
        current = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));
        new = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));
        // write_snapshot_serial expects int* (it copies into an unsigned char
        // buffer internally before writing to the PGM file), so keep this as
        // int even though only the low byte is actually used.
        color_value = (int *)malloc(wave_params.M * wave_params.M * sizeof(int));

        // Initialize the matrix in parallel, using a simple domain partitioning, like a rectangle
        // row major one. Here no unbalance registered, since each single square needs to be
        // initialised in the exact same way.

        // Now define a region where the initial pulse will exist.

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params.M; i++) {
            for (int j = 0; j < wave_params.M; j++) {

                // Evaluate only once
                double start_impulse_i_j = initialize_gaussian(
                    i, j, half_side, wave_params.i0, wave_params.j0, wave_params.intensity);

                old[i * wave_params.M + j] = start_impulse_i_j;
                current[i * wave_params.M + j] = start_impulse_i_j;
            }
        }

        // ------------------------------------------------------------------------------------------------------
        // Share intensity for normalizing
        // Since we have only 2 ranks involved we wouldn't benefit from the tree distribution that
        // broadcast uses
        // ------------------------------------------------------------------------------------------------------

        MPI_Send(wave_params.intensity, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(wave_params.intensity, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);

        // ------------------------------------------------------------------------------------------------------
        // Now share 1st frame with ranks 2 and 3
        // Since we have only 2 ranks involved we wouldn't benefit from the tree distribution that
        // broadcast uses
        // ------------------------------------------------------------------------------------------------------

        MPI_Send(current, wave_params.M * wave_params.M, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Send(current, wave_params.M * wave_params.M, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);

        // ------------------------------------------------------------------------------------------------------
        // Now normalize the colors (1st frame)
        // ------------------------------------------------------------------------------------------------------

        // min/max only depend on intensity: they are constant across the whole
        // simulation, so compute them once outside the time loop instead of
        // redoing it N times. Computed here (before frame 0) so it can also be
        // used to normalize the very first frame consistently.

        int min_val = -abs(wave_params.intensity);
        int max_val = abs(wave_params.intensity);
        int range = 2 * max_val;
        double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params.M; i++) {
            for (int j = 0; j < wave_params.M; j++) {
                color_value[i * wave_params.M + j] =
                    rescale_discretize_intensity(start_impulse_i_j, &min_val, &inv_range);
            }
        }

        // Do a save of old on the file using the function
        //  remember path are defined fromn the root Makefile, don't use a last /
        write_snapshot_serial(color_value, wave_params.M, 0, "damped_wave/MPI/sim1");

        // ------------------------------------------------------------------------------------------------------
        // Evaluate next frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        // Here the iteration steps, each one produces a frame
        for (int iter = 1; iter < wave_params.N; iter++) {
#pragma omp parallel for schedule(static)
            for (int i = 1; i < wave_params.M - 1; i++) {
                for (int j = 1; j < wave_params.M - 1; j++) {
                    // Leapfrog update with isotropic 9-point Laplacian, implemented in misc.c
                    new[i * wave_params.M + j] = wave_update_9_pts(
                        old, current, i, j, wave_params.M, factor, damp, c2dt2, inv_dx2);
                }
            }

            // ------------------------------------------------------------------------------------------------------
            // Share next frame without normalizing
            // Since we have only 2 ranks involved we wouldn't benefit from the tree distribution
            // that broadcast uses
            // ------------------------------------------------------------------------------------------------------

            MPI_Send(new, wave_params.M * wave_params.M, MPI_DOUBLE, 1, iter, MPI_COMM_WORLD);
            MPI_Send(new, wave_params.M * wave_params.M, MPI_DOUBLE, 2, iter, MPI_COMM_WORLD);

            // ------------------------------------------------------------------------------------------------------
            // Normalize next frame
            // ------------------------------------------------------------------------------------------------------

            // Rescale to unsigned char, clamping to [0,255] to avoid silent
            // wrap-around if the wave amplitude ever exceeds the assumed range.
#pragma omp parallel for schedule(static)
            for (int i = 0; i < wave_params.M * wave_params.M; ++i) {
                color_value[i] = rescale_discretize_intensity(new[i], &min_val, &inv_range);
            }

            write_snapshot_serial(color_value, wave_params.M, iter, "damped_wave/MPI/sim1");

            // Exchange pointers. We need this since if I only point old to new then when i will
            // write new it will overwrite.
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

    // This node does the interference with two simultaneous pulses
    if (rank == 1) {

        // Read remaining parameters from file
        if (read_params("damped_wave/parameters/second_wave.txt", &wave_params) != 0) {
            printf("Error while reading the second param file!\n");
            MPI_Abort(MPI_COMM_WORLD);
        }

        printf("Second wave params: M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d "
               "start_frame=%d\n",
               wave_params.M, wave_params.N, wave_params.dx, wave_params.dt, wave_params.c,
               wave_params.gamma, wave_params.i0, wave_params.j0, wave_params.intensity,
               wave_params.frame_start);

        double damp = wave_params.gamma * wave_params.dt * 0.5; /* γΔt/2         */
        double factor = 1.0 / (1.0 + damp);                     /* 1/(1+γΔt/2)   */
        double c2dt2 = wave_params.c * wave_params.c * wave_params.dt * wave_params.dt; /* c²Δt² */
        double inv_dx2 = 1.0 / (wave_params.dx * wave_params.dx);

        int gaussian_pulse_dimension = (int)(wave_params.M * 0.1);
        int half_side = (int)(gaussian_pulse_dimension * 0.5);

        // Initialize this so we can sum this wave with the one coming from rank 0
        double *current_rank_0 = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));

        old = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));
        current = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));
        new = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));
        // write_snapshot_serial expects int* (it copies into an unsigned char
        // buffer internally before writing to the PGM file), so keep this as
        // int even though only the low byte is actually used.
        color_value = (int *)malloc(wave_params.M * wave_params.M * sizeof(int));

        // ------------------------------------------------------------------------------------------------------
        // initialize 1st frame without normalizing
        // ------------------------------------------------------------------------------------------------------

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params.M; i++) {
            for (int j = 0; j < wave_params.M; j++) {

                // Evaluate only once
                double start_impulse_i_j =
                    initialize_gaussian(i, j, half_side, i0, j0, wave_params.intensity);

                old[i * wave_params.M + j] = start_impulse_i_j;
                current[i * wave_params.M + j] = start_impulse_i_j;
            }
        }

        int rank_0_intensity;
        // ------------------------------------------------------------------------------------------------------
        // Receive intensity for normalizing
        // ------------------------------------------------------------------------------------------------------

        MPI_Recv(rank_0_intensity, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // ------------------------------------------------------------------------------------------------------
        // Now receive 1st frame from 0 rank
        // ------------------------------------------------------------------------------------------------------

        MPI_Recv(current_rank_0, wave_params.M * wave_params.M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // ------------------------------------------------------------------------------------------------------
        // Sum frames values(1st frame)
        // ------------------------------------------------------------------------------------------------------

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params.M * wave_params.M; i++) {
            current_rank_0[i] += current[i];
        }

        // ------------------------------------------------------------------------------------------------------
        // Now normalize the colors (1st frame)
        // ------------------------------------------------------------------------------------------------------

        // Evaluate once and for all the simulation the max intensity reachable in the worst case of
        // interfering waves. Since the simulation is damped, this is the max for all instants
        int min_val = -abs(wave_params.intensity) - abs(rank_0_intensity);
        int max_val = abs(wave_params.intensity) + abs(rank_0_intensity);
        int range = 2 * max_val;
        double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params.M; i++) {
            for (int j = 0; j < wave_params.M; j++) {
                color_value[i * wave_params.M + j] = rescale_discretize_intensity(
                    current_rank_0[i * wave_params.M + j], &min_val, &inv_range);
            }
        }

        // Do a save of old on the file using the function
        // remember path are defined fromn the root Makefile
        write_snapshot_serial(color_value, wave_params.M, 0, "damped_wave/MPI/sim2");

        // ------------------------------------------------------------------------------------------------------
        // Evaluate next frames without normalizing
        // ------------------------------------------------------------------------------------------------------

        // Here the iteration steps, each one produces a frame
        for (int iter = 1; iter < wave_params.N; iter++) {
#pragma omp parallel for schedule(static)
            for (int i = 1; i < wave_params.M - 1; i++) {
                for (int j = 1; j < wave_params.M - 1; j++) {
                    // Leapfrog update with isotropic 9-point Laplacian, implemented in misc.c
                    new[i * wave_params.M + j] = wave_update_9_pts(
                        old, current, i, j, wave_params.M, factor, damp, c2dt2, inv_dx2);
                }
            }

            // ------------------------------------------------------------------------------------------------------
            // Share next frame without normalizing
            // ------------------------------------------------------------------------------------------------------

            MPI_Recv(current_rank_0, wave_params.M * wave_params.M, MPI_DOUBLE, 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // ------------------------------------------------------------------------------------------------------
            // Sum next frame
            // ------------------------------------------------------------------------------------------------------

#pragma omp parallel for schedule(static)
            for (int i = 0; i < wave_params.M * wave_params.M; i++) {
                current_rank_0[i] += new[i];
            }

            // Scale using the calculated parameters at the beginning
#pragma omp parallel for schedule(static)
            for (int i = 0; i < wave_params.M * wave_params.M; ++i) {
                color_value[i] = rescale_discretize_intensity(current_rank_0[i * wave_params.M + j],
                                                              &min_val, &inv_range);
            }

            write_snapshot_serial(color_value, wave_params.M, iter, "damped_wave/MPI/sim2");

            // Exchange pointers. We need this since if I only point old to new then when i will
            // write new it will overwrite.

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
        free(current_rank_0);
        free(color_value);
    }

    // This sums the delayed wave
    if (rank == 2) {

        // Read remaining parameters from file
        if (read_params("damped_wave/parameters/third_wave.txt", &wave_params) != 0) {
            printf("Error while reading the third param file!\n");
            MPI_Abort(MPI_COMM_WORLD);
        }

        printf("Third wave params: M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d "
               "start_frame=%d\n",
               wave_params.M, wave_params.N, wave_params.dx, wave_params.dt, wave_params.c,
               wave_params.gamma, wave_params.i0, wave_params.j0, wave_params.intensity,
               wave_params.frame_start);

        double damp = wave_params.gamma * wave_params.dt * 0.5; /* γΔt/2         */
        double factor = 1.0 / (1.0 + damp);                     /* 1/(1+γΔt/2)   */
        double c2dt2 = wave_params.c * wave_params.c * wave_params.dt * wave_params.dt; /* c²Δt² */
        double inv_dx2 = 1.0 / (wave_params.dx * wave_params.dx);

        int gaussian_pulse_dimension = (int)(wave_params.M * 0.1);
        int half_side = (int)(gaussian_pulse_dimension * 0.5);

        double *current_rank_0 = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));

        // Evaluate the starting frame of the simulation

        int starting_frame = wave_params.frame_start;

        // ------------------------------------------------------------------------------------------------------
        // initialize 1st frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        old = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));
        current = (double *)malloc(wave_params.M * wave_params.M * sizeof(double));

        // use calloc so elements are set to 0
        new = (double *)calloc(wave_params.M * wave_params.M * sizeof(double));

        // write_snapshot_serial expects int* (it copies into an unsigned char
        // buffer internally before writing to the PGM file), so keep this as
        // int even though only the low byte is actually used.
        color_value = (int *)malloc(wave_params.M * wave_params.M * sizeof(int));

        int rank_0_intensity;
        // ------------------------------------------------------------------------------------------------------
        // Receive intensity for normalizing
        // ------------------------------------------------------------------------------------------------------

        MPI_Recv(rank_0_intensity, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // ------------------------------------------------------------------------------------------------------
        // Now receive 1st frame from 0 rank
        // ------------------------------------------------------------------------------------------------------

        MPI_Recv(current_rank_0, wave_params.M * wave_params.M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // ------------------------------------------------------------------------------------------------------
        // Sum frames the colors (1st frame)
        // ------------------------------------------------------------------------------------------------------
        current_rank = current_rank_0;
        // ------------------------------------------------------------------------------------------------------
        // Now normalize the colors (1st frame)
        // ------------------------------------------------------------------------------------------------------

        int min_val = -abs(wave_params.intensity) - abs(rank_0_intensity);
        int max_val = abs(wave_params.intensity) + abs(rank_0_intensity);
        int range = 2 * max_val;
        double inv_range = (range > 0.0) ? 255.0 / range : 0.0;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < wave_params.M; i++) {
            for (int j = 0; j < wave_params.M; j++) {
                color_value[i * wave_params.M + j] = rescale_discretize_intensity(
                    current_rank[i * wave_params.M + j], &min_val, &inv_range);
            }
        }

        // Do a save of old on the file using the function
        //  remember path are defined fromn the root Makefile
        write_snapshot_serial(color_value, wave_params.M, 0, "damped_wave/MPI/sim3");

        // ------------------------------------------------------------------------------------------------------
        // Evaluate next frame without normalizing
        // ------------------------------------------------------------------------------------------------------

        // Here the iteration steps, each one produces a frame
        for (int iter = 1; iter < wave_params.N; iter++) {
            if (iter >= starting_frame) {

#pragma omp parallel for schedule(static)
                for (int i = 1; i < wave_params.M - 1; i++) {
                    for (int j = 1; j < wave_params.M - 1; j++) {
                        // Leapfrog update with isotropic 9-point Laplacian, implemented in misc.c
                        new[i * wave_params.M + j] = wave_update_9_pts(
                            old, current, i, j, wave_params.M, factor, damp, c2dt2, inv_dx2);
                    }
                }
            }
            // ------------------------------------------------------------------------------------------------------
            // Share next frame without normalizing
            // ------------------------------------------------------------------------------------------------------

            MPI_Recv(current_rank_0, wave_params.M * wave_params.M, MPI_DOUBLE, 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // ------------------------------------------------------------------------------------------------------
            // Sum next frame
            // ------------------------------------------------------------------------------------------------------

#pragma omp parallel for schedule(static)
            for (int i = 0; i < wave_params.M * wave_params.M; i++) {
                current_rank_0[i] += new[i];
            }

            // Rescale to unsigned char, clamping to [0,255] to avoid silent
            // wrap-around if the wave amplitude ever exceeds the assumed range.
#pragma omp parallel for schedule(static)
            for (int i = 0; i < wave_params.M * wave_params.M; ++i) {
                color_value[i] = rescale_discretize_intensity(current_rank_0[i * wave_params.M + j],
                                                              &min_val, &inv_range);
            }

            write_snapshot_serial(color_value, wave_params.M, iter, "damped_wave/MPI/sim3");
            // Exchange pointers. We need this since if I only point old to new then when i will
            // write new it will overwrite.
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
        free(current_rank_0);
        free(color_value);
    }

    MPI_Finalize();
    return 0;
}
