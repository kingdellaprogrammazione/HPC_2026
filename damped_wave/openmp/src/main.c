#include "damped_wave/general_functions/misc.h"
#include "damped_wave/openmp/src/openmp_helper.h"

#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// write the matrix inside the file? this can be done only by one thread due to the problems of I/O
// collective access. moreover no sense since the normal hdd, ssd. Investigate on the possible
// benefits if the HPC implements a parallel filesystem (so no normal hdd). Evaluate slowness of
// this write operation vs calculation time.

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

    printf("M=%d N=%d dx=%g dt=%g c=%g gamma=%g i0=%d j0=%d intensity=%d start_frame=%d\n",
           first_wave_params.M, first_wave_params.N, first_wave_params.dx, first_wave_params.dt,
           first_wave_params.c, first_wave_params.gamma, first_wave_params.i0, first_wave_params.j0,
           first_wave_params.intensity, first_wave_params.frame_start);

    /* ═══════════════ TIMING START ═══════════════ */
    int num_threads = omp_get_max_threads();
    printf("=== Running simulation with %d OpenMP threads ===\n", num_threads);

    double t_start = omp_get_wtime();
    /* ═════════════════════════════════════════════ */           
    simulate_wave(first_wave_params.gamma, first_wave_params.c, first_wave_params.dt,
                  first_wave_params.dx, first_wave_params.M, first_wave_params.N,
                  first_wave_params.i0, first_wave_params.j0, first_wave_params.intensity,
                  "damped_wave/openmp/sim");

    /* ═══════════════ TIMING END ═══════════════ */
    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

    printf("=== Total time: %.4f seconds ===\n", elapsed);

    FILE *csv = fopen("timing_results.csv", "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0) {
            fprintf(csv, "threads,M,N,time_seconds\n");
        }
        fprintf(csv, "%d,%d,%d,%.6f\n",
                num_threads, first_wave_params.M, first_wave_params.N, elapsed);
        fclose(csv);
    } else {
        perror("fopen timing_results.csv");
    }
    /* ═══════════════════════════════════════════ */
              

    return 0;
}
