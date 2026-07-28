#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include "damped_wave/MPI/src/sim_coordinator_mpi.h"
#include "damped_wave/MPI/src/sim_independent_mpi.h"

#include "damped_wave/general_functions/misc.h"
#include "damped_wave/general_functions/params.h"

#include <float.h>
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
    Params wave_params_2;

    wave_params.M = atoi(argv[1]);
    wave_params.N = atoi(argv[2]);

    wave_params_2.M = atoi(argv[1]);
    wave_params_2.N = atoi(argv[2]);

    // Define the 3 computation array frames pointers
    double *old = NULL;
    double *current = NULL;
    double *new = NULL;

    // Define the 1 output array frame pointers
    int *color_value = NULL;

    independent_simulation(rank, &wave_params, &wave_params_2, old, current, new, color_value);

    MPI_Finalize();
    return 0;
}
