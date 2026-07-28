#ifndef SIM_COORDINATOR_MPI_H
#define SIM_COORDINATOR_MPI_H

#include "damped_wave/general_functions/misc.h"
#include "damped_wave/general_functions/params.h"

void coordinated_simulation(int rank, Params *wave_params, double *old, double *current,
                            double *new, int *color_value);

#endif