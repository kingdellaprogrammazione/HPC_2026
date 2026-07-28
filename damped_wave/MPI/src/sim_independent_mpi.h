#ifndef SIM_INDEPENDENT_MPI_H
#define SIM_INDEPENDENT_MPI_H

#include "damped_wave/general_functions/misc.h"
#include "damped_wave/general_functions/params.h"

void independent_simulation(int rank, Params *wave_params, Params *wave_params_2, double *old,
                            double *current, double *new, int *color_value);

#endif