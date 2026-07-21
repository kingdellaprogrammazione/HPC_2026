#ifndef MISC_H
#define MISC_H

#include "damped_wave/general_functions/params.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int eval_grid_expression(const char *expr, Params *p);

int read_params(const char *filename, Params *p);

// TODO remember to check if filename chars are sufficient, to explain size_t and unsigned_char use.
void write_snapshot_serial(int *frame_array, int side_dimension, int step, char *sim_folder_path);

// This function evaluates for a given (i,j) cartesian position the evolved wave amplitude
double wave_update(const double *prev, const double *curr, int i, int j, int M, double factor,
                   double damp, double c2dt2, double inv_dx2);

double wave_update_9_pts(const double *prev, const double *curr, int i, int j, int M, double factor,
                         double damp, double c2dt2, double inv_dx2);

double gaussian_function(int i, int j, int mean_i, int mean_j, double one_over_sigma,
                         double max_intensity);

double initialize_gaussian(int i, int j, int half_pulse_side, int i_0, int j_0,
                           double max_intensity);

double initialize_delta(int i, int j, int i_0, int j_0, double intensity);

int rescale_discretize_intensity(double actual_intensity, int *min_intensity, double *inv_range);

#endif MISC_H