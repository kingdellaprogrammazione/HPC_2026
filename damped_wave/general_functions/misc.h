#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO remember to check if filename chars are sufficient, to explain size_t and unsigned_char use.
void write_snapshot_serial(int *frame_array, int side_dimension, int step, char *sim_folder_path);

// This function evaluates for a given (i,j) cartesian position the evolved wave amplitude
double wave_update(const double *prev, const double *curr, int i, int j, int M, double factor,
                   double damp, double c2dt2, double inv_dx2);

double wave_update_9_pts(const double *prev, const double *curr, int i, int j, int M, double factor,
                         double damp, double c2dt2, double inv_dx2);
