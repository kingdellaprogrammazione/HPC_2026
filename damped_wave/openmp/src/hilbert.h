#ifndef HILBERT_H
#define HILBERT_H

#include <stdint.h>

double wave_update_9_pts_hilbert(const double *prev, const double *curr, const uint32_t *H, int i,
                                 int j, int M, double factor, double damp, double c2dt2,
                                 double inv_dx2);

void simulate_wave_hilbert(double gamma, double c, double dt, double dx, int M, int N, int i0,
                           int j0, int intensity, char *relative_path_sim_folder);

/*
 * Hilbert curve encode
 *
 * bits: order of the Hilbert curve
 *       grid size = 2^bits x 2^bits
 *
 * x,y:  integer coordinates
 *
 * returns:
 *       Hilbert distance d
 */
uint32_t hilbert_encode(int bits, uint32_t x, uint32_t y);

/*
 * Hilbert curve decode
 *
 * bits: order of the Hilbert curve
 *
 * d: Hilbert distance
 *
 * returns x,y coordinates through pointers
 */
void hilbert_decode(int bits, uint32_t d, uint32_t *x, uint32_t *y);

#endif
