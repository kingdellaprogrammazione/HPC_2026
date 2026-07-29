#include "damped_wave/CUDA/src/cuda_helper.h"
#include "damped_wave/CUDA/src/color_cuda.h"
#include "damped_wave/general_functions/misc.h"

#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Switch between the two Laplacian stencils implemented in misc.c.
 *   5 points  -> cheaper, but anisotropic: a point source spreads slightly
 *                faster along the axes than along the diagonals, so the
 *                wavefronts tend to look diamond-shaped far from the source.
 *   9 points  -> isotropic version, more memory traffic per cell but visibly
 *                rounder fronts, which matters a lot once the frames are
 *                coloured and the shape becomes obvious.
 * The stencil also changes the stability limit, see the CFL check below. */
#define USE_9_POINT_STENCIL 1

#if USE_9_POINT_STENCIL
#define WAVE_UPDATE wave_update_9_pts
/* Largest eigenvalue of the (1,4,-20)/6 stencil is 32/6, so the limit is
 * sqrt(4 / (32/6)) = sqrt(0.75) ~ 0.866. */
#define CFL_LIMIT 0.866
#else
#define WAVE_UPDATE wave_update
/* Largest eigenvalue of the 5-point stencil is 8, so the limit is
 * sqrt(4/8) = 1/sqrt(2) ~ 0.707. */
#define CFL_LIMIT 0.7071
#endif

/* Colour counterpart of write_snapshot_serial() from misc.c.
 *
 * It is kept local to this file, and not added to general_functions/misc.c, so
 * that the shared code used by the OpenMP and MPI parts is left untouched.
 *
 * Same structure as the PGM writer, but PPM P3 stores three ASCII numbers
 * (R G B) per pixel instead of one binary byte, so an M x M image becomes M
 * rows of 3*M values. rgb_frame is interleaved: R0 G0 B0 R1 G1 B1 ... */
static void write_snapshot_color_serial(unsigned char *rgb_frame, int side_dimension, int step,
                                        char *sim_folder_path) {
    /* 32 chars: "frame_00001.ppm" needs 16, but %05d is a minimum width, not a
     * maximum, so a step above 99999 would print more digits and overflow a
     * tighter buffer. */
    char filename[32];
    snprintf(filename, sizeof(filename), "frame_%05d.ppm", step);

    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", sim_folder_path, filename);

    /* "w" and not "wb": unlike the PGM, P3 is a text format. */
    FILE *f = fopen(full_path, "w");
    if (!f) {
        perror("fopen error");
        return;
    }

    /* ASCII PPM is verbose: without a large buffer every fprintf may trigger a
     * syscall. 1 MB of buffering cuts the number of write() calls drastically. */
    setvbuf(f, NULL, _IOFBF, 1 << 20);

    /* Print metadata at file head: P3 marks the ASCII RGB encoding. */
    fprintf(f, "P3\n%d %d\n255\n", side_dimension, side_dimension);

    for (int i = 0; i < side_dimension; i++) {
        for (int j = 0; j < side_dimension; j++) {
            size_t idx = (size_t)i * side_dimension + j;
            fprintf(f, "%d %d %d ", rgb_frame[3 * idx + 0], rgb_frame[3 * idx + 1],
                    rgb_frame[3 * idx + 2]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

/* Benchmark switch.
 *
 * Writing the frames is by far the most expensive stage of the whole program:
 * a P3 file holds 3*M*M numbers in ASCII, so at M=1024 every single frame is
 * about 12 MB of text. That is fine for the actual simulation, but it makes a
 * parameter sweep over several grid sizes unusable, and it measures the disk
 * rather than the code.
 *
 * Setting WAVE_SKIP_IO=1 in the environment keeps the whole computation and all
 * the GPU transfers, and only skips the file writing. The GPU timings are
 * collected exactly as usual, so the benchmark numbers stay valid.
 * Unset (the normal case), everything behaves as before. */
static int skip_io_enabled(void) {
    const char *e = getenv("WAVE_SKIP_IO");
    return (e != NULL && e[0] == '1');
}

void simulate_wave_cuda(double gamma, double c, double dt, double dx, int M, int N, int i0, int j0,
                        int intensity, char *relative_path_sim_folder) {

    const int skip_io = skip_io_enabled();
    if (skip_io)
        printf("(WAVE_SKIP_IO=1: frames are computed and coloured but not written to disk)\n");

    size_t cells = (size_t)M * M;

    /* The three wave buffers never leave the CPU, so plain calloc is right.
     * calloc and not malloc: zeroing the buffers gives us the u = 0 initial
     * state and, more importantly, the homogeneous Dirichlet boundary for free.
     * The update loops only touch the interior (1 .. M-2), so the border cells
     * are never written and keep the value they were allocated with. With
     * malloc they would hold garbage, which the stencil would then read back
     * into the domain on the following step. */
    double *old = (double *)calloc(cells, sizeof(double));
    double *current = (double *)calloc(cells, sizeof(double));
    double *new = (double *)calloc(cells, sizeof(double));

    /* These two buffers do cross the PCIe bus every frame, so they go through
     * color_cuda_alloc_host: pageable memory by default, page-locked (pinned)
     * memory when the module is built with -DUSE_PINNED_MEMORY. This is the
     * only difference between the two builds. */

    /* Greyscale frame: kept as int for consistency with write_snapshot_serial
     * and with the OpenMP version, even though only the low byte is used. */
    int *color_value = (int *)color_cuda_alloc_host(cells * sizeof(int));

    /* Interleaved RGB frame produced by the GPU: R0 G0 B0 R1 G1 B1 ... */
    unsigned char *rgb_frame =
        (unsigned char *)color_cuda_alloc_host(cells * 3 * sizeof(unsigned char));

    if (!old || !current || !new || !color_value || !rgb_frame) {
        perror("calloc/alloc_host");
        free(old);
        free(current);
        free(new);
        color_cuda_free_host(color_value);
        color_cuda_free_host(rgb_frame);
        return;
    }

    /* color_cuda_alloc_host does not zero the memory, unlike calloc. Every cell
     * is written before being read below, but zeroing keeps the state defined. */
    memset(color_value, 0, cells * sizeof(int));

    /* Initial pulse spread over a Gaussian instead of a single cell. A delta on
     * one cell carries spatial frequencies the grid cannot represent, so it
     * disperses almost immediately and the amplitude collapses within a few
     * steps. A finite-width pulse keeps far more energy in the front and gives
     * a much more readable video. initialize_delta() is still available in
     * misc.c if the point source is wanted instead. */
    int gaussian_pulse_dimension = (int)(M * 0.1);
    int half_side = (int)(gaussian_pulse_dimension * 0.5);

    /* The normalisation range only depends on intensity, so it is computed once
     * here rather than N times inside the loop. Using the fixed range
     * [-intensity, +intensity] is legitimate because the wave is damped: its
     * amplitude can never exceed the initial pulse. It also keeps the colour
     * scale consistent across frames, which a per-frame min/max would not: the
     * video would flicker because every frame would use a different scale. */
    int min_val = -abs(intensity);
    int max_val = abs(intensity);
    int range = 2 * max_val;
    double inv_range = (range > 0) ? 255.0 / range : 0.0;

    /* Precomputed constants: none of them depends on i, j or the time step, so
     * computing them inside the loops would repeat M*M*N useless operations.
     * Note inv_dx2: a reciprocal computed once and then multiplied, since a
     * floating point division costs far more than a multiplication. */
    double damp = gamma * dt * 0.5;     /* gamma*dt/2        */
    double factor = 1.0 / (1.0 + damp); /* 1/(1 + gamma*dt/2) */
    double c2dt2 = c * c * dt * dt;     /* c^2 * dt^2         */
    double inv_dx2 = 1.0 / (dx * dx);   /* 1/dx^2             */

    /* Stability check: an explicit scheme diverges if the wave travels further
     * in one time step than the grid can follow. Better to warn now than to
     * discover NaNs after the whole run. */
    double cfl = c * dt / dx;
    if (cfl >= CFL_LIMIT)
        fprintf(stderr, "WARNING: CFL = %.4f >= %.4f, the simulation will be unstable!\n", cfl,
                CFL_LIMIT);

    /* Initialise the GPU buffers once, not once per frame. */
    color_cuda_init(M);

    /* Initial state. old = current enforces zero initial velocity: the update
     * formula needs u^{n-1}, which does not exist at n = 0, and du/dt = 0 is
     * the natural way to close the scheme. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            double start_impulse_i_j = initialize_gaussian(i, j, half_side, i0, j0, intensity);

            old[i * M + j] = start_impulse_i_j;
            current[i * M + j] = start_impulse_i_j;
            color_value[i * M + j] =
                rescale_discretize_intensity(start_impulse_i_j, &min_val, &inv_range);
        }
    }

    /* No initialization of new, it is a waste: calloc already zeroed it. */

    /* Frame 0: literal 0 as the index, iter does not exist yet. */
    color_cuda_colorize(color_value, rgb_frame, M);
    if (!skip_io)
        write_snapshot_color_serial(rgb_frame, M, 0, relative_path_sim_folder);

    /* Time loop. This one is inherently serial: u^{n+1} needs both u^{n} and
     * u^{n-1}, so step n cannot start before step n-1 is complete. It is a
     * property of the physics, not a limitation of the implementation. */
    for (int iter = 1; iter < N; iter++) {

        /* The two spatial loops are the parallel part: within one time step the
         * new cells only read data from the two previous steps, which nobody is
         * writing, so all M*M cells are independent and no synchronisation is
         * needed. This is only true because we never write into the buffer we
         * are reading, hence the three separate arrays.
         * schedule(static): the work per cell is identical, so a static block
         * split is optimal and gives each thread contiguous rows, which is what
         * the cache wants. */
#pragma omp parallel for schedule(static)
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < M - 1; j++) {
                new[i * M + j] = WAVE_UPDATE(old, current, i, j, M, factor, damp, c2dt2, inv_dx2);
            }
        }

        /* Map the signed wave amplitude onto [0,255]. Element-wise and fully
         * independent, so it parallelises trivially as well.
         * rescale_discretize_intensity clamps the result: a value above 255
         * would not fit in a byte and would corrupt the image. */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)cells; ++i) {
            color_value[i] = rescale_discretize_intensity(new[i], &min_val, &inv_range);
        }

        /* Part 3: the GPU turns the greyscale frame into RGB, the CPU writes it
         * out. CPU and GPU alternate frame by frame, they never overlap: a
         * possible improvement would be CUDA streams and asynchronous copies,
         * so that the colouring of frame n runs while the CPU computes n+1. */
        color_cuda_colorize(color_value, rgb_frame, M);
        if (!skip_io)
            write_snapshot_color_serial(rgb_frame, M, iter, relative_path_sim_folder);

        /* Exchange pointers instead of copying the matrices: three assignments
         * against 2*M*M double copies per step. The buffer holding u^{n-1} is
         * no longer needed and is recycled as the next u^{n+1}. */
        double *temp = old;
        old = current;
        current = new;
        new = temp;
    }

    color_cuda_report_timing(M, N);
    color_cuda_free();

    /* After the rotation above, all three original buffers are still reachable
     * through old/current/new (just relabeled): free all of them. */
    free(old);
    free(current);
    free(new);

    /* These two came from color_cuda_alloc_host, so they must go back through
     * the matching deallocator: cudaFreeHost for pinned memory, free otherwise. */
    color_cuda_free_host(color_value);
    color_cuda_free_host(rgb_frame);
}
