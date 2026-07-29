#ifndef COLOR_CUDA_H
#define COLOR_CUDA_H

/* size_t is used in the prototypes below, so the header must be
 * self-contained and not rely on the includer having pulled it in already. */
#include <stddef.h>

/* GPU colouring module (part 3).
 *
 * The wave simulation stays on the CPU exactly as in part 1: it produces, for
 * every frame, a greyscale matrix of ints in [0,255]. This module only maps
 * each grey value to an RGB triplet, which is what the assignment asks for:
 * "the original matrix is used to continue the simulation".
 *
 * The module does GPU work only. Writing the .ppm file is left to
 * write_snapshot_color_serial() in general_functions/misc.c, so that all file
 * I/O stays in one place, mirroring what the OpenMP part does with the PGM.
 *
 * The .cu file is compiled by nvcc (which treats it as C++), while the callers
 * are compiled by gcc as C. extern "C" disables C++ name mangling so the two
 * object files link together.
 *
 * Typical use inside simulate_wave_cuda():
 *     color_cuda_init(M);                              // once, before the loop
 *     ... for every frame ...
 *     color_cuda_colorize(color_value, rgb_frame, M);  // grey -> RGB, on GPU
 *     write_snapshot_color_serial(rgb_frame, M, iter, folder);
 *     ...
 *     color_cuda_report_timing();                      // once, for the report
 *     color_cuda_free();                               // once, after the loop
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Allocates the device buffers and creates the CUDA events used for timing.
 * Called once: cudaMalloc synchronises with the driver and is far too
 * expensive to be repeated for every one of the N frames. */
void color_cuda_init(int M);

/* Colours one frame on the GPU.
 *   gray_host : input,  M*M ints in [0,255]        (host memory)
 *   rgb_host  : output, 3*M*M bytes, interleaved   (host memory, caller-owned)
 * Performs: copy H2D, kernel launch (one thread per pixel), copy D2H. */
void color_cuda_colorize(const int *gray_host, unsigned char *rgb_host, int M);

/* Allocates a host buffer that will take part in transfers to/from the device.
 *
 * Two implementations, selected at compile time with -DUSE_PINNED_MEMORY:
 *
 *   default        malloc, i.e. ordinary pageable memory. The OS may swap those
 *                  pages out, so the driver cannot DMA from them directly: it
 *                  first copies the data into an internal pinned staging
 *                  buffer, and only then starts the transfer. Every byte is
 *                  therefore copied twice.
 *
 *   USE_PINNED_...  cudaHostAlloc, i.e. page-locked (pinned) memory. The pages
 *                  are guaranteed to stay resident, so the DMA engine reads
 *                  them directly and the staging copy disappears.
 *
 * Pinned memory is not free: it cannot be swapped, so allocating too much of it
 * starves the rest of the system. Here it is a handful of MB, allocated once.
 *
 * The two builds share every other line of code, so any difference measured
 * between them is due to the allocation strategy and to nothing else. */
void *color_cuda_alloc_host(size_t bytes);

/* Releases a buffer obtained from color_cuda_alloc_host. Host and device
 * allocators must not be mixed: memory from cudaHostAlloc has to go back
 * through cudaFreeHost, not free(). */
void color_cuda_free_host(void *ptr);

/* "pinned" or "pageable", for logging and for the CSV. */
const char *color_cuda_memory_mode(void);

/* Prints the accumulated timings (H2D transfer, kernel, D2H transfer) and
 * appends them to cuda_timing_results.csv, ready for the report plots. */
void color_cuda_report_timing(int M, int N);

/* Frees device buffers and destroys the CUDA events. */
void color_cuda_free(void);

#ifdef __cplusplus
}
#endif

#endif /* COLOR_CUDA_H */
