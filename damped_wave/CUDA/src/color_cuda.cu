#include "damped_wave/CUDA/src/color_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* CUDA calls do not throw: they return an error code that, if ignored, lets the
 * program carry on silently and produce empty images. This macro checks every
 * call and reports file and line of the failure.
 * The do/while(0) wrapper is the standard idiom that lets a multi-statement
 * macro be used as a single statement, even inside a brace-less if. */
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (call);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__, __LINE__,                        \
                    cudaGetErrorString(_err));                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

/* Exponent of the non-linear transfer function, see colorize_pixel().
 * 1.0 gives a plain linear map; lower values give more contrast. */
#define COLOR_ENHANCE 0.30f

/* Thread block side. 16x16 = 256 threads, a multiple of the 32-thread warp. */
#define BLOCK_SIDE 16

/* ─────────────────────────────────────────────────────────────────────────────
 * Colour transfer function, executed on the GPU (__device__).
 *
 * Diverging colour map required by the assignment:
 *     grey ~127 (still water) -> WHITE (255,255,255)
 *     grey  255 (crest, max)  -> RED   (255,  0,  0)
 *     grey    0 (trough, min) -> BLUE  (  0,  0,255)
 * so rising and falling fronts get different colours and calm areas stay light.
 *
 * A linear map alone is not enough: the wave is damped and its amplitude drops
 * by three orders of magnitude during the run, so every frame after the first
 * few would be almost pure white. The signed power law
 *     d' = sign(d) * |d|^COLOR_ENHANCE
 * compresses that dynamic range, amplifying weak fronts while preserving both
 * the sign and the ordering, so maxima and minima remain the brightest points.
 * It is the same idea as gamma correction in imaging.
 * ─────────────────────────────────────────────────────────────────────────── */
__device__ void colorize_pixel(int gray, unsigned char *R, unsigned char *G, unsigned char *B) {
    /* normalised deviation from the rest level, in [-1,+1] */
    float d = ((float)gray - 127.5f) / 127.5f;

    float s = (d >= 0.0f) ? 1.0f : -1.0f;
    float a = __powf(fabsf(d), COLOR_ENHANCE); /* fast-math powf, plenty accurate here */
    if (a > 1.0f)
        a = 1.0f;
    d = s * a;

    if (d >= 0.0f) { /* crest side: white -> red */
        *R = 255;
        *G = (unsigned char)(255.0f * (1.0f - d));
        *B = (unsigned char)(255.0f * (1.0f - d));
    } else { /* trough side: white -> blue */
        float b = -d;
        *R = (unsigned char)(255.0f * (1.0f - b));
        *G = (unsigned char)(255.0f * (1.0f - b));
        *B = 255;
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Kernel: one thread per pixel.
 *
 * This is where the OpenMP and CUDA paradigms differ. In OpenMP we write the
 * two spatial loops and a pragma splits them across the cores. Here the loops
 * disappear: we describe the work of a single pixel and launch it on thousands
 * of threads, each recovering its own (i,j) from its block/thread indices.
 *
 * Colouring is embarrassingly parallel: a pixel depends only on itself, it does
 * not even read its neighbours, so no synchronisation is needed at all.
 * ─────────────────────────────────────────────────────────────────────────── */
__global__ void colorize_kernel(const int *gray, unsigned char *rgb, int M) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; /* column */
    int i = blockIdx.y * blockDim.y + threadIdx.y; /* row    */

    /* The grid is rounded up to a multiple of the block size, so the threads
     * that fall outside the image must do nothing. */
    if (i < M && j < M) {
        int idx = i * M + j;
        unsigned char R, G, B;
        colorize_pixel(gray[idx], &R, &G, &B);
        rgb[3 * idx + 0] = R;
        rgb[3 * idx + 1] = G;
        rgb[3 * idx + 2] = B;
    }
}

/* Device buffers, allocated once and reused for every frame.
 * The d_ prefix marks device memory, as opposed to host memory. */
static int *d_gray = NULL;
static unsigned char *d_rgb = NULL;

/* CUDA events, used instead of a CPU timer: CUDA calls are asynchronous, so a
 * CPU clock would measure the time to enqueue the command, not the time the GPU
 * actually spends on it. Events are markers inserted into the device stream. */
static cudaEvent_t ev_start, ev_stop;

static float total_h2d_ms = 0.0f;    /* grey  host -> device */
static float total_kernel_ms = 0.0f; /* kernel execution     */
static float total_d2h_ms = 0.0f;    /* rgb  device -> host  */
static int frame_count = 0;

/* ─── host buffer allocation: pinned or pageable, chosen at compile time ─────
 * See color_cuda.h for the rationale. Everything else in the file is identical
 * between the two builds. */
extern "C" void *color_cuda_alloc_host(size_t bytes) {
    void *p = NULL;
#ifdef USE_PINNED_MEMORY
    /* Page-locked memory: the DMA engine can read it directly, no staging copy. */
    CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
#else
    p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "malloc of %zu bytes failed\n", bytes);
        exit(EXIT_FAILURE);
    }
#endif
    return p;
}

extern "C" void color_cuda_free_host(void *ptr) {
    if (!ptr)
        return;
#ifdef USE_PINNED_MEMORY
    cudaFreeHost(ptr);
#else
    free(ptr);
#endif
}

extern "C" const char *color_cuda_memory_mode(void) {
#ifdef USE_PINNED_MEMORY
    return "pinned";
#else
    return "pageable";
#endif
}

extern "C" void color_cuda_init(int M) {
    size_t n = (size_t)M * M;

    /* cudaMalloc takes a pointer-to-pointer because it has to modify the
     * pointer we pass in. The RGB buffer is three times larger: one byte per
     * colour channel. */
    CUDA_CHECK(cudaMalloc((void **)&d_gray, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_rgb, n * 3 * sizeof(unsigned char)));

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Warm-up launch, deliberately not timed.
     *
     * The first kernel launch of a process also pays for the creation of the
     * CUDA context and the loading of the module onto the device, which costs
     * tens of milliseconds. Without this dummy launch that cost is charged to
     * the first timed frame and completely distorts the average: measured on
     * this machine the kernel appeared to take 1.22 ms/frame over 30 frames,
     * against 0.12 ms/frame over 200, purely because of how the fixed cost was
     * amortised. Running the kernel once here moves that overhead out of the
     * measurement, so what the report shows is the steady-state cost.
     *
     * cudaDeviceSynchronize waits for the warm-up to actually complete: kernel
     * launches are asynchronous, so without it the initialisation could still
     * be in flight when the first real frame is timed. */
    dim3 wblock(BLOCK_SIDE, BLOCK_SIDE);
    dim3 wgrid((M + BLOCK_SIDE - 1) / BLOCK_SIDE, (M + BLOCK_SIDE - 1) / BLOCK_SIDE);
    colorize_kernel<<<wgrid, wblock>>>(d_gray, d_rgb, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    total_h2d_ms = total_kernel_ms = total_d2h_ms = 0.0f;
    frame_count = 0;
}

extern "C" void color_cuda_colorize(const int *gray_host, unsigned char *rgb_host, int M) {
    size_t n = (size_t)M * M;
    float ms;

    /* 1) grey matrix host -> device.
     * Host and device have separate address spaces: the GPU cannot dereference
     * a CPU pointer, data must cross the PCIe bus explicitly. */
    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(d_gray, gray_host, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    total_h2d_ms += ms;

    /* 2) kernel launch.
     * The integer division (M + BLOCK_SIDE - 1) / BLOCK_SIDE rounds up, so the
     * grid covers the whole image even when M is not a multiple of the block. */
    dim3 block(BLOCK_SIDE, BLOCK_SIDE);
    dim3 grid((M + BLOCK_SIDE - 1) / BLOCK_SIDE, (M + BLOCK_SIDE - 1) / BLOCK_SIDE);

    CUDA_CHECK(cudaEventRecord(ev_start));
    colorize_kernel<<<grid, block>>>(d_gray, d_rgb, M);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError()); /* catches launch configuration errors */
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    total_kernel_ms += ms;

    /* 3) RGB frame device -> host, so the CPU can write it to disk. */
    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(rgb_host, d_rgb, n * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    total_d2h_ms += ms;

    frame_count++;
}

extern "C" void color_cuda_report_timing(int M, int N) {
    if (frame_count == 0)
        return;

    float total = total_h2d_ms + total_kernel_ms + total_d2h_ms;

    printf("=== GPU colouring, %d frames, %s host memory ===\n", frame_count,
           color_cuda_memory_mode());
    printf("  H2D    (host->device) : %9.2f ms  (%.4f ms/frame, %5.1f%%)\n", total_h2d_ms,
           total_h2d_ms / frame_count, 100.0f * total_h2d_ms / total);
    printf("  kernel                : %9.2f ms  (%.4f ms/frame, %5.1f%%)\n", total_kernel_ms,
           total_kernel_ms / frame_count, 100.0f * total_kernel_ms / total);
    printf("  D2H    (device->host) : %9.2f ms  (%.4f ms/frame, %5.1f%%)\n", total_d2h_ms,
           total_d2h_ms / frame_count, 100.0f * total_d2h_ms / total);
    printf("  total GPU side        : %9.2f ms\n", total);
    printf("  (file writing is disk I/O and is deliberately not counted here)\n");

    /* Same CSV logic as the OpenMP main: one row per run, ready to be plotted.
     * The mode column tells the pinned and pageable runs apart. */
    FILE *csv = fopen("cuda_timing_results.csv", "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0)
            fprintf(csv, "mode,M,N,frames,h2d_ms,kernel_ms,d2h_ms,total_gpu_ms\n");
        fprintf(csv, "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n", color_cuda_memory_mode(), M, N,
                frame_count, total_h2d_ms, total_kernel_ms, total_d2h_ms, total);
        fclose(csv);
    } else {
        perror("fopen cuda_timing_results.csv");
    }
}

extern "C" void color_cuda_free(void) {
    /* Memory obtained from cudaMalloc must be released with cudaFree: mixing
     * the host and device allocators is undefined behaviour. */
    if (d_gray)
        cudaFree(d_gray);
    if (d_rgb)
        cudaFree(d_rgb);
    d_gray = NULL;
    d_rgb = NULL;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}
