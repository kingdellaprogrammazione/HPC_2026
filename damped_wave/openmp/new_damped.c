#include <stdlib.h>
#include <stdio.h>
#include <string.h>   /* ADDED: for memcpy */
#include <omp.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════════════
 * FIX [1] — write_snapshot_serial
 *
 * ORIGINAL BUG: color_value already holds values in [0,255], but
 * the function treated them as floats in [0,1] → clamped to 1.0 →
 * multiplied by 255 again → almost entirely white image.
 *
 * CHANGE: 3 lines inside the for loop.
 * ═══════════════════════════════════════════════════════════════ */
void write_snapshot_serial(int *u, int M, int step)
{
    char filename[256];
    snprintf(filename, sizeof(filename), "sim/frame_%05d.pgm", step);

    FILE *f = fopen(filename, "wb");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "P5\n%d %d\n255\n", M, M);

    size_t n = (size_t)M * M;
    unsigned char *buffer = malloc(n);
    if (!buffer) { perror("malloc"); fclose(f); return; }

    for (size_t i = 0; i < n; ++i)
    {
        /* --- BEFORE (wrong) ---
         * double val = u[i];
         * if (val < 0.0) val = 0.0;
         * if (val > 1.0) val = 1.0;               // clamped anything > 1 to 1
         * buffer[i] = (unsigned char)(val * 255.0); // scaled by 255 again!
         *
         * --- AFTER (correct): values are already in [0,255] --- */
        int val = u[i];
        if (val < 0)   val = 0;
        if (val > 255) val = 255;
        buffer[i] = (unsigned char)val;
    }

    fwrite(buffer, sizeof(unsigned char), n, f);
    free(buffer);
    fclose(f);
}

/* ═══════════════════════════════════════════════════════════════
 * FIX [2] — NEW FUNCTION wave_update
 *
 * INSERT HERE, right before simulate_wave().
 *
 * Computes u^{n+1}_{i,j} using the correct discrete formula for
 * the damped wave equation (central finite differences, O(Δt²+Δx²)):
 *
 *   u^{n+1}_{i,j} = 1/(1+γΔt/2) · [
 *       2 · u^n_{i,j}
 *     + (γΔt/2 − 1) · u^{n-1}_{i,j}
 *     + c²Δt² · ∇²u^n_{i,j}
 *   ]
 *
 * with: ∇²u = (u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]−4u[i,j]) / Δx²
 *
 * Receives factor/damp/c2dt2/inv_dx2 already precomputed outside
 * the time loop.
 * ═══════════════════════════════════════════════════════════════ */
static double wave_update(const double *prev, const double *curr,
                           int i, int j, int M,
                           double factor, double damp,
                           double c2dt2,  double inv_dx2)
{
    /* Discrete Laplacian — 5-point stencil */
    double lap = (
        curr[(i+1)*M + j] +
        curr[(i-1)*M + j] +
        curr[i*M + (j+1)] +
        curr[i*M + (j-1)] -
        4.0 * curr[i*M + j]
    ) * inv_dx2;

    /* Damped wave equation update */
    return factor * (
        2.0          * curr[i*M + j] +
        (damp - 1.0) * prev[i*M + j] +
        c2dt2        * lap
    );
}

void simulate_wave(double gamma, double c, double dt, double dx,
                   int M, int N, int i0, int j0, int intensity)
{
    /* ═══ FIX [3] — ALLOCATE THREE ARRAYS, calloc instead of malloc ═══
     *
     * ORIGINAL BUG: only old and new → u^{n-1} was missing.
     * The wave equation requires three time levels.
     *
     * calloc (instead of malloc) zero-initialises everything →
     * boundary cells are already 0 (Dirichlet condition) with no
     * extra memset needed.
     *
     * BEFORE:
     *   double *old = (double *)malloc(M * M * sizeof(double));
     *   double *new = (double *)malloc(M * M * sizeof(double));
     *
     * AFTER: ↓ */
    double *prev        = calloc(M * M, sizeof(double));  /* u^{n-1} */
    double *old         = calloc(M * M, sizeof(double));  /* u^{n}   */
    double *new         = calloc(M * M, sizeof(double));  /* u^{n+1} */
    int    *color_value = malloc(M * M * sizeof(int));

    if (!prev || !old || !new || !color_value) {
        perror("calloc/malloc");
        free(prev); free(old); free(new); free(color_value);
        return;
    }

    /* Initialisation — identical to original, no changes here */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (i != i0 || j != j0) {
                old[i * M + j]         = 0.0;
                color_value[i * M + j] = 0;
            } else {
                old[i * M + j]         = (double)intensity;
                color_value[i * M + j] = intensity;
            }
        }
    }

    /* ═══ FIX [4] — INITIALISE prev AS A COPY OF old ═══
     *
     * This enforces zero initial velocity (du/dt = 0 at t=0).
     * Without this, prev holds garbage values at the first step
     * and the simulation is immediately corrupted.
     *
     * ADD THIS right after the initialisation loop: */
    memcpy(prev, old, M * M * sizeof(double));

    write_snapshot_serial(color_value, M, 0);

    /* ═══ FIX [5] — PRECOMPUTED CONSTANTS before the time loop ═══
     *
     * Avoids redundant divisions and multiplications per cell.
     * ADD THESE right before the for (int iter = ...) loop.
     * They were completely absent from the original code. */
    double damp    = gamma * dt * 0.5;          /* γΔt/2         */
    double factor  = 1.0 / (1.0 + damp);        /* 1/(1+γΔt/2)   */
    double c2dt2   = c * c * dt * dt;            /* c²Δt²         */
    double inv_dx2 = 1.0 / (dx * dx);            /* 1/Δx²         */

    /* CFL stability check: c*dt/dx must be < 1/√2 ≈ 0.707 */
    double cfl = c * dt / dx;
    if (cfl >= 0.7071)
        fprintf(stderr,
            "WARNING: CFL = %.4f >= 0.707 → simulation will be unstable!\n", cfl);

    for (int iter = 1; iter < N; iter++)
    {
        /* ═══ FIX [6] — REPLACE the simple average with wave_update ═══
         *
         * ORIGINAL BUG (completely wrong — not the wave equation):
         *   new[i*M+j] = 0.25*(old[(i-1)*M+j] + old[(i+1)*M+j]
         *                    + old[i*M+(j-1)] + old[i*M+(j+1)]);
         *
         * That formula:
         *   - ignores γ, c, dt, dx entirely
         *   - ignores the previous time step u^{n-1}
         *   - is the discrete Laplace equation (steady-state), not the wave equation
         *
         * AFTER (correct): ↓ */
#pragma omp parallel for schedule(static)
        for (int i = 1; i < M - 1; i++)
        {
            for (int j = 1; j < M - 1; j++)
            {
                new[i * M + j] = wave_update(prev, old, i, j, M,
                                              factor, damp, c2dt2, inv_dx2);
            }
        }
        /* Boundary cells: Dirichlet u=0 — never written → stay 0 (calloc) */

        /* ═══ Normalisation — your original idea kept unchanged ═══
         * Using ±intensity as a fixed range avoids scanning for min/max
         * every frame. Valid because the wave is damped: amplitude only
         * decreases, so the initial impulse is always the global maximum. */
        double min_val   = -(double)intensity;
        double max_val   =  (double)intensity;
        double range     = 2.0 * max_val;
        double inv_range = (range > 0.0) ? (255.0 / range) : 0.0;

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)M * M; ++i)
        {
            int val = (int)((new[i] - min_val) * inv_range);
            color_value[i] = (val < 0) ? 0 : (val > 255 ? 255 : val);
        }

        write_snapshot_serial(color_value, M, iter);

        /* ═══ FIX [7] — 3-POINTER ROTATION instead of 2 ═══
         *
         * BEFORE (wrong for 2 arrays only):
         *   double *temp = old; old = new; new = temp;
         *
         * AFTER: cyclic rotation prev←old←new←(prev's memory)
         *   After rotation:
         *     prev = old's previous content → u^{n-1} for next step
         *     old  = new's previous content → u^{n}   for next step
         *     new  = prev's old memory      → free buffer for u^{n+1}
         *   Boundary cells of new stay 0: they come from prev, which
         *   was zero-initialised by calloc at the start. */
        double *temp = prev;
        prev = old;
        old  = new;
        new  = temp;
    }

    /* FIX: added free(prev) and free(color_value) which were missing */
    free(prev);
    free(old);
    free(new);
    free(color_value);
}

int main(int argc, char *argv[])
{
    if (argc != 10)
    {
        fprintf(stderr, "Usage: %s M N dx dt c gamma i0 j0 intensity\n", argv[0]);
        fprintf(stderr, "Example: %s 200 200 0.01 0.00001 343 35.5 100 100 54\n", argv[0]);
        return 1;
    }

    int M         = atoi(argv[1]);
    int N         = atoi(argv[2]);
    double dx     = atof(argv[3]);
    double dt     = atof(argv[4]);
    double c      = atof(argv[5]);
    double gamma  = atof(argv[6]);
    int i0        = atoi(argv[7]);
    int j0        = atoi(argv[8]);
    int intensity = atoi(argv[9]);

    simulate_wave(gamma, c, dt, dx, M, N, i0, j0, intensity);
    return 0;
}