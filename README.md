# HPC_2026

Here's a quick guide to build and run the damped wave simulation using `make`.

### 1. Clone the repo
Clone the repo with `git clone`.

### 2. Build and run with default parameters
```bash
make
```
This compiles the program and immediately runs the **openMP** step  with:
- Grid size: 30 × 30
- Total time steps: 40
- dx = 0.01, dt = 0.0000073
- Wave speed c = 343 m/s
- Gamma at 23
- Source at (10, 10)
- Intensity at 54

### 3. Override any parameters on the command line
To do this the correct makefile must be run.

Execute
```bash
cd damped_wave/openmp
```
and then
```bash
make M=400 N=400 T=2000 dx=0.005 dt=0.000003
```

### 4. Only compile (no run)
```bash
make wave_sim          # or just: gcc -fopenmp -O3 -o wave_sim main.c -lm
```

### 5. Run again without recompiling
```bash
make run
```

### 6. Clean up binaries
```bash
make clean
```

All parameters are optional – if you omit them, the defaults from the `Makefile` are used. The project will generate snapshots in the `sim` folder (for now needs to be created manually) and should work on any Linux/macOS system with `gcc` and OpenMP.