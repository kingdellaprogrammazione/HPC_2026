# Damped Wave HPC simulation

Here's a quick guide to build and run the damped wave simulation using `make`.

## Project Requirements

The project requires a system environment configured for parallel programming using **OpenMP** and **MPI**. The implementation must be compiled using standard C compilers and MPI-aware compilation tools.

* **GNU Compiler Collection (GCC)**
  The project must be compiled using `gcc` for standard C compilation and OpenMP support.

* **OpenMPI**
  MPI functionality requires the use of the `mpicc` compiler wrapper provided by OpenMPI. Before compiling or running the MPI-based programs, the OpenMPI environment module must be loaded in the execution shell:

  ```
  module load openmpi
  ```

    Notice that each system has its particularly named openmpi module, so find it with 
    ```
    module avail
    ```
* **Make**
  A `Makefile` is provided to automate the compilation process. `make` command should be available.

* **FFmpeg**
  FFmpeg must be installed and available in the system environment. It is required for multimedia processing tasks, such as reading, converting, or generating video/audio data used by the project.

## Execute 

### 1. Clone the repo
Clone the repo with `git clone`.

### 2. Variable Parameters
The variable parameters of the simulations are:
- The number of points of the 2D space grid M (defaulted to 1000 inside the Makefile);
- The number of frame generated N (defaulted to 1000 inside the Makefile);

### 2. Fixed Parameters
All the other parameters can be found inside the various files in ```damped_wave/parameters``` folder, either in explicit numbers or in functions of M and/or N.

We have chosen to generate 30 FPS videos with ffmpeg. 
The time step is thus fixed to obtain only the necessary number of frames per unit time. The fixed parameters (common to all the simulations) are thus:
- dx = 0.01, dt = 0.033
- Wave speed c = 0.55
- Damping coefficient gamma = 0.152

We have also:
- the starting x coordinate i0, variying depending on the wave;
- the starting y coordinate j0, variying depending on the wave;
- the intensity, variying depending on the wave;
- the frame where the pulse appears frame_start, variying depending on the wave;

### 2. Build and run with default parameters
The makefile can run both the openmp and mpi simulations.
To build and run the openmpi one run 
```bash
make run_omp
```
and then to generate the video, that will be inside ```damped_wave/openmp/video```, run 
```bash 
make video_omp
```

To run instead the mpi simulation run
```bash
make run_mpi
```
and then to generate the video, that will be inside ```damped_wave/MPI/video```, run 
```bash 
make video_mpi
```

### 6. Clean up all generated files (including videos)
```bash
make clean
```

## Credits
This project uses [TinyExpr](https://github.com/codeplea/tinyexpr) parser developed by codeplea, whom we thanks, to parse algebraic expressions for M and N.