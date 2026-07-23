# Common directories (Openmp and MPI)
COMMON_SUBDIR = damped_wave/general_functions external/tinyexpr

# Directories containing the source code to compile (Openmp)
SUBDIRS_OMP = damped_wave/openmp/src 

# Directories containing the source code to compile (MPI)
SUBDIRS_MPI = damped_wave/MPI/src 

# Directory where OpenMP simulation frames are stored
SIM_DIR_OMP = damped_wave/openmp/sim

# Directories where MPI simulation frames are stored
SIM_DIR_MPI_1 = damped_wave/MPI/sim1
SIM_DIR_MPI_2 = damped_wave/MPI/sim2
SIM_DIR_MPI_3 = damped_wave/MPI/sim3

# Output video (OpenMP)
VIDEO_OMP = damped_wave/openmp/video/openmp_simulation.mp4

# Output videos (MPI)
VIDEO_MPI_1 = damped_wave/MPI/video/mpi_simulation_1.mp4
VIDEO_MPI_2 = damped_wave/MPI/video/mpi_simulation_2.mp4
VIDEO_MPI_3 = damped_wave/MPI/video/mpi_simulation_3.mp4

# Compiler options

# Use the GNU C compiler
CC      = gcc
MCC 	= mpicc


# Enable OpenMP, optimize for speed, enable warnings, and search for header files in the current directory       
CFLAGS  = -fopenmp -O3 -Wall -Wextra -I.

# Link against the math library
LDLIBS  = -lm

# Find all .c source files in the listed directories
SRCS_OMP    = $(wildcard $(addsuffix /*.c,$(COMMON_SUBDIR))) $(wildcard $(addsuffix /*.c,$(SUBDIRS_OMP)))

SRCS_MPI    = $(wildcard $(addsuffix /*.c,$(COMMON_SUBDIR))) $(wildcard $(addsuffix /*.c,$(SUBDIRS_MPI)))

# Generate the corresponding object (.o) file names
OBJS_OMP    = $(SRCS_OMP:.c=.o)
OBJS_MPI    = $(SRCS_MPI:.c=.o)

# Name of the final executable
TARGET_OMP  = wave_sim_omp.out
TARGET_MPI  = wave_sim_mpi.out

# Params
M ?= 1000
N ?= 1000

.PHONY: all build run clean video build_omp build_mpi run_omp run_mpi clean_omp clean_mpi video_omp video_mpi    # These targets are commands, not files

all: build run video                        # Running "make" is equivalent to running "make build"

omp: build_omp run_omp video_omp

mpi: build_mpi run_mpi video_mpi

build: build_omp build_mpi         # Build the executable

run: run_omp run_mpi

clean: clean_omp clean_mpi         # Remove the executable and all generated object files, simulation files and video

#list:                               # Print the source directories included in the build
#	@printf "Included directories:"
#	@for d in $(SUBDIRS); do printf "   $$d"; done

video: video_omp video_mpi

build_omp: $(TARGET_OMP)

build_mpi: $(TARGET_MPI)

# This is for all files, not needed to use mpicc
%.o: %.c                            # Compile each source file into its corresponding object file
	$(CC) $(CFLAGS) -c $< -o $@

# MPI source files
damped_wave/MPI/src/%.o: damped_wave/MPI/src/%.c
	$(MCC) $(CFLAGS) -c $< -o $@

$(TARGET_OMP): $(OBJS_OMP)                  # Link all object files to create the executable
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	@printf "=== Built: %s ===\n" "$@"

$(TARGET_MPI): $(OBJS_MPI)                  # Link all object files to create the executable
	$(MCC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	@printf "=== Built: %s ===\n" "$@"

run_omp: $(SIM_DIR_OMP)/simulation.done

# Use only one flag file
run_mpi: $(SIM_DIR_MPI_1)/simulation.done

$(SIM_DIR_OMP)/simulation.done: $(TARGET_OMP)		# Build the executable if needed, then run the simulation
	@printf "=== Running $(TARGET_OMP) ===\n"
	./$(TARGET_OMP) $(M) $(N)
	touch $@

$(SIM_DIR_MPI_1)/simulation.done : $(TARGET_MPI)
	@printf "=== Running $(TARGET_MPI) ===\n"
	mpirun -np 3 ./$(TARGET_MPI) $(M) $(N)
	touch $@

clean_omp:
	rm -f $(TARGET_OMP) $(OBJS_OMP)
	rm -f $(SIM_DIR_OMP)/*.pgm 
	rm -f $(VIDEO_OMP)

clean_mpi:
	rm -f $(TARGET_MPI) $(OBJS_MPI)
	rm -f $(SIM_DIR_MPI_1)/*.pgm $(SIM_DIR_MPI_2)/*.pgm $(SIM_DIR_MPI_3)/*.pgm 
	rm -f $(VIDEO_MPI_1) $(VIDEO_MPI_2) $(VIDEO_MPI_3)

video_omp: run_omp
	@printf "=== Creating video ==="
	ffmpeg -framerate 30 \
	       -i $(SIM_DIR_OMP)/frame_%05d.pgm \
	       -c:v libx264 -pix_fmt yuv420p \
	       $(VIDEO_OMP)

video_mpi: run_mpi
	@printf "=== Creating videos ==="
	ffmpeg -framerate 30 \
	       -i $(SIM_DIR_MPI_1)/frame_%05d.pgm \
	       -c:v libx264 -pix_fmt yuv420p \
	       $(VIDEO_MPI_1)
	ffmpeg -framerate 30 \
	       -i $(SIM_DIR_MPI_2)/frame_%05d.pgm \
	       -c:v libx264 -pix_fmt yuv420p \
	       $(VIDEO_MPI_2)
	ffmpeg -framerate 30 \
	       -i $(SIM_DIR_MPI_3)/frame_%05d.pgm \
	       -c:v libx264 -pix_fmt yuv420p \
	       $(VIDEO_MPI_3)

# Aggiungi -g mantenendo -O3 (VTune funziona bene anche con ottimizzazioni attive)
CFLAGS = -fopenmp -O3 -g -Wall -Wextra -I.