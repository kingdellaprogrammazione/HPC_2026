# Directories containing the source code to compile
SUBDIRS = damped_wave/openmp damped_wave/general_functions

# Directory where simulation frames are stored
SIM_DIR = damped_wave/openmp/sim

# Filepath of final video
VIDEO = damped_wave/openmp/video/openmp_simulation.mp4

# Compiler options

# Use the GNU C compiler
CC      = gcc

# Enable OpenMP, optimize for speed, enable warnings, and search for header files in the current directory       
CFLAGS  = -fopenmp -O3 -Wall -Wextra -I.

# Link against the math library
LDLIBS  = -lm

# Default simulation parameters (can be overridden from the command line)

# Number of grid points each direction
M ?= 1000

# Number of time iterations
N ?= 1000 

# Spatial grid spacing  
dx ?= 0.05

# Time step   
dt ?= 0.033

# Wave propagation speed
c ?= 0.55

# Damping coefficient   
gamma ?= 0.152

# x-coordinate of the initial disturbance
i0 ?= 500

# y-coordinate of the initial disturbance  
j0 ?= 500

# Amplitude of the initial pulse
intensity ?= -37

# Find all .c source files in the listed directories
SRCS    = $(wildcard $(addsuffix /*.c,$(SUBDIRS)))

# Generate the corresponding object (.o) file names
OBJS    = $(SRCS:.c=.o)

# Name of the final executable
TARGET  = wave_sim.out

.PHONY: all build run clean list video    # These targets are commands, not files

all: build run video                        # Running "make" is equivalent to running "make build"

build: $(TARGET)                    # Build the executable

%.o: %.c                            # Compile each source file into its corresponding object file
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)                  # Link all object files to create the executable
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	@printf "=== Built: %s ===\n" "$@"

run: $(SIM_DIR)/simulation.done

$(SIM_DIR)/simulation.done: $(TARGET)		# Build the executable if needed, then run the simulation
	@printf "=== Running $(TARGET) ===\n"
	@printf "Grid: M=$(M), N=$(N)\n"
	@printf "dx=$(dx), dt=$(dt), c=$(c), gamma=$(gamma)\n"
	@printf "Source: ($(i0), $(j0)), intensity=$(intensity)\n"
	./$(TARGET) $(M) $(N) $(dx) $(dt) $(c) $(gamma) $(i0) $(j0) $(intensity)
	touch $@

clean:                              # Remove the executable and all generated object files, simulation files and video
	rm -f $(TARGET) $(OBJS)
	rm -f $(SIM_DIR)/*.pgm
	rm -f $(VIDEO)

list:                               # Print the source directories included in the build
	@printf "Included directories:"
	@for d in $(SUBDIRS); do printf "   $$d"; done

video: run
	@printf "=== Creating video ==="
	ffmpeg -framerate 30 \
	       -i $(SIM_DIR)/frame_%05d.pgm \
	       -c:v libx264 -pix_fmt yuv420p \
	       $(VIDEO)

