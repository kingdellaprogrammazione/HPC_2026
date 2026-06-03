# Compiler and flags
CC      = gcc
CFLAGS  = -fopenmp -O3 -Wall -Wextra
LDLIBS  = -lm

# Default parameters
M         ?= 30
N         ?= 40
dx        ?= 0.01
dt        ?= 0.0000073
c         ?= 343
gamma     ?= 23
i0        ?= 10
j0        ?= 10
intensity ?= 54 


# Output binary name
TARGET  = wave_sim.out

# Source files (adjust if you have more .c files)
SRCS    = damped.c
OBJS    = $(SRCS:.c=.o)

# Default target: compile and run
all: $(TARGET)
	./$(TARGET) $(M) $(N) $(dx) $(dt) $(c) $(gamma) $(i0) $(j0) $(intensity)

# Compile
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) $(OBJS)

# Run with custom parameters (override defaults)
run: $(TARGET)
	./$(TARGET) $(M) $(N) $(dx) $(dt) $(c) $(i0) $(j0)

.PHONY: all clean run