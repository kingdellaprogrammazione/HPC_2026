# Top-level Makefile - Single executable with cwalk
SUBDIRS = damped_wave/openmp damped_wave/general_functions 

CC      = gcc
CFLAGS  = -fopenmp -O3 -Wall -Wextra -I. 
LDLIBS  = -lm

# Default simulation parameters
M         ?= 50
N         ?= 100
dx        ?= 1
dt        ?= 0.033
c         ?= 0.55
gamma     ?= 0.152
i0        ?= 25
j0        ?= 25
intensity ?= -37

SRCS    = $(wildcard $(addsuffix /*.c,$(SUBDIRS)))
OBJS    = $(SRCS:.c=.o)
TARGET  = wave_sim.out

.PHONY: all build run clean

all: build

build: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	@echo "=== Built: $@ ==="

run: $(TARGET)
	./$(TARGET) $(M) $(N) $(dx) $(dt) $(c) $(gamma) $(i0) $(j0) $(intensity)

clean:
	rm -f $(TARGET) $(OBJS)

list:
	@echo "Included directories:"
	@for d in $(SUBDIRS); do echo "   $$d"; done
