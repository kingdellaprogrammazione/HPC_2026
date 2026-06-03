# Top-level Makefile – delegates to wave_sim/ (compile only)
SUBDIR = damped_wave/openmp

.PHONY: all build run clean $(SUBDIR)

all: build

build: $(SUBDIR)

$(SUBDIR):
	$(MAKE) -C $(SUBDIR) $(TARGET)   # compile only, don't run

run:
	$(MAKE) -C $(SUBDIR) run         # run with current parameters

clean:
	$(MAKE) -C $(SUBDIR) clean

# Optional: one command to compile and run
run_all: build run