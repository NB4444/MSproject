# Makefile for compiling all .cu files with nvcc

# Compiler
NVCC = nvcc

# Flags
CFLAGS =
#  -O3 -std=c++11

# Find all .cu files in the current directory
CU_FILES := $(wildcard *.cu)

# Get the filenames without the extension
EXECUTABLES := $(CU_FILES:.cu=)

# Rule to compile each .cu file into an executable
all: $(EXECUTABLES)

%: %.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Clean rule
clean:
	rm -f $(EXECUTABLES)

