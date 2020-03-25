
NVCC ?= /usr/local/cuda-10.2/bin/nvcc

# 75 for Turing architecture.
# requires: cuda 10.2
SM ?= 50

OMP_FLAGS = -Xcompiler -fopenmp

NVFLAGS = -I. -O0 -std=c++14 -arch sm_$(SM) --relocatable-device-code=true --extended-lambda $(OMP_FLAGS) -g
#  --cudart static
#         --gpu-architecture=sm_50' is equivalent to 'nvcc --gpu-architecture=compute_50
#         --gpu-code=sm_50,compute_50'.

default:
	$(NVCC) $(NVFLAGS) gpma_bfs_demo.cu -o gpma_bfs_demo -lgomp

MINI_TEST_DEV ?= GPU

mini:
	$(NVCC) $(NVFLAGS) mini.cu -o mini -lgomp -DDEBUG -DTEST_DEV=$(MINI_TEST_DEV)

format:
	clang-format --style=file -i *.cuh *.cu *.hpp

.PHONY: mini default
clean:
	rm -f gpma_bfs_demo mini
