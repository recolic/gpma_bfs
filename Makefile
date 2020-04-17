
NVCC ?= /usr/local/cuda-10.2/bin/nvcc

# 75 for Turing architecture.
# requires: cuda 10.2
SM ?= 50

TEST_DEV ?= GPU

OMP_FLAGS = -Xcompiler -fopenmp

NVFLAGS = -I. -O3 -std=c++14 -DCUDA_SM=$(SM) -arch sm_$(SM) --relocatable-device-code=true --extended-lambda $(OMP_FLAGS) -g

default:
	$(NVCC) $(NVFLAGS) gpma_bfs_demo.cu -o gpma_bfs_demo -lgomp


mini:
	$(NVCC) $(NVFLAGS) mini.cu -o mini -lgomp -DDEBUG -DTEST_DEV=$(TEST_DEV)

format:
	clang-format --style=file -i *.cuh *.cu *.hpp

.PHONY: mini default
clean:
	rm -f gpma_bfs_demo mini
