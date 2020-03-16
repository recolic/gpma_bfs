NVCC ?= /usr/local/cuda-10.2/bin/nvcc

# $(arch) for Turing architecture.
# requires: cuda 10.2
arch = 75

NVFLAGS = -I. -O3 -std=c++14 -gencode arch=compute_$(arch),code=sm_$(arch) --relocatable-device-code=true
#  --cudart static

default:
	$(NVCC) $(NVFLAGS) gpma_bfs_demo.cu -o gpma_bfs_demo

mini:
	$(NVCC) $(NVFLAGS) mini.cu -o mini

clean:
	rm -f gpma_bfs_demo mini
