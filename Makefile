NVCC ?= /usr/local/cuda-10.2/bin/nvcc

# $(arch) for Turing architecture.
# requires: cuda 10.2
arch = 75

default:
	$(NVCC) -I. -O3 -std=c++14 -gencode arch=compute_$(arch),code=sm_$(arch) gpma_bfs_demo.cu --cudart static --relocatable-device-code=true -o gpma_bfs_demo

mini:
	$(NVCC) -I. -O3 -std=c++14 -gencode arch=compute_$(arch),code=sm_$(arch) mini.cu --cudart static --relocatable-device-code=true -o mini

gpma_bfs_demo:
	$(NVCC) -I./ -O3 -std=c++14 -w -gencode arch=compute_$(arch),code=sm_$(arch) -odir "." -M -o "gpma_bfs_demo.d" "./gpma_bfs_demo.cu"
	$(NVCC) -I./ -O3 -std=c++14 -w --compile --relocatable-device-code=true -gencode arch=compute_$(arch),code=sm_$(arch) -x cu -o "gpma_bfs_demo.o" "gpma_bfs_demo.cu"
	$(NVCC) --cudart static --relocatable-device-code=true -gencode arch=compute_$(arch),code=sm_$(arch) -link -o "gpma_bfs_demo" ./gpma_bfs_demo.o

clean:
	rm -f gpma_bfs_demo.o gpma_bfs_demo.d gpma_bfs_demo mini
