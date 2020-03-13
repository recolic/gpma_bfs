NVCC = nvcc

# 75 for Turing architecture.

default:
	$(NVCC) -I. -O3 -std=c++14 -gencode arch=compute_75,code=sm_75 gpma_bfs_demo.cu --cudart static --relocatable-device-code=true -o gpma_bfs_demo

gpma_bfs_demo:
	$(NVCC) -I./ -O3 -std=c++14 -w -gencode arch=compute_75,code=sm_75 -odir "." -M -o "gpma_bfs_demo.d" "./gpma_bfs_demo.cu"
	$(NVCC) -I./ -O3 -std=c++14 -w --compile --relocatable-device-code=true -gencode arch=compute_75,code=sm_75 -x cu -o "gpma_bfs_demo.o" "gpma_bfs_demo.cu"
	$(NVCC) --cudart static --relocatable-device-code=true -gencode arch=compute_75,code=sm_75 -link -o "gpma_bfs_demo" ./gpma_bfs_demo.o

clean:
	rm ./gpma_bfs_demo.o ./gpma_bfs_demo.d gpma_bfs_demo
