NVCC = /usr/local/cuda-10.2/bin/nvcc

# 75 for Turing architecture.

gpma_bfs_demo:
	$(NVCC) -I./ -O3 -std=c++11 -w -gencode arch=compute_75,code=sm_75 -odir "." -M -o "gpma_bfs_demo.d" "./gpma_bfs_demo.cu"
	$(NVCC) -I./ -O3 -std=c++11 -w --compile --relocatable-device-code=true -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75 -x cu -o "gpma_bfs_demo.o" "gpma_bfs_demo.cu"
	$(NVCC) --cudart static --relocatable-device-code=true -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75 -link -o "gpma_bfs_demo" ./gpma_bfs_demo.o

clean:
	rm ./gpma_bfs_demo.o ./gpma_bfs_demo.d gpma_bfs_demo
