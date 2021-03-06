#include <iostream>

#include "gpma.cuh"
#include "gpma_bfs.cuh"

#ifndef TEST_DEV
#define TEST_DEV GPU
#endif

void load_data(const char *file_path, thrust::host_vector<int> &host_x, thrust::host_vector<int> &host_y, int &node_size, int &edge_size) {

    FILE *fp;
    fp = fopen(file_path, "r");
    if (not fp) {
        printf("Open graph file failed.\n");
        exit(0);
    }

    fscanf(fp, "%d %d", &node_size, &edge_size);
    printf("node_num: %d, edge_num: %d\n", node_size, edge_size);

    host_x.resize(edge_size);
    host_y.resize(edge_size);

    for (int i = 0; i < edge_size; i++) {
        int x, y;
        (void)fscanf(fp, "%d %d", &x, &y);
        host_x[i] = x;
        host_y[i] = y;
    }

    printf("Graph file is loaded.\n");
    fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Invalid arguments.\n");
        return -1;
    }

    char *data_path = argv[1];

    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024ll * 1024 * 1024);
    //cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 5);

    thrust::host_vector<int> host_x;
    thrust::host_vector<int> host_y;
    int node_size;
    int edge_size;
    load_data(data_path, host_x, host_y, node_size, edge_size);

    int half = edge_size / 2;
    thrust::host_vector<KEY_TYPE> h_base_keys(half);
    for (int i = 0; i < half; i++) {
        h_base_keys[i] = ((KEY_TYPE)host_x[i] << 32) + host_y[i];
    }

    NATIVE_VEC_KEY<TEST_DEV> base_keys = h_base_keys;
    NATIVE_VEC_VALUE<TEST_DEV> base_values(half, 1);
    cudaDeviceSynchronize();

    int num_slide = 1;
    int step = half / num_slide;

    LOG_TIME("before init_csr_gpma")
    GPMA<TEST_DEV> gpma(node_size);
    cudaDeviceSynchronize();

    LOG_TIME("before update_gpma 1")
    update_gpma(gpma, base_keys, base_values);
    cudaDeviceSynchronize();

    LOG_TIME("before main loop")
    for (int i = 0; i < num_slide; i++) {
        thrust::host_vector<KEY_TYPE> hk(step * 2);
        for (int j = 0; j < step; j++) {
            int idx = half + i * step + j;
            hk[j] = ((KEY_TYPE)host_x[idx] << 32) + host_y[idx];
        }
        for (int j = 0; j < step; j++) {
            int idx = i * step + j;
            hk[j + step] = ((KEY_TYPE)host_x[idx] << 32) + host_y[idx];
        }

        NATIVE_VEC_VALUE<TEST_DEV> update_values(step * 2);
        thrust::fill(update_values.begin(), update_values.begin() + step, 1);
        thrust::fill(update_values.begin() + step, update_values.end(), VALUE_NONE);
        NATIVE_VEC_KEY<TEST_DEV> update_keys = hk;
        cudaDeviceSynchronize();

        update_gpma(gpma, update_keys, update_values);
        cudaDeviceSynchronize();
    }
    printf("Graph is updated.\n");

    return 0;
}
