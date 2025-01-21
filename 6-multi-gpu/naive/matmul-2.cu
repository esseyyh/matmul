#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_THREADS 1024
#define NUM_REPETITIONS 5

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

void matmul(float *A, float *B, float *C, int M, int N, int K) {
  // float *d_A, *d_B, *d_C;

  int M_per_gpu = M / 2;

  // Device pointers for each GPU
  float *d_A[2], *d_B[2], *d_C[2];

  for (int gpu = 0; gpu < 2; ++gpu) {
    cudaSetDevice(gpu);

    // Allocate memory on the current GPU
    cudaMalloc(&d_A[gpu], M_per_gpu * K * sizeof(float));
    cudaMalloc(&d_B[gpu], K * N * sizeof(float));
    cudaMalloc(&d_C[gpu], M_per_gpu * N * sizeof(float));

    // Copy relevant parts of A and full B to the GPU
    cudaMemcpy(d_A[gpu], A + gpu * M_per_gpu * K, M_per_gpu * K * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B[gpu], B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  }

  // size_t size_A = M * K * sizeof(float);
  // size_t size_B = K * N * sizeof(float);
  // size_t size_C = M * N * sizeof(float);

  // cudaMalloc(&d_A, size_A);
  // cudaMalloc(&d_B, size_B);
  // cudaMalloc(&d_C, size_C);

  // cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  dim3 blockDim(16, 16); // Each block contains 16x16 threads
  dim3 gridDim((N + 15) / 16, (M_per_gpu + 15) / 16);
  for (int gpu = 0; gpu < 2; ++gpu) {
    cudaSetDevice(gpu);
    matmul_kernel<<<gridDim, blockDim>>>(d_A[gpu], d_B[gpu], d_C[gpu],
                                         M_per_gpu, N, K);
  }

  // Copy results back to host
  for (int gpu = 0; gpu < 2; ++gpu) {
    cudaSetDevice(gpu);
    cudaMemcpy(C + gpu * M_per_gpu * N, d_C[gpu], M_per_gpu * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A[gpu]);
    cudaFree(d_B[gpu]);
    cudaFree(d_C[gpu]);
  }
  // dim3 block_size(32, 32);

  // dim3 grid_size((N + block_size.x - 1) / block_size.x,
  //               (M + block_size.y - 1) / block_size.y);

  // matmul_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

  // cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
  int sizes[][3] = {{128, 128, 128},      {512, 512, 512},
                    {1024, 1024, 1024},   {2048, 2048, 2048},
                    {4096, 4096, 4096},   {8192, 8192, 8192},
                    {16384, 16384, 16384}};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads = 4; // Default value
  if (argc > 1) {
    num_threads = atoi(argv[1]);
    if (num_threads <= 0 || num_threads > MAX_THREADS) {
      fprintf(stderr, "Invalid number of threads. Using default (4).\n");
      num_threads = 4;
    }
  }
  srand(time(NULL));

  printf("m,n,k,time,gflops\n");

  double best_gflops = 0.0;
  int best_m = 0, best_n = 0, best_k = 0;

  for (int i = 0; i < num_sizes; i++) {
    int M = sizes[i][0];
    int N = sizes[i][1];
    int K = sizes[i][2];
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    if (!A || !B || !C) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }

    for (int j = 0; j < M * K; j++) {
      A[j] = (float)rand() / RAND_MAX;
    }
    for (int j = 0; j < K * N; j++) {
      B[j] = (float)rand() / RAND_MAX;
    }

    double total_time = 0.0;
    double min_time = DBL_MAX;

    for (int rep = 0; rep < NUM_REPETITIONS; rep++) {
      double start_time = get_time();
      matmul(A, B, C, M, N, K);
      double end_time = get_time();
      double elapsed_time = end_time - start_time;

      total_time += elapsed_time;
      if (elapsed_time < min_time) {
        min_time = elapsed_time;
      }
    }

    double avg_time = total_time / NUM_REPETITIONS;
    double flops = 2.0 * M * N * K;
    double avg_gflops = flops / (avg_time * 1e9);
    double max_gflops = flops / (min_time * 1e9);

    printf("%d,%d,%d,%.6f,%.2f\n", M, N, K, avg_time, avg_gflops);

    if (max_gflops > best_gflops) {
      best_gflops = max_gflops;
      best_m = M;
      best_n = N;
      best_k = K;
    }

    free(A);
    free(B);
    free(C);
  }

  printf("\nBest configuration:\n");
  printf("M=%d, N=%d, K=%d\n", best_m, best_n, best_k);
  printf("Best performance: %.2f GFLOPS\n", best_gflops);

  return 0;
}
