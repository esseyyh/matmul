#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NUM_REPETITIONS 1
#define TILE_SIZE 64

__global__ void matmul_kernel_tiled(float *A, float *B, float *C, int M, int N,
                                    int K) {
  __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
  __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float sum = 0.0f;

  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    if (row < M && tile * TILE_SIZE + tx < K) {
      A_shared[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
    } else {
      A_shared[ty][tx] = 0.0f;
    }

    if (col < N && tile * TILE_SIZE + ty < K) {
      B_shared[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
    } else {
      B_shared[ty][tx] = 0.0f;
    }

    __syncthreads();

    float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k = 0; k < TILE_SIZE; k += 4) {
      // Load 4 consecutive elements from A_shared and B_shared into float4
      float4 a_vec = *reinterpret_cast<float4 *>(&A_shared[ty][k]);
      float4 b_vec = make_float4(B_shared[k][tx], B_shared[k + 1][tx],
                                 B_shared[k + 2][tx], B_shared[k + 3][tx]);

      // Perform element-wise multiplication and accumulate
      sum4.x += a_vec.x * b_vec.x;
      sum4.y += a_vec.y * b_vec.y;
      sum4.z += a_vec.z * b_vec.z;
      sum4.w += a_vec.w * b_vec.w;
    }

    // Combine results from sum4 back into a scalar sum
    sum += sum4.x + sum4.y + sum4.z + sum4.w;
    // #pragma unroll
    //    for (int k = 0; k < TILE_SIZE; ++k) {
    //      sum += A_shared[ty][k] * B_shared[k][tx];
    //    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

void matmul(float *A, float *B, float *C, int M, int N, int K) {
  float *d_A, *d_B, *d_C;

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));

  CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

  matmul_kernel_tiled<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

  CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// main function
int main(int argc, char *argv[]) {
  int sizes[][3] = {{128, 128, 128},      {512, 512, 512},
                    {1024, 1024, 1024},   {2048, 2048, 2048},
                    {4096, 4096, 4096},   {8192, 8192, 8192},
                    {16384, 16384, 16384}};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
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
      // double start_time = get_time();

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      cudaDeviceSynchronize();
      matmul(A, B, C, M, N, K);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float elapsed_time_ms;
      cudaEventElapsedTime(&elapsed_time_ms, start, stop);
      double elapsed_time = elapsed_time_ms / 1000.0;

      // double end_time = get_time();
      // double elapsed_time = end_time - start_time;

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
