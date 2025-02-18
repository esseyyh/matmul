
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

void transpose(float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void matmul(float *A, float *B, float *C, int M, int N, int K, int TILE_SIZE) {
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE && ii < M; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE && jj < N; jj++) {
                        __m512 acc = _mm512_setzero_ps();
                        for (int kk = k; kk < k + TILE_SIZE && kk < K; kk+=16) {
		            __m512 a = _mm512_loadu_ps(&A[ii * K + kk]);
		            __m512 b = _mm512_loadu_ps(&B[jj * K + kk]);
                            acc = _mm512_fmadd_ps(a, b, acc);
                        }
                        float sum = _mm512_reduce_add_ps(acc);
                        C[ii * N + jj] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    int sizes[][3] = {{512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int tile_sizes[] = {16,32,64};
    int num_tile_sizes = sizeof(tile_sizes) / sizeof(tile_sizes[0]);

    srand(time(NULL));

    printf("m,n,k,tile_size,time,flops\n");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *B_transposed = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));

        for (int j = 0; j < M * K; j++) {
            A[j] = (float)rand() / RAND_MAX;
        }
        for (int j = 0; j < K * N; j++) {
            B[j] = (float)rand() / RAND_MAX;
        }

        transpose(B, B_transposed, K, N);

        for (int t = 0; t < num_tile_sizes; t++) {
            int TILE_SIZE = tile_sizes[t];

            for (int j = 0; j < M * N; j++) {
                C[j] = 0.0f;
            }

            clock_t start_time = clock();

            matmul(A, B_transposed, C, M, N, K, TILE_SIZE);

            clock_t end_time = clock();

            double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / iteration_time / 1e9;

            printf("%d,%d,%d,%d,%.6f,%.2f\n", M, N, K, TILE_SIZE, iteration_time, flops_per_second);
        }

        free(A);
        free(B);
        free(B_transposed);
        free(C);
    }

    return 0;
}
