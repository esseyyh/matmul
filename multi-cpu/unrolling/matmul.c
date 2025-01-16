#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <immintrin.h>
double get_time() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
}




void matmul(float *A, float *B, float *C, int M, int N, int K, int TILE_SIZE) {
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {



                int max_i = (i + TILE_SIZE < M) ? i + TILE_SIZE : M;
                int max_j = (j + TILE_SIZE < N) ? j + TILE_SIZE : N;
                int max_k = (k + TILE_SIZE < K) ? k + TILE_SIZE : K;





                for (int ii = i; ii <max_i -5; ii+=6) {
                    for (int kk = k; kk < max_k; kk++) {

                        __m512 a0 = _mm512_set1_ps(A[i * K + k]);
                        __m512 a1 = _mm512_set1_ps(A[(i + 1) * K + k]);
                        __m512 a2 = _mm512_set1_ps(A[(i + 2) * K + k]);
                        __m512 a3 = _mm512_set1_ps(A[(i + 3) * K + k]);

                        __m512 a4 = _mm512_set1_ps(A[(i + 4) * K + k]);
                        __m512 a5 = _mm512_set1_ps(A[(i + 5) * K + k]);

                        for (int jj = j; jj < max_j; jj+=32) {
                           

                                __m512 b0 = _mm512_loadu_ps(&B[j * K + k]);
                                __m512 b1 = _mm512_loadu_ps(&B[(j+16) * K + k]);



                                __m512 c00 = _mm512_loadu_ps(&C[i * N + j]);
                                __m512 c01 = _mm512_loadu_ps(&C[i * N + j+16]);



                                
                                __m512 c10 = _mm512_loadu_ps(&C[(i+1) * N + j]);
                                __m512 c11 = _mm512_loadu_ps(&C[(i+1) * N + j+16]);
 
                                
                                __m512 c20 = _mm512_loadu_ps(&C[(i+2) * N + j]);
                                __m512 c21 = _mm512_loadu_ps(&C[(i+2) * N + j+16]);


                                __m512 c30 = _mm512_loadu_ps(&C[(i+3) * N + j]);
                                __m512 c31 = _mm512_loadu_ps(&C[(i+3) * N + j+16]);

                                __m512 c41 = _mm512_loadu_ps(&C[(i+4) * N + j]);
                                __m512 c42 = _mm512_loadu_ps(&C[(i+4) * N + j+16]);
                                
                                __m512 c51 = _mm512_loadu_ps(&C[(i+5) * N + j]);
                                __m512 c52 = _mm512_loadu_ps(&C[(i+5) * N + j+16]);

                            //__m256 b0 = _mm256_loadu_ps(&B[j * K + k]);
                            //__m256 b1 = _mm256_loadu_ps(&B[(j+8) * K + k]);

                            //__m256 c00 = _mm256_loadu_ps(&C[i * N + j]);
                            //__m256 c01 = _mm256_loadu_ps(&C[i * N + j + 8]);
                            //__m256 c10 = _mm256_loadu_ps(&C[(i+1) * N + j]);
                            //__m256 c11 = _mm256_loadu_ps(&C[(i+1) * N + j + 8]);
                            //__m256 c20 = _mm256_loadu_ps(&C[(i+2) * N + j]);
                            //__m256 c21 = _mm256_loadu_ps(&C[(i+2) * N + j + 8]);
                            //__m256 c30 = _mm256_loadu_ps(&C[(i+3) * N + j]);
                            //__m256 c31 = _mm256_loadu_ps(&C[(i+3) * N + j + 8]);

                            c00 = _mm512_fmadd_ps(a0, b0, c00);
                            c01 = _mm512_fmadd_ps(a0, b1, c01);
                            c10 = _mm512_fmadd_ps(a1, b0, c10);
                            c11 = _mm512_fmadd_ps(a1, b1, c11);
                            c20 = _mm512_fmadd_ps(a2, b0, c20);
                            c21 = _mm512_fmadd_ps(a2, b1, c21);
                            c30 = _mm512_fmadd_ps(a3, b0, c30);
                            c31 = _mm512_fmadd_ps(a3, b1, c31);

                            c41 = _mm512_fmadd_ps(a4, b0, c41);
                            c42 = _mm512_fmadd_ps(a4, b1, c42);
                            c51 = _mm512_fmadd_ps(a5, b0, c51);
                            c52 = _mm512_fmadd_ps(a5, b1, c52);
                            
                            _mm512_storeu_ps(&C[i * N + j], c00);
                            _mm512_storeu_ps(&C[i * N + j + 16], c01);
                            _mm512_storeu_ps(&C[(i+1) * N + j], c10);
                            _mm512_storeu_ps(&C[(i+1) * N + j + 16], c11);
                            _mm512_storeu_ps(&C[(i+2) * N + j], c20);
                            _mm512_storeu_ps(&C[(i+2) * N + j + 16], c21);
                            _mm512_storeu_ps(&C[(i+3) * N + j], c30);
                            _mm512_storeu_ps(&C[(i+3) * N + j + 16], c31);

                            _mm512_storeu_ps(&C[(i+4) * N + j], c41);
                            _mm512_storeu_ps(&C[(i+4) * N + j + 16], c42);
                            _mm512_storeu_ps(&C[(i+5) * N + j], c51);
                            _mm512_storeu_ps(&C[(i+5) * N + j + 16], c52);
                        }
                    }
                }
            
        for (int i0 = max_i - (max_i % 4); i0 < max_i; i0++) {
                    for (int k0 = k; k0 < max_k; k0++) {
                        __m256 a = _mm256_broadcast_ss(&A[i0 * K + k0]);
                        for (int j0 = j; j0 < max_j - 7; j0 += 8) {
                            __m256 b = _mm256_loadu_ps(&B[j0 * K + k0]);
                            __m256 c = _mm256_loadu_ps(&C[i0 * N + j0]);
                            c = _mm256_fmadd_ps(a, b, c);
                            _mm256_storeu_ps(&C[i0 * N + j0], c);
                        }
                    }
                }

                for (int i0 = i; i0 < max_i; i0++) {
                    for (int k0 = k; k0 < max_k; k0++) {
                        float a = A[i0 * K + k0];
                        for (int j0 = max_j - (max_j % 8); j0 < max_j; j0++) {
                            C[i0 * N + j0] += a *B[j0 * K + k0];
                        }
                    }
                }

            }
        }
    }



}


void initialize_matrix(float *matrix, int M, int K)
{
        for (int j =  0; j < M * K; j++) {
                matrix[j] = (float)rand() / RAND_MAX;
            }
}


int main(int argc, char* argv[]) {

   int rank, size;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) { printf("Matrix multiplication\n");
        printf("Cores,Tile size ,m,n,k,time,  gflops for matmul\n");
    }
   //srand(time(NULL));

    int sizes[][3] = {
        //{128, 128, 128}, {512, 512, 512}, {1024,
        //1024, 1024},{2048,2048,2048},{4096,4096,4096},
        {8192,8192,8192},{16384,16384,16384},{32768,32768,32768},{65536,65536,65536}};
    
    int tile_sizes[] = {2,4,8,16,32,64};

    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {


        int rows_per_proc =sizes[i][0] / size;

        if (rank == size - 1) {
            rows_per_proc += sizes[i][0] % size;
        }

        float  *A = NULL,
               *B = NULL,
               *C = NULL;
        
        float *local_A =(float*)malloc(rows_per_proc * sizes[i][1] * sizeof(float));
        
        float *local_C =(float*)malloc(rows_per_proc * sizes[i][2] * sizeof(float));
        

        if (rank == 0) {
            A = (float*)malloc(sizes[i][0]* sizes[i][1]* sizeof(float));
            B = (float*)malloc(sizes[i][1]* sizes[i][2]* sizeof(float));
            C = (float*)malloc(sizes[i][0]* sizes[i][2]* sizeof(float));
            if (!A || !B || !C) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            initialize_matrix(A, sizes[i][0],sizes[i][1]);
            initialize_matrix(B, sizes[i][1],sizes[i][2]);
            initialize_matrix(C, sizes[i][0],sizes[i][2]);
            }
        

        if (rank != 0) { 
            B = (float*)malloc(sizes[i][1] *sizes[i][2] * sizeof(float));
            if (!B) {
                fprintf(stderr, "Memory allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        


        MPI_Bcast(B, sizes[i][1] * sizes[i][2], MPI_FLOAT, 0,MPI_COMM_WORLD);
        int *sendcounts = (int*)malloc(size *sizeof(int));
        int *displs = (int*)malloc(size *sizeof(int));

        if (rank == 0) { int offset = 0; for (int j = 0; j < size;
            j++) {
                sendcounts[j] = (j == size - 1) ? (sizes[i][0] - offset) * sizes[i][1] :(sizes[i][0] / size) * sizes[i][1]; 
                displs[j] =  offset * sizes[i][1]; offset += sizes[i][0] / size;
            }
        }
        

        MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT, local_A, rows_per_proc * sizes[i][1], MPI_FLOAT, 0,MPI_COMM_WORLD);

        //if (rank == 0) {
        //      double start_time = get_time();
        //}

        for (int j=0 ; j < 3; j++) {
        int tile_size = tile_sizes[j];
        double start_time, end_time;

        start_time = MPI_Wtime();
        MPI_Bcast(&start_time, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        matmul(local_A, B, local_C, rows_per_proc, sizes[i][1], sizes[i][2], tile_size);

//        for (int a = 0; a < rows_per_proc; a++) { 
//                for (int b = 0; b< sizes[i][2]; b++) {
//                    local_C[a * sizes[i][2] + b] = 0.0; 
//                    for (int c = 0;c < sizes[i][1]; c++) {
//                        local_C[a * sizes[i][2] + b] += local_A[a * sizes[i][1] + c] * B[c * sizes[i][2] + b];
//                }
//            }
//        }

         end_time = MPI_Wtime();
        if (rank == 0) {
                double elapsed_time = end_time - start_time;
                double flops = 2.0 * sizes[i][0] * sizes[i][1] * sizes[i][2];
                double gflops = flops / (elapsed_time * 1e9);
                printf("%d,%d,%d,%d,%d,%.6f,%.10f\n", size,tile_size,sizes[i][1], sizes[i][1],sizes[i][1],elapsed_time, gflops);
        }
        }
        //};
        MPI_Gatherv(local_C, rows_per_proc * sizes[i][2],MPI_FLOAT,C, sendcounts, displs, MPI_FLOAT, 0,MPI_COMM_WORLD);

        free(local_A);
        free(local_C);
        free(B);

        if (rank == 0) {
                free(A);
                free(C);
        }


     
        free(sendcounts);
        free(displs);
    }
        

        MPI_Finalize(); 
return 0;
}



