#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
double get_time() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
}





void matmul(float *A, float *B, float *C, int M, int N, int K, int TILE_SIZE) {
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE && ii < M; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE && jj < N; jj++) {
                        float acc = 0.0f;
                        for (int kk = k; kk < k + TILE_SIZE && kk < K; kk++) {
                            acc += A[ii * K + kk] * B[jj * K + kk];
                        }
                        C[ii * N + jj] += acc;
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

    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024,
    1024, 1024},{2048,2048,2048},{4096,4096,4096}};
    
    int tile_sizes[] = {16,32,64};

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

        
        double start_time, end_time;

        start_time = MPI_Wtime();
        MPI_Bcast(&start_time, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //matmul(local_A, B, local_C, rows_per_proc, sizes[i][1], sizes[i][2], tile_size);

        for (int a = 0; a < rows_per_proc; a++) { 
                for (int b = 0; b< sizes[i][2]; b++) {
                    local_C[a * sizes[i][2] + b] = 0.0; 
                    for (int c = 0;c < sizes[i][1]; c++) {
                        local_C[a * sizes[i][2] + b] += local_A[a * sizes[i][1] + c] * B[c * sizes[i][2] + b];
                }
            }
        }

         end_time = MPI_Wtime();
        if (rank == 0) {
                double elapsed_time = end_time - start_time;
                double flops = 2.0 * sizes[i][0] * sizes[i][1] * sizes[i][2];
                double gflops = flops / (elapsed_time * 1e9);
                printf("%d,%d,%d,%d,%.6f,%.10f\n", size,sizes[i][0], sizes[i][1],sizes[i][2],elapsed_time, gflops);
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



