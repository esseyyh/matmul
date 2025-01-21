#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

void matmul(float *A,float *B, float *C, int M, int K, int N, int tile_size){


    for(int i=0; i<M;i+=tile_size){
        for(int k=0; k<K;k+=tile_size){
            for(int j=0; j<N;j+=tile_size){

                for(int ii=i; ii<M && i<i+tile_size ;ii++){
                    for(int kk=k; kk<K && kk<k+tile_size ;kk++){
			 _mm_prefetch((char*)&A[ii *K + kk+ tile_size], _MM_HINT_T0);
			 //__m256 a = _mm256_broadcast_ss(&A[ii * K + kk]);
                         //_mm_prefetch((char*)&A[ii * K + kk + 4], _MM_HINT_T0);
                        __m512 a = _mm512_set1_ps(A[ii * K + kk]); // Broadcast A[ii, kk]
                        for(int jj=j; jj<tile_size+j && jj<N;jj+=16){
                           // _mm_prefetch((char*)&B[jj *K + jj+ 4], _MM_HINT_T0);
		            __m512 b = _mm512_loadu_ps(&B[jj * K + kk]);
                            __m512 c = _mm512_loadu_ps(&C[ii * N + jj]);
                            c = _mm512_fmadd_ps(a, b, c);
                            _mm512_storeu_ps(&C[ii * N + jj], c);
			}

        }}

}}}
}




void transpose(float* A,float *B,int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
        B[j * m+ i ] = A [ i * n + j];
        }
    }
}


int main() {


    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024},{2048,2048,2048}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int tile_sizes[] = {16, 32, 64,128};
    int num_tile_sizes = sizeof(tile_sizes) / sizeof(tile_sizes[0]);

    srand(time(NULL));

    printf("m,n,k,tile_size,time,flops\n");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];

        float *A = (float *)aligned_alloc(32, M * K * sizeof(float));
        float *B = (float *)aligned_alloc(32, K * N * sizeof(float));
        float *B_transposed = (float *)aligned_alloc(32, K * N * sizeof(float));
        float *C = (float *)aligned_alloc(32, M * N * sizeof(float));

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
