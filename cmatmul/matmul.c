#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void matmul(float *A,float *B,float *C, int i, int j , int k){

    for (int ii=0;ii<i;ii++){

        for (int jj=0;jj<j;jj++){
            float value=0;
            for (int kk=0;kk<k;kk++){
               value= A[ii * k  + kk ]  * B [j * kk + jj ];
            }
            C[ii * j +  jj ]=value;

        }
    }



}


int main(){


    int sizes [][3]= { {128,128,128}, {256,256,256}, {512,512,512}};
    int iteration=10;

    //printf("%d \n",(sizeof(sizes)/sizeof(sizes[0])));
    srand(time(NULL));
    printf("m,n,k,time,flops\n");
    for (int i=0 ; i < sizeof(sizes)/sizeof(sizes[0]) ; i++){
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));

        for (int j = 0; j < M * K; j++) {
            A[j] = (float)rand() / RAND_MAX;
        }
        for (int j = 0; j < K * N; j++) {
            B[j] = (float)rand() / RAND_MAX;
        }

        
        for (int iter = 0; iter < iteration; iter++) {
            clock_t start_time = clock();
            matmul(A,B,C, M, N,K);
            clock_t end_time = clock();

            double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / iteration_time / 1e9;  // Convert to gigaflops

            printf("%d,%d,%d,%.6f,%.2f\n", M, N, K, iteration_time, flops_per_second);
        }



    
      
        free(A);
        free(B);
        free(C);

    }
    return 0;



}
