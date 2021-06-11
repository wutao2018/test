#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <cublas_v2.h>
#include "../include/util.h"
#include "kernel.h"

#define N_RUNS 10

int  main (int argc, char** argv) {

	ErrChk(cudaSetDevice(0));

	if(argc<2){
		printf("Usage: input the batch size\n");
		exit(EXIT_FAILURE);
	}

	int BATCH = 4; // atoi(argv[1]);
	//int TLP_thres = atoi(argv[2]);
	int TLP_thres = 65536;
	
	int *M;
	int *N;
	int *K;

	M = (int*) malloc(BATCH * sizeof(int));
	N = (int*) malloc(BATCH * sizeof(int));
	K = (int*) malloc(BATCH * sizeof(int));

	std::fstream fs;
	fs.open("../data/input");
	if (!fs.is_open()){
		printf("Error opening input\n");
		exit(EXIT_FAILURE);
	}
	
	//read matrix config	
	for (int i=0; i<BATCH; ++i){
		fs>>M[i]>>N[i]>>K[i];
	}

    float **A;
	float **B;
	float **C;

	A = (float**) malloc(BATCH * sizeof(float*));
	B = (float**) malloc(BATCH * sizeof(float*));
	C = (float**) malloc(BATCH * sizeof(float*));

	for (int i=0; i<BATCH; ++i){
		ErrChk(cudaMalloc((void**)&A[i], M[i]*K[i]*sizeof(float)));
		ErrChk(cudaMalloc((void**)&B[i], K[i]*N[i]*sizeof(float)));
		ErrChk(cudaMalloc((void**)&C[i], M[i]*N[i]*sizeof(float)));
	}

	float **dev_A;
	float **dev_B;
	float **dev_C;

    ErrChk(cudaMalloc((void**)&dev_A, BATCH*sizeof(float*)));
    ErrChk(cudaMalloc((void**)&dev_B, BATCH*sizeof(float*)));
    ErrChk(cudaMalloc((void**)&dev_C, BATCH*sizeof(float*)));

	ErrChk(cudaMemcpy(dev_A, A, BATCH*sizeof(float*), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(dev_B, B, BATCH*sizeof(float*), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(dev_C, C, BATCH*sizeof(float*), cudaMemcpyHostToDevice));


	int *dev_M, *dev_N, *dev_K;
	ErrChk(cudaMalloc((void**)&dev_M, BATCH*sizeof(int)));
	ErrChk(cudaMalloc((void**)&dev_N, BATCH*sizeof(int)));
	ErrChk(cudaMalloc((void**)&dev_K, BATCH*sizeof(int)));

	ErrChk(cudaMemcpy(dev_M, M, BATCH*sizeof(int), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(dev_N, N, BATCH*sizeof(int), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(dev_K, K, BATCH*sizeof(int), cudaMemcpyHostToDevice));

	
	float elapsedTime = 0.f;
    double time=0.f;
	float gflops_per_sec = 0.f;
	double gflops = 0.f;
	for (int i=0; i<BATCH; ++i)  // A*B + C
		gflops += ((2 * int64_t(M[i]) * int64_t(N[i]) * int64_t(K[i])) + (2 * int64_t(M[i]) * int64_t(N[i])) ) / 1.0e9;
	cudaEvent_t start, stop;

	//compute grid size and block size

	//int kThreads = 128;
	int TLP = 0;

	const int tile_size[6][2] = {
		16, 16,
		32, 32,
		64, 64,
		128, 64,
		64, 128,
		128, 128
	};
	
	int *t_strategy; // t_strategy表示tiling策略
	t_strategy = (int*) malloc(BATCH * sizeof(int));

	int t;	
	for (t=0; t<6; ++t){
		TLP = 0;
		for (int j=0; j<BATCH; ++j)
			TLP += (M[j]/tile_size[t][0])*(N[j]/tile_size[t][1])*256;
		
		if (TLP < TLP_thres)
			break;
	}

	for (int j=0; j<BATCH; ++j){
	
		t_strategy[j] = 0;
		t = (t==6?5:t);

		if (tile_size[t][0] <= M[j] && tile_size[t][1] <= N[j])
			t_strategy[j] = t;
		else
		{
			for (int k=0; k<t; ++k)
			{
				if (tile_size[k][0] == M[j] && tile_size[k][1] <= N[j])
				{
					t_strategy[j] = k;
				}
			}
		}
		
		t_strategy[j] = 1;
	}

/*	
	//print the obtained tiling strategy
	for (int j=0; j<BATCH; ++j)
		printf("%d ", t_strategy[j]);
	printf("\n");
*/

	

	int *dev_T;  
	ErrChk(cudaMalloc((void**)&dev_T, BATCH*sizeof(int)));
	ErrChk(cudaMemcpy(dev_T, t_strategy, BATCH*sizeof(int), cudaMemcpyHostToDevice));


    dim3 block_size;   // block的形状影响性能吗
    block_size.x = 128;
    block_size.y = 1;
	block_size.z = 1;

    dim3 grid_size;
	
    grid_size.x = M[0] / tile_size[t_strategy[0]][0];
    grid_size.y = N[0] / tile_size[t_strategy[0]][1];
	grid_size.z = BATCH;
	for (int j=1; j<BATCH; ++j) // max
	{
		grid_size.x = (grid_size.x > M[j]/tile_size[t_strategy[j]][0])? (grid_size.x):(M[j]/tile_size[t_strategy[j]][0]);
		grid_size.y = (grid_size.y > N[j]/tile_size[t_strategy[j]][1])? (grid_size.y):(N[j]/tile_size[t_strategy[j]][1]);
	}

	//printf("%d %d %d\n", grid_size.x, grid_size.y, grid_size.z);

	//warm-up
	gemm<128><<<grid_size, block_size, sizeof(float)*4*64*8>>>(dev_M, dev_N, dev_K, dev_A, dev_B, dev_C);
	KernelErrChk();

	ErrChk(cudaEventCreate(&start));
	ErrChk(cudaEventRecord(start,0));

	for (int run = 0; run<N_RUNS; ++run)  // 10 times
	{
		// sizeof(float)*4*128*8  shared memory
		gemm<128><<<grid_size, block_size, sizeof(float)*4*64*8>>>(dev_M, dev_N, dev_K, dev_A, dev_B, dev_C);
		KernelErrChk();
	}

	ErrChk(cudaEventCreate(&stop));
	ErrChk(cudaEventRecord(stop,0));
	ErrChk(cudaEventSynchronize(stop));
	ErrChk(cudaEventElapsedTime(&elapsedTime, start,stop));

	time = elapsedTime/N_RUNS;
	time /= 1.0e3; //convert time unit from millisecond to second
	gflops_per_sec   = gflops / time;
	printf("%f\n", gflops_per_sec);

	for (int i=0; i<BATCH; ++i){
		ErrChk(cudaFree(A[i]));		
		ErrChk(cudaFree(B[i]));		
		ErrChk(cudaFree(C[i]));		
	}

	free(M);
	free(N);
	free(K);
	free(A);
	free(B);
	free(C);
	free(t_strategy);

	ErrChk(cudaFree(dev_M));		
	ErrChk(cudaFree(dev_N));		
	ErrChk(cudaFree(dev_K));		
	ErrChk(cudaFree(dev_T));		

	ErrChk(cudaFree(dev_A));		
	ErrChk(cudaFree(dev_B));		
	ErrChk(cudaFree(dev_C));		

	return 0;
}