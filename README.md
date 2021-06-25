
#include <stdio.h>
#include <cublas_v2.h>

#define THREADS_NUM 1024
// 65536/4 = 16383 = 1024x16, 64*1024 = 65536
#define L1_SIZE (65536/4)
#define WARP_SIZE 32

__global__ void l1_bw(uint32_t *startClk, uint32_t *stopClk, float *dsink, uint32_t *posArray)
{
	// Thread index
	uint32_t tid = threadIdx.x;
	
	// Side-effect variable, intended to avoid compiler elimination of this code
	float sink = 0;
	
	// Warm up the L1 cache by populating it
	for (uint32_t i = tid; i < L1_SIZE; i += THREADS_NUM)  // THREADS_NUM: block的线程数
	{
		float* ptr = posArray + i;
		asm volatile ("{\t\n"
					  ".reg .f32 data;\n\t"
					  "ld.global.ca.f32 data, [%1];\n\t"
					  "add.f32 %0, data, %0;\n\t"
					  "}" : "+f"(sink) : "l"(ptr) : "memory"
		);
	}
	
	// Synchronize all threads
	asm volatile ("bar.sync 0;");
	
	// Start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// Load data from L1 cache, accumulate
	for (uint32_t i = 0; i < L1_SIZE; i += THREADS_NUM) 
	{
		float* ptr = posArray + i;  // 所有的线程都是从0开始的
		
		// every warp loads all data in l1 cache: 一次内循环，一个warp读取THREADS_NUM个数
		for (uint32_t j = 0; j < THREADS_NUM; j += WARP_SIZE) 
		{
			uint32_t offset = (tid + j) % THREADS_NUM;
			// 计时中的add部分怎么理解，有overlap？
			// 这里的计时又怎么转化为bandwidth？(L1_CACHE_SIZE*4)/(32*latency) 4是float的大小   
			// 理论带宽的计算：DGEMM？
			asm volatile ("{\t\n"
						  ".reg .f32 data;\n\t"
						  "ld.global.ca.f32 data, [%1];\n\t"
						  "add.f64 %0, data, %0;\n\t"
						  "}" : "+f"(sink) : "l"(ptr+offset) : "memory"
			);
		}
	}
	
	// Synchronize all threads
	asm volatile ("bar.sync 0;");
	
	// Stop timing
	uint32_t stop = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
	
	// Write time and data back to memory
	startClk[tid] = start;
	stopClk[tid] = stop;
	dsink[tid] = sink;
}


int main(int argc, char* argv[]) 
{   
   dim3 gridDim2;
   dim3 blockDim2;
   
   gridDim2.x = 1; gridDim2.y = 1;  gridDim2.z = 1;
   blockDim2.x = THREADS_NUM; blockDim2.y = 1; blockDim2.z = 1; 
   
   uint32_t *startClk;
   uint32_t *stopClk;
   float *dsink;
   uint32_t *posArray;
   
   uint32_t* time_host_st;
   uint32_t* time_host_ed;
   
   time_host_st = (uint32_t*)malloc(THREADS_NUM * sizeof(uint32_t));
   time_host_ed = (uint32_t*)malloc(THREADS_NUM * sizeof(uint32_t));

   cudaErrCheck(cudaMalloc((void**)&startClk, THREADS_NUM * sizeof(uint32_t)));
   cudaErrCheck(cudaMalloc((void**)&stopClk, THREADS_NUM * sizeof(uint32_t)));    
   cudaErrCheck(cudaMalloc((void**)&dsink, THREADS_NUM * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&posArray, L1_SIZE * sizeof(uint32_t))); 
   
   l1_bw<<<gridDim2, blockDim2>>>(startClk, stopClk, dsink, posArray);
   

   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(time_host_st, startClk, THREADS_NUM * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(time_host_ed, stopClk, THREADS_NUM * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   
   // compute bandwidth
   double bandwidth = 0.0;
   for (int i = 0; i < THREADS_NUM; i++)
   {
		bandwidth += (L1_SIZE*4.0)/(32.0*(time_host_ed[i] - time_host_st[i]));
   }
   printf("\nBandWidth = %f\n", bandwidth/THREADS_NUM);
   
   cudaErrCheck(cudaFree(startClk));
   cudaErrCheck(cudaFree(stopClk));
   cudaErrCheck(cudaFree(dsink));
   cudaErrCheck(cudaFree(posArray));
   
   cudaErrCheck(cudaDeviceReset());
   
   return 0;
}

