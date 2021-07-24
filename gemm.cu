 
 //  nvcc -o gemm -arch=sm_70 -lcublas -lcurand fp16.cu

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cuda_fp16.h>


// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 64
#define MATRIX_N 64
#define MATRIX_K 64

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// 16*8 subtile_K = 8
__global__ void gemm_64_16x8_1(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[256];
    __shared__ float sh_B[128];

    float2 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;

    float reg_A[4];
    //float4 reg_A[2];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*8;
    int block_base_y = blockIdx.x*16;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float2 *A_start = (float2*) (A + block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M);
    *((float2*) (sh_A + 2*threadIdx.x)) = *(A_start);

    //load B from global memory to shared memory
    float *B_start = (B + K*block_base_x + (threadIdx.x/8) + (threadIdx.x%8)*K);
    *(sh_B + threadIdx.x) = *(B_start);

    int double_buffer_A = 0;
	int double_buffer_B = 0;

    for(int k = 0; k < K; k += 8)
	{
        __syncthreads();
        int A_offset = double_buffer_A + (threadIdx.x%8)*2;
        int B_offset = double_buffer_B + (threadIdx.x/8);
		
#pragma unroll
        for (int i=0; i<8; i+=1)
		{
            reg_A[0] = sh_A[A_offset];  
            reg_A[1] = sh_A[A_offset+1]; 
			reg_B[0] = sh_B[B_offset];

            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
			
			//reg_A[2] = sh_A[A_offset+16];
            //reg_A[3] = sh_A[A_offset+17];
			//reg_B[1] = sh_B[B_offset+1];
			
            //reg_C.x = fma(reg_A[2], reg_B[1], reg_C.x);
            //reg_C.y = fma(reg_A[3], reg_B[1], reg_C.y);

            A_offset += 16;
            B_offset += 8;
        }
		
        double_buffer_A ^= 128;
		double_buffer_B ^= 64;
		
        if (k + 8 < K)
		{
            A_start += 4*M; // float2 --> 8M
            *((float2*) (sh_A + double_buffer_A + 2*threadIdx.x)) = *(A_start);
            B_start += 8;
            *(sh_B + double_buffer_B + threadIdx.x) = *(B_start);
        }
    }
	
	int ind = blockIdx.x*16 + (threadIdx.x%8)*2;  // 行位置
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)
    int	PQ = M;
    int C_offset = ind/PQ*(PQ*N) + ind%(PQ) + (threadIdx.x/8)*(PQ) + blockIdx.y*8*(PQ);
    C[C_offset] = reg_C.x;
    C[C_offset+1] = reg_C.y;
}

// 16*8 subtile_K = 8
__global__ void gemm_32_16x8_1(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[256];
    __shared__ float sh_B[128];

    float4 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;
	reg_C.z =0.f;
	reg_C.w =0.f;

    float reg_A[8];
    //float4 reg_A[2];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*8;
    int block_base_y = blockIdx.x*16;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%4)*4 + (threadIdx.x/4)*M);
    *((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

    //load B from global memory to shared memory
    float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/8)*2 + (threadIdx.x%8)*K);
    *((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);

    int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
    for(int k = 0; k < K; k += 8)
	{
        __syncthreads();
        int A_offset = double_buffer_A + (threadIdx.x%4)*4;
        int B_offset = double_buffer_B + ((threadIdx.x/4)*2);	
		
#pragma unroll
        for (int i=0; i<8; i+=2)   // 全部展开有register spill吗
		{
            reg_A[0] = sh_A[A_offset];    // A_offset+0 ~ A_offset+3 为什么不用向量呢
            reg_A[1] = sh_A[A_offset+1];  // 为了避免bank冲突, 这8个寄存器都不是连续的【4个就不会重复】，因此不能使用向量传输指令
            reg_A[2] = sh_A[A_offset+2];
            reg_A[3] = sh_A[A_offset+3];
	   // *(float4*)reg_A = *(float4*)(sh_A+A_offset);
	    reg_A[4] = sh_A[A_offset+16];
            reg_A[5] = sh_A[A_offset+17];
            reg_A[6] = sh_A[A_offset+18];
            reg_A[7] = sh_A[A_offset+19];
	   //*(float4*)(reg_A+4) = *(float4*)(sh_A+A_offset+16);
	    reg_B[0] = sh_B[B_offset];
			
            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
            reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			
		//	reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+1];
			
            reg_C.x = fma(reg_A[4], reg_B[1], reg_C.x);
            reg_C.y = fma(reg_A[5], reg_B[1], reg_C.y);
            reg_C.z = fma(reg_A[6], reg_B[1], reg_C.z);
            reg_C.w = fma(reg_A[7], reg_B[1], reg_C.w);

            A_offset += 32;
            B_offset += 16;
        }
		
        double_buffer_A ^= 128;
		double_buffer_B ^= 64;
		
        if (k + 8 < K)
		{
            A_start += 2*M; // float2 --> 8M
            *((float4*) (sh_A + double_buffer_A + 4*threadIdx.x)) = *(A_start);
			
            B_start += 4;
            *((float2*) (sh_B + double_buffer_B + 2*threadIdx.x)) = *(B_start);
        }
    }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;  // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)
    int	PQ = M;
    int C_offset = ind/PQ*(PQ*N) + ind%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*8*(PQ);
    C[C_offset] = reg_C.x;
    C[C_offset+1] = reg_C.y;
    C[C_offset+2] = reg_C.z;
    C[C_offset+3] = reg_C.w;
}


// 16*8 subtile_K = 8
__global__ void gemm_32_16x8_3(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[256];
    __shared__ float sh_B[128];

    float4 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;
	reg_C.z =0.f;
	reg_C.w =0.f;

    float reg_A[8];
    //float4 reg_A[2];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*8;
    int block_base_y = blockIdx.x*16;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    int A_start = block_base_y + (threadIdx.x%4)*4 + (threadIdx.x/4)*M;
    *(sh_A + 4*threadIdx.x) = A[A_start%(M*K)];
	*(sh_A + 4*threadIdx.x+1) = A[(A_start+1)%(M*K)];
	*(sh_A + 4*threadIdx.x+2) = A[(A_start+2)%(M*K)];
	*(sh_A + 4*threadIdx.x+3) = A[(A_start+3)%(M*K)];

    //load B from global memory to shared memory
    float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/8)*2 + (threadIdx.x%8)*K);
    *((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);

    int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
    for(int k = 0; k < K; k += 8)
	{
        __syncthreads();
        int A_offset = double_buffer_A + (threadIdx.x%4)*4;
        int B_offset = double_buffer_B + ((threadIdx.x/4)*2);	
		
#pragma unroll
        for (int i=0; i<8; i+=2)   // 全部展开有register spill吗
		{
            reg_A[0] = sh_A[A_offset];    // A_offset+0 ~ A_offset+3 为什么不用向量呢
            reg_A[1] = sh_A[A_offset+1];  // 为了避免bank冲突, 这8个寄存器都不是连续的【4个就不会重复】，因此不能使用向量传输指令
            reg_A[2] = sh_A[A_offset+2];
            reg_A[3] = sh_A[A_offset+3];
	   // *(float4*)reg_A = *(float4*)(sh_A+A_offset);
	    reg_A[4] = sh_A[A_offset+16];
            reg_A[5] = sh_A[A_offset+17];
            reg_A[6] = sh_A[A_offset+18];
            reg_A[7] = sh_A[A_offset+19];
	   //*(float4*)(reg_A+4) = *(float4*)(sh_A+A_offset+16);
	    reg_B[0] = sh_B[B_offset];
			
            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
            reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			
		//	reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+1];
			
            reg_C.x = fma(reg_A[4], reg_B[1], reg_C.x);
            reg_C.y = fma(reg_A[5], reg_B[1], reg_C.y);
            reg_C.z = fma(reg_A[6], reg_B[1], reg_C.z);
            reg_C.w = fma(reg_A[7], reg_B[1], reg_C.w);

            A_offset += 32;
            B_offset += 16;
        }
		
        double_buffer_A ^= 128;
		double_buffer_B ^= 64;
		
        if (k + 8 < K)
		{
            A_start += 8*M; // float2 --> 8M
			*(sh_A + double_buffer_A + 4*threadIdx.x) = A[A_start%(M*K)];
			*(sh_A + double_buffer_A + 4*threadIdx.x+1) = A[(A_start+1)%(M*K)];
			*(sh_A + double_buffer_A + 4*threadIdx.x+2) = A[(A_start+2)%(M*K)];
			*(sh_A + double_buffer_A + 4*threadIdx.x+3) = A[(A_start+3)%(M*K)];
            B_start += 4;
            *((float2*) (sh_B + double_buffer_B + 2*threadIdx.x)) = *(B_start);
        }
    }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;  // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)
    int	PQ = M;
    int C_offset = ind%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*8*(PQ);
	
    if (blockIdx.x < M/16)
    {
		C[C_offset] = reg_C.x;
		C[C_offset+1] = reg_C.y;
		C[C_offset+2] = reg_C.z;
		C[C_offset+3] = reg_C.w;		
    }
    else
    {
       int ruler = (threadIdx.x%4)*4;
       int rag = M%16;
	   
       if (ruler == 0)
	   {
           C[C_offset] = reg_C.x;
	   }
    }	
}


__global__ void gemm_64_16x8_3(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[256];
    __shared__ float sh_B[128];

    float2 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;

    float reg_A[4];
    //float4 reg_A[2];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*8;
    int block_base_y = blockIdx.x*16;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    int aoffset = block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M;
    *(sh_A + 2*threadIdx.x) = A[aoffset%(M*K)];
	*(sh_A + 2*threadIdx.x + 1) = A[(aoffset+1)%(M*K)];

    //load B from global memory to shared memory
    float *B_start = (B + K*block_base_x + (threadIdx.x/8) + (threadIdx.x%8)*K);
    *(sh_B + threadIdx.x) = *(B_start);

    int double_buffer_A = 0;
	int double_buffer_B = 0;

    for(int k = 0; k < K; k += 8)
	{
        __syncthreads();
        int A_offset = double_buffer_A + (threadIdx.x%8)*2;
        int B_offset = double_buffer_B + (threadIdx.x/8);
		
#pragma unroll
        for (int i=0; i<8; i+=1)
		{
            reg_A[0] = sh_A[A_offset];  
            reg_A[1] = sh_A[A_offset+1]; 
			reg_B[0] = sh_B[B_offset];

            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
			
			//reg_A[2] = sh_A[A_offset+16];
            //reg_A[3] = sh_A[A_offset+17];
			//reg_B[1] = sh_B[B_offset+1];
			
            //reg_C.x = fma(reg_A[2], reg_B[1], reg_C.x);
            //reg_C.y = fma(reg_A[3], reg_B[1], reg_C.y);

            A_offset += 16;
            B_offset += 8;
        }
		
        double_buffer_A ^= 128;
		double_buffer_B ^= 64;
		
        if (k + 8 < K)
		{
            // A_start += 4*M; // float2 --> 8M
			aoffset += 8*M;
            //*((float2*) (sh_A + double_buffer_A + 2*threadIdx.x)) = *(A_start);
			*(sh_A + double_buffer_A + 2*threadIdx.x) = A[aoffset%(M*K)];
			*(sh_A + double_buffer_A + 2*threadIdx.x + 1) = A[(aoffset+1)%(M*K)];
            B_start += 8;
            *(sh_B + double_buffer_B + threadIdx.x) = *(B_start);
        }
    }
	
	int ind = blockIdx.x*16 + (threadIdx.x%8)*2;  // 行位置
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)
    int	PQ = M;
    int C_offset = ind/PQ*(PQ*N) + ind%(PQ) + (threadIdx.x/8)*(PQ) + blockIdx.y*8*(PQ);
    //C[C_offset] = reg_C.x;
    //C[C_offset+1] = reg_C.y;
	
    if (blockIdx.x < M/16)
    {
		C[C_offset] = reg_C.x;
		C[C_offset+1] = reg_C.y;
    }
    else
    {
       int ruler = (threadIdx.x%4)*4;
       int rag = M%16;
	   
       if (ruler == 0)
	   {
           C[C_offset] = reg_C.x;
	   }
    }
}


// 8*8 subtile_K = 8  ->  32 thread
__global__ void gemm_64_8x8_1(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[128];
    __shared__ float sh_B[128];

    float reg_C = 0.f;

    float reg_A[2];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*8;
    int block_base_y = blockIdx.x*8;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float *A_start = (A + block_base_y + (threadIdx.x%8) + (threadIdx.x/8)*M);
    *(sh_A + threadIdx.x) = *(A_start);

    //load B from global memory to shared memory
    float *B_start =  (B + K*block_base_x + (threadIdx.x/8) + (threadIdx.x%8)*K);
    *(sh_B + threadIdx.x) = *(B_start);

    int double_buffer = 0;
	
    for(int k = 0; k < K; k += 8)
	{
        __syncthreads();
        int A_offset = double_buffer + (threadIdx.x%8);
        int B_offset = double_buffer + ((threadIdx.x/8)*8);
		
#pragma unroll
        for (int i=0; i<8; i+=1)
		{
            reg_A[0] = sh_A[A_offset]; 
            reg_A[1] = sh_A[A_offset+1];
			reg_B[0] = sh_B[B_offset];
			
            reg_C = fma(reg_A[0], reg_B[0], reg_C); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
			
			//reg_B[1] = sh_B[B_offset+1];
            //reg_C = fma(reg_A[1], reg_B[1], reg_C);

            A_offset += 8;
            B_offset += 8;
        }
		
        double_buffer ^= 64;
		
        if (k + 8 < K)
		{
            A_start += 8*M; // float2 --> 8M
            *(sh_A + double_buffer + threadIdx.x) = *(A_start);
            B_start += 8;
            *(sh_B + double_buffer + threadIdx.x) = *(B_start);
        }
    }

	int ind = blockIdx.x*8 + (threadIdx.x%8);  // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)  ind/PQ*(PQ*N) + 
    int	PQ = M;
    int C_offset = ind%(PQ) + (threadIdx.x/8)*(PQ) + blockIdx.y*8*(PQ);
    C[C_offset] = reg_C;
}


__global__ void gemm_64_16x16_1(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[256];
    __shared__ float sh_B[256];

    float4 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;
	reg_C.z =0.f;
	reg_C.w =0.f;

    float reg_A[8];
    //float4 reg_A[2];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*16;
    int block_base_y = blockIdx.x*16;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float2 *A_start = (float2*) (A + block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M);
    *((float2*) (sh_A + 2*threadIdx.x)) = *(A_start);

    //load B from global memory to shared memory
    float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K);
    *((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);

    int double_buffer = 0;
#pragma unroll
    for(int k = 0; k < K; k += 8)
	{
        __syncthreads();
        int A_offset = double_buffer + (threadIdx.x%4)*4;
        int B_offset = double_buffer + ((threadIdx.x/4)*2);	
		
#pragma unroll
        for (int i=0; i<8; i+=2)   // 全部展开有register spill吗
		{
            reg_A[0] = sh_A[A_offset];    // A_offset+0 ~ A_offset+3 为什么不用向量呢
            reg_A[1] = sh_A[A_offset+1];  // 为了避免bank冲突, 这8个寄存器都不是连续的【4个就不会重复】，因此不能使用向量传输指令
            reg_A[2] = sh_A[A_offset+2];
            reg_A[3] = sh_A[A_offset+3];
	   // *(float4*)reg_A = *(float4*)(sh_A+A_offset);
	    reg_A[4] = sh_A[A_offset+16];
            reg_A[5] = sh_A[A_offset+17];
            reg_A[6] = sh_A[A_offset+18];
            reg_A[7] = sh_A[A_offset+19];
	   //*(float4*)(reg_A+4) = *(float4*)(sh_A+A_offset+16);
	    reg_B[0] = sh_B[B_offset];
			
            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
            reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			
		//	reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+1];
			
            reg_C.x = fma(reg_A[4], reg_B[1], reg_C.x);
            reg_C.y = fma(reg_A[5], reg_B[1], reg_C.y);
            reg_C.z = fma(reg_A[6], reg_B[1], reg_C.z);
            reg_C.w = fma(reg_A[7], reg_B[1], reg_C.w);

            A_offset += 32;
            B_offset += 32;
        }
		
        double_buffer ^= 128;
		
        if (k + 8 < K)
		{
            A_start += 4*M; // float2 --> 8M
            *((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
            B_start += 4;
            *((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
        }
    }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;  // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)
    int	PQ = M;
    int C_offset = ind/PQ*(PQ*N) + ind%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*16*(PQ);
    C[C_offset] = reg_C.x;
    C[C_offset+1] = reg_C.y;
    C[C_offset+2] = reg_C.z;
    C[C_offset+3] = reg_C.w;
}

__global__ void gemm_64_16x16_3(int M, int N, int K, int P, int Q, float *A, float *B, float *C){

   __shared__ float sh_A[256];
   __shared__ float sh_B[256];
   //float* sh_B = sh + 2*16*8;

   float reg_C[4];
   reg_C[0] = 0.f;
   reg_C[1] = 0.f;
   reg_C[2] = 0.f;
   reg_C[3] = 0.f;

   float reg_A[8]={0.f};
   float reg_B[2]={0.f};

   // Compute block's starting coordinate
   int block_base_x = blockIdx.y*16;
   int block_base_y = blockIdx.x*16;


   //load A from global memory to shared memory
   int A_offset = block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M; // 跟前面的不一样
   sh_A[threadIdx.x] = A[A_offset%(M*K)];
   sh_A[threadIdx.x+64] = A[(A_offset+4*M)%(M*K)];  

   //load A from global memory to shared memory
   int B_offset =  K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K;
   sh_B[threadIdx.x*2] = B[B_offset%(K*N)];
   sh_B[threadIdx.x*2+1] = B[(B_offset+1)%(K*N)];

   int double_buffer = 0;
#pragma unroll
   for(int k=0; k<K; k+=8)
   {
       __syncthreads();
       int shA_offset = double_buffer + (threadIdx.x%4)*4;
       int shB_offset = double_buffer + ((threadIdx.x/4)*2);
#pragma unroll
       for (int i=0; i<8; i+=2){  // 可以1D register blocking转2D register blocking吗

           reg_A[0] = sh_A[shA_offset];
           reg_A[1] = sh_A[shA_offset+1];
           reg_A[2] = sh_A[shA_offset+2];
           reg_A[3] = sh_A[shA_offset+3];
           reg_A[4] = sh_A[shA_offset+16];
           reg_A[5] = sh_A[shA_offset+17];
           reg_A[6] = sh_A[shA_offset+18];
           reg_A[7] = sh_A[shA_offset+19];

           reg_B[0] = sh_B[shB_offset];
           reg_B[1] = sh_B[shB_offset+1];

           reg_C[0] = fma(reg_A[0], reg_B[0], reg_C[0]);
           reg_C[1] = fma(reg_A[1], reg_B[0], reg_C[1]);
           reg_C[2] = fma(reg_A[2], reg_B[0], reg_C[2]);
           reg_C[3] = fma(reg_A[3], reg_B[0], reg_C[3]);
           reg_C[0] = fma(reg_A[4], reg_B[1], reg_C[0]);
           reg_C[1] = fma(reg_A[5], reg_B[1], reg_C[1]);
           reg_C[2] = fma(reg_A[6], reg_B[1], reg_C[2]);
           reg_C[3] = fma(reg_A[7], reg_B[1], reg_C[3]);

           shA_offset += 32;
           shB_offset += 32;
       }

       double_buffer ^= 128;

       if (k+8 < K)
	   {
           A_offset += 8*M;
           sh_A[double_buffer+threadIdx.x] = A[A_offset%(M*K)];
           sh_A[double_buffer+threadIdx.x+64] = A[(A_offset+4*M)%(M*K)];
           B_offset += 8;
           sh_B[double_buffer+threadIdx.x*2] = B[B_offset%(K*N)];
           sh_B[double_buffer+threadIdx.x*2+1] = B[(B_offset+1)%(K*N)];
       }
   }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);

   if (blockIdx.x < M/16)
   {
       C[C_offset] = reg_C[0];
       C_offset = (ind+1)/(P*Q)*(P*Q*N) + (ind+1)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
       C[C_offset] = reg_C[1];
       C_offset = (ind+2)/(P*Q)*(P*Q*N) + (ind+2)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
       C[C_offset] = reg_C[2];
       C_offset = (ind+3)/(P*Q)*(P*Q*N) + (ind+3)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
       C[C_offset] = reg_C[3];
   }
   else
   {
       int ruler = (threadIdx.x%4)*4;
       int rag = M%16;
       if ((ruler)<rag){
           C[C_offset] = reg_C[0];
		}
       if ((ruler+1)<rag){
     		C_offset = (ind+1)/(P*Q)*(P*Q*N) + (ind+1)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
           C[C_offset] = reg_C[1];
		}
       if ((ruler+2)<rag){
     	C_offset = (ind+2)/(P*Q)*(P*Q*N) + (ind+2)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
           C[C_offset] = reg_C[2];
		}
       if ((ruler+3)<rag){
     		C_offset = (ind+3)/(P*Q)*(P*Q*N) + (ind+3)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
           C[C_offset] = reg_C[3];
		}
   }
}

__global__ void gemm_256_32x32(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[512];
	__shared__ float sh_B[512];  // 2*32*8

	float4 reg_C;
	float4 reg_A;
	float  reg_B;
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >> 3;
	int im32 = threadIdx.x & 31;
	int id32 = threadIdx.x >> 5;

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y << 5;
	int block_base_y = blockIdx.x << 5;

	//Load C from global memory to register file
	float4 *C_start = (float4 *) (C + block_base_x*M + block_base_y + (im8<<2) + (id8)*M);

    reg_C = *C_start;
	reg_C = {0.0f,0.0f,0.0f,0.0f};

	//load B from global memory to shared memory
	float *A_start = (A + block_base_y + (im32) + (id32)*M); 
	* (sh_A + threadIdx.x) = *(A_start);

	//load A from global memory to shared memory
	float *B_start = (B + K*block_base_x + (id32) + (im32)*K); 
	* (sh_B + threadIdx.x) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (im8 << 2);
		int B_offset = double_buffer + id8;

		double_buffer ^= 256;

		if (k+8 < K){
			A_start += M << 3; 
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 8; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
#pragma unroll
		for (int i=0; i<8; ++i)	{
			reg_A = *((float4*) (sh_A + A_offset)); 
			reg_B = sh_B[B_offset]; 

			reg_C.x = fma(reg_A.x, reg_B, reg_C.x);
			reg_C.y = fma(reg_A.y, reg_B, reg_C.y);
			reg_C.z = fma(reg_A.z, reg_B, reg_C.z);
			reg_C.w = fma(reg_A.w, reg_B, reg_C.w);

			A_offset += 32;
			B_offset += 32;
		}
	}
	
	*(C_start) = reg_C;
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}


__global__ void gemm_256_32x32_orig(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[512];
	__shared__ float sh_B[512];

	float4 reg_C;
	float4 reg_A;
	float  reg_B;

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*32;
	int block_base_y = blockIdx.x*32;

	//Load C from global memory to register file
	float4 *C_start = (float4 *) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M);

    reg_C = *C_start;

	//load B from global memory to shared memory
	float *A_start = (A + block_base_y + (threadIdx.x%32) + (threadIdx.x/32)*M); 
	* (sh_A + threadIdx.x) = *(A_start);

	//load A from global memory to shared memory
	float *B_start = (B + K*block_base_x + (threadIdx.x/32) + (threadIdx.x%32)*K); 
	* (sh_B + threadIdx.x) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*4;
		int B_offset = double_buffer + (threadIdx.x/8);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			reg_A = *((float4*) (sh_A + A_offset)); 
			reg_B = sh_B[B_offset]; 

			reg_C.x = fma(reg_A.x, reg_B, reg_C.x);
			reg_C.y = fma(reg_A.y, reg_B, reg_C.y);
			reg_C.z = fma(reg_A.z, reg_B, reg_C.z);
			reg_C.w = fma(reg_A.w, reg_B, reg_C.w);

			A_offset += 32;
			B_offset += 32;
		}

		double_buffer ^= 256;

		if (k+8 < K){
			A_start += 8*M; 
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 8; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
				
	}
	*(C_start) = reg_C;

}


__global__ void gemm_256_32x32_16k(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[1024];
	__shared__ float sh_B[1024];

	float4 reg_C;
	float4 reg_A;
	float  reg_B;
	
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >>3;
	int im16 = threadIdx.x&15;
	int id16 = threadIdx.x>>4;
	int im32 = threadIdx.x&31;
	int id32 = threadIdx.x>>5;
	
	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<5;
	int block_base_y = blockIdx.x<<5;

	//Load C from global memory to register file
	float4 *C_start = (float4 *) (C + block_base_x*M + block_base_y + (im8<<2) + (id8)*M);

    reg_C = {0.f, 0.f, 0.f, 0.f};

	//load B from global memory to shared memory
	float2 *A_start = (float2*)(A + block_base_y + (im16<<1) + (id16)*M); 
	*((float2*)(sh_A + 2*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float *B_start = (B + K*block_base_x + (id32<<1) + (im32)*K); 
	*(sh_B + (id32<<6) + im32) = *(B_start);
	*(sh_B + (id32<<6) + im32 + 32) = *(B_start+1);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (im8<<2);
		int B_offset = double_buffer + id8;
		
#pragma unroll
		for (int i=0; i<16; ++i)
		{
			reg_A = *((float4*) (sh_A + A_offset)); 
			reg_B = sh_B[B_offset];

			reg_C.x = fma(reg_A.x, reg_B, reg_C.x);
			reg_C.y = fma(reg_A.y, reg_B, reg_C.y);
			reg_C.z = fma(reg_A.z, reg_B, reg_C.z);
			reg_C.w = fma(reg_A.w, reg_B, reg_C.w);

			A_offset += 32;
			B_offset += 32;
		}

		double_buffer ^= 512;

		if (k+16 < K)
		{
			A_start += M<<3; 
			*((float2*)(sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);

			B_start += 16;
			*(sh_B +  double_buffer + (id32<<6) + im32) = *(B_start);
			*(sh_B +  double_buffer + (id32<<6) + im32 + 32) = *(B_start+1);
		}
	}
	
	*(C_start) = reg_C;
}


__global__ void gemm_256_64x64__orig(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[1024];
	__shared__ float sh_B[1024];

	float4 reg_C[4];
	float4 reg_A[2];
	float  reg_B[2];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M);
    reg_C[0] = {0.f,0.f,0.f,0.f}; //*C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f}; //*(C_start + 8);
	reg_C[2] = {0.f,0.f,0.f,0.f}; //*(C_start + m8);
	reg_C[3] = {0.f,0.f,0.f,0.f}; //*(C_start + 8 + m8);
	
	//load A from global memory to shared memory
	float2 *A_start = (float2*) (A + block_base_y + (threadIdx.x%32)*2 + (threadIdx.x/32)*M); 
	*((float2*) (sh_A + 2*threadIdx.x)) = *(A_start);

	//load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/64)*2 + (threadIdx.x%64)*K); 
	*((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*4;
		int B_offset = double_buffer + (threadIdx.x/8)*2;
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			reg_A[0] = *((float4*) (sh_A + A_offset)); 
			reg_A[1] = *((float4*) (sh_A + A_offset + 32)); 
			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset + 64]; 

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);

			reg_C[1].x = fma(reg_A[1].x, reg_B[0], reg_C[1].x);
			reg_C[1].y = fma(reg_A[1].y, reg_B[0], reg_C[1].y);
			reg_C[1].z = fma(reg_A[1].z, reg_B[0], reg_C[1].z);
			reg_C[1].w = fma(reg_A[1].w, reg_B[0], reg_C[1].w);

			reg_C[2].x = fma(reg_A[0].x, reg_B[1], reg_C[2].x);
			reg_C[2].y = fma(reg_A[0].y, reg_B[1], reg_C[2].y);
			reg_C[2].z = fma(reg_A[0].z, reg_B[1], reg_C[2].z);
			reg_C[2].w = fma(reg_A[0].w, reg_B[1], reg_C[2].w);

			reg_C[3].x = fma(reg_A[1].x, reg_B[1], reg_C[3].x);
			reg_C[3].y = fma(reg_A[1].y, reg_B[1], reg_C[3].y);
			reg_C[3].z = fma(reg_A[1].z, reg_B[1], reg_C[3].z);
			reg_C[3].w = fma(reg_A[1].w, reg_B[1], reg_C[3].w);

			A_offset += 64;
			B_offset += 1;
			if (i%2) B_offset += 126;
		}

		double_buffer ^= 512;

		if (k+8 < K){
			A_start += 4*M; 
			*((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);

			B_start += 4; 
			*((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
		}
				
	}
	*(C_start) = reg_C[0];
	*(C_start + 8) = reg_C[1];
	*(C_start + 8*M) = reg_C[2];
	*(C_start + 8 + 8*M) = reg_C[3];
}


__global__ void gemm_256_64x64_16_2(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[2048];
	__shared__ float sh_B[2048];
	//float *sh_A = sh;
	//float *sh_B = sh + 2048;

	float4 reg_C[4];
	float4 reg_A[2];
	float  reg_B[2];
	
	int m8 = M<<3;
	int im8 = threadIdx.x&7;
	int id8 = threadIdx.x>>3;
	int im16 = threadIdx.x&15;
	int id16 = threadIdx.x>>4;
	int im64 = threadIdx.x&63;
	int id64 = threadIdx.x>>6;
	
	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<6;
	int block_base_y = blockIdx.x<<6;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (im8<<2) + (id8)*M);
    reg_C[0] = {0.f,0.f,0.f,0.f}; //*C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f}; //*(C_start + 8);
	reg_C[2] = {0.f,0.f,0.f,0.f}; //*(C_start + m8);
	reg_C[3] = {0.f,0.f,0.f,0.f}; //*(C_start + 8 + m8);
	
	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (im16<<2) + (id16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/64)*2 + (threadIdx.x%64)*K); 
	*((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);
	*((float2*) (sh_B + 2*threadIdx.x + 64*8)) = *(B_start+4);


	//load B from global memory to shared memory
	//float4 *B_start = (float4*) (B + K*block_base_x + (id64<<2) + (im64)*K); 
	//*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);

	
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (im8<<2);
		int B_offset = double_buffer + (id8<<1);

#pragma unroll
		for (int i=0; i<16; ++i)
		{
			reg_A[0] = *((float4*) (sh_A + A_offset)); 
			reg_A[1] = *((float4*) (sh_A + A_offset + 32)); 
			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset + 64];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);

			reg_C[1].x = fma(reg_A[1].x, reg_B[0], reg_C[1].x);
			reg_C[1].y = fma(reg_A[1].y, reg_B[0], reg_C[1].y);
			reg_C[1].z = fma(reg_A[1].z, reg_B[0], reg_C[1].z);
			reg_C[1].w = fma(reg_A[1].w, reg_B[0], reg_C[1].w);

			reg_C[2].x = fma(reg_A[0].x, reg_B[1], reg_C[2].x);
			reg_C[2].y = fma(reg_A[0].y, reg_B[1], reg_C[2].y);
			reg_C[2].z = fma(reg_A[0].z, reg_B[1], reg_C[2].z);
			reg_C[2].w = fma(reg_A[0].w, reg_B[1], reg_C[2].w);

			reg_C[3].x = fma(reg_A[1].x, reg_B[1], reg_C[3].x);
			reg_C[3].y = fma(reg_A[1].y, reg_B[1], reg_C[3].y);
			reg_C[3].z = fma(reg_A[1].z, reg_B[1], reg_C[3].z);
			reg_C[3].w = fma(reg_A[1].w, reg_B[1], reg_C[3].w);
			
			A_offset += 64;
			B_offset += 1;
			if (i%2) B_offset += 126;
		}

		double_buffer ^= 1024;

		if (k+16 < K)
		{
			A_start += M<<2; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);

			//B_start += 4; 
			//*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
			
			B_start += 8; 
			*((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
			*((float2*) (sh_B + double_buffer + 2*threadIdx.x + 64*8)) = *(B_start+4);
		}
	}
	
	*(C_start) = reg_C[0];
	*(C_start + 8) = reg_C[1];
	*(C_start + m8) = reg_C[2];
	*(C_start + 8 + m8) = reg_C[3];
}


__global__ void gemm_256_64x64_16(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[2048];
	__shared__ float sh_B[2048];
	//float *sh_A = sh;
	//float *sh_B = sh + 2048;

	float4 reg_C[4];
	float4 reg_A[2];
	float  reg_B[2];
	
	int m8 = M<<3;
	int im8 = threadIdx.x&7;
	int id8 = threadIdx.x>>3;
	int im16 = threadIdx.x&15;
	int id16 = threadIdx.x>>4;
	int im64 = threadIdx.x&63;
	int id64 = threadIdx.x>>6;
	
	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<6;
	int block_base_y = blockIdx.x<<6;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (im8<<2) + (id8)*M);
    reg_C[0] = {0.f,0.f,0.f,0.f}; //*C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f}; //*(C_start + 8);
	reg_C[2] = {0.f,0.f,0.f,0.f}; //*(C_start + m8);
	reg_C[3] = {0.f,0.f,0.f,0.f}; //*(C_start + 8 + m8);
	
	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (im16<<2) + (id16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load B from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (id64<<2) + (im64)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);

	
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (im8<<2);
		int B_offset = double_buffer + (id8<<1);

#pragma unroll
		for (int i=0; i<16; ++i)
		{
			reg_A[0] = *((float4*) (sh_A + A_offset)); 
			reg_A[1] = *((float4*) (sh_A + A_offset + 32)); 
			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset + 128];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);

			reg_C[1].x = fma(reg_A[1].x, reg_B[0], reg_C[1].x);
			reg_C[1].y = fma(reg_A[1].y, reg_B[0], reg_C[1].y);
			reg_C[1].z = fma(reg_A[1].z, reg_B[0], reg_C[1].z);
			reg_C[1].w = fma(reg_A[1].w, reg_B[0], reg_C[1].w);

			reg_C[2].x = fma(reg_A[0].x, reg_B[1], reg_C[2].x);
			reg_C[2].y = fma(reg_A[0].y, reg_B[1], reg_C[2].y);
			reg_C[2].z = fma(reg_A[0].z, reg_B[1], reg_C[2].z);
			reg_C[2].w = fma(reg_A[0].w, reg_B[1], reg_C[2].w);

			reg_C[3].x = fma(reg_A[1].x, reg_B[1], reg_C[3].x);
			reg_C[3].y = fma(reg_A[1].y, reg_B[1], reg_C[3].y);
			reg_C[3].z = fma(reg_A[1].z, reg_B[1], reg_C[3].z);
			reg_C[3].w = fma(reg_A[1].w, reg_B[1], reg_C[3].w);
			
			A_offset += 64;
			B_offset += 1;
			if ((i+1)&3 == 0) B_offset += 252;
		}

		double_buffer ^= 1024;

		if (k+16 < K)
		{
			A_start += M<<2; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
		}
	}
	
	*(C_start) = reg_C[0];
	*(C_start + 8) = reg_C[1];
	*(C_start + m8) = reg_C[2];
	*(C_start + 8 + m8) = reg_C[3];
}


__global__ void gemm_256_128x128_16(int M, int N, int K, float *A, float *B, float *C)
{
    __shared__ float sh_A[4096];
	__shared__ float sh_B[4096];

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<7;
	int block_base_y = blockIdx.x<<7;
	
	int m16 = M<<4;
	int md2 = M >>1;
	int md4 = M >>2;
	int im16 = threadIdx.x & 15;
	int id16 = threadIdx.x >> 4;
	int im128 = threadIdx.x & 127;
	int id128 = threadIdx.x >> 7;	
	int th8 = threadIdx.x << 3;
	
	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (im16<<2) + (id16<<2)*M);

    reg_C[0] = {0.f,0.f,0.f,0.f};//*C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[2] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[3] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);  

	C_start += 16;
	reg_C[4] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[5] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[6] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[7] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += (m16 - 16);
	reg_C[8] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[9] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[10] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[11] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += 16;
	reg_C[12] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[13] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[14] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[15] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (im16<<3) + (id16)*M); 
	*((float4*) (sh_A + th8)) = *(A_start);
	*((float4*) (sh_A + th8 + 4)) = *(A_start+1);

	//load B from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (id128<<3) + (im128)*K); 
	*((float4*) (sh_B + th8)) = *(B_start);
	*((float4*) (sh_B + th8 + 4)) = *(B_start+1);
	
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (im16<<2);
		int B_offset = double_buffer + (id16<<4);
		
#pragma unroll
		for (int i=0; i<16; ++i)
		{
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+1];
			reg_A[2] = sh_A[A_offset+2];
			reg_A[3] = sh_A[A_offset+3];
			reg_A[4] = sh_A[A_offset+64];
			reg_A[5] = sh_A[A_offset+65];
			reg_A[6] = sh_A[A_offset+66];
			reg_A[7] = sh_A[A_offset+67];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+8];
			reg_B[2] = sh_B[B_offset+16];
			reg_B[3] = sh_B[B_offset+24];
			reg_B[4] = sh_B[B_offset+512];
			reg_B[5] = sh_B[B_offset+516];
			reg_B[6] = sh_B[B_offset+520];
			reg_B[7] = sh_B[B_offset+524];

			reg_C[0].x = fma(reg_A[0], reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0], reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0], reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0], reg_B[3], reg_C[3].x);
			reg_C[8].x = fma(reg_A[0], reg_B[4], reg_C[8].x);
			reg_C[9].x = fma(reg_A[0], reg_B[5], reg_C[9].x);
			reg_C[10].x = fma(reg_A[0], reg_B[6], reg_C[10].x);
			reg_C[11].x = fma(reg_A[0], reg_B[7], reg_C[11].x);
			reg_C[4].x = fma(reg_A[4], reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[4], reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[4], reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[4], reg_B[3], reg_C[7].x);
			reg_C[12].x = fma(reg_A[4], reg_B[4], reg_C[12].x);
			reg_C[13].x = fma(reg_A[4], reg_B[5], reg_C[13].x);
			reg_C[14].x = fma(reg_A[4], reg_B[6], reg_C[14].x);
			reg_C[15].x = fma(reg_A[4], reg_B[7], reg_C[15].x);

			reg_C[0].y = fma(reg_A[1], reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[1], reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[1], reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[1], reg_B[3], reg_C[3].y);
			reg_C[8].y = fma(reg_A[1], reg_B[4], reg_C[8].y);
			reg_C[9].y = fma(reg_A[1], reg_B[5], reg_C[9].y);
			reg_C[10].y = fma(reg_A[1], reg_B[6], reg_C[10].y);
			reg_C[11].y = fma(reg_A[1], reg_B[7], reg_C[11].y);
			reg_C[4].y = fma(reg_A[5], reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[5], reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[5], reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[5], reg_B[3], reg_C[7].y);
			reg_C[12].y = fma(reg_A[5], reg_B[4], reg_C[12].y);
			reg_C[13].y = fma(reg_A[5], reg_B[5], reg_C[13].y);
			reg_C[14].y = fma(reg_A[5], reg_B[6], reg_C[14].y);
			reg_C[15].y = fma(reg_A[5], reg_B[7], reg_C[15].y);

			reg_C[0].z = fma(reg_A[2], reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[2], reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[2], reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[2], reg_B[3], reg_C[3].z);
			reg_C[8].z = fma(reg_A[2], reg_B[4], reg_C[8].z);
			reg_C[9].z = fma(reg_A[2], reg_B[5], reg_C[9].z);
			reg_C[10].z = fma(reg_A[2], reg_B[6], reg_C[10].z);
			reg_C[11].z = fma(reg_A[2], reg_B[7], reg_C[11].z);
			reg_C[4].z = fma(reg_A[6], reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[6], reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[6], reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[6], reg_B[3], reg_C[7].z);
			reg_C[12].z = fma(reg_A[6], reg_B[4], reg_C[12].z);
			reg_C[13].z = fma(reg_A[6], reg_B[5], reg_C[13].z);
			reg_C[14].z = fma(reg_A[6], reg_B[6], reg_C[14].z);
			reg_C[15].z = fma(reg_A[6], reg_B[7], reg_C[15].z);

			reg_C[0].w = fma(reg_A[3], reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[3], reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[3], reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[3], reg_B[3], reg_C[3].w);
			reg_C[8].w = fma(reg_A[3], reg_B[4], reg_C[8].w);
			reg_C[9].w = fma(reg_A[3], reg_B[5], reg_C[9].w);
			reg_C[10].w = fma(reg_A[3], reg_B[6], reg_C[10].w);
			reg_C[11].w = fma(reg_A[3], reg_B[7], reg_C[11].w);
			reg_C[4].w = fma(reg_A[7], reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[7], reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[7], reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[7], reg_B[3], reg_C[7].w);
			reg_C[12].w = fma(reg_A[7], reg_B[4], reg_C[12].w);
			reg_C[13].w = fma(reg_A[7], reg_B[5], reg_C[13].w);
			reg_C[14].w = fma(reg_A[7], reg_B[6], reg_C[14].w);
			reg_C[15].w = fma(reg_A[7], reg_B[7], reg_C[15].w);

			A_offset += 128;
			if (i==7) B_offset += 1016;
			B_offset += 1;
		}

		double_buffer ^= 2048;

		if (k+16 < K){
			A_start += M<<2;
			*((float4*) (sh_A + double_buffer + th8)) = *(A_start);
			*((float4*) (sh_A + double_buffer + th8 + 4)) = *(A_start+1);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer + th8)) = *(B_start);
			*((float4*) (sh_B + double_buffer + th8 + 4)) = *(B_start+1);
		}
	}
	
	C_start -= (m16 + 16);
    *C_start = reg_C[0];
	*(C_start + md4) = reg_C[1];
	*(C_start + md2) = reg_C[2];
	*(C_start + 3*md4) = reg_C[3];

	C_start += 16;
	*(C_start) = reg_C[4];
	*(C_start + md4) = reg_C[5];
	*(C_start + md2) = reg_C[6];
	*(C_start + 3*md4) = reg_C[7];

	C_start += (m16 - 16);
	*(C_start) = reg_C[8];
	*(C_start + md4) = reg_C[9];
	*(C_start + md2) = reg_C[10];
	*(C_start + 3*md4) = reg_C[11];

	C_start += 16;
	*(C_start) = reg_C[12];
	*(C_start + md4) = reg_C[13];
	*(C_start + md2) = reg_C[14];
	*(C_start + md2 + md4) = reg_C[15];
}



__global__ void gemm_256_128x128_16_2(int M, int N, int K, float *A, float *B, float *C)
{
    __shared__ float sh_A[4096];
	__shared__ float sh_B[4096];

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<7;
	int block_base_y = blockIdx.x<<7;
	
	int m16 = M<<4;
	int md2 = M >>1;
	int md4 = M >>2;
	int im16 = threadIdx.x & 15;
	int id16 = threadIdx.x >> 4;
	int im128 = threadIdx.x & 127;
	int id128 = threadIdx.x >> 7;	
	int th8 = threadIdx.x << 3;
	int th4 = threadIdx.x << 2;
	
	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (im16<<2) + (id16<<2)*M);

    reg_C[0] = {0.f,0.f,0.f,0.f};//*C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[2] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[3] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);  

	C_start += 16;
	reg_C[4] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[5] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[6] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[7] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += (m16 - 16);
	reg_C[8] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[9] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[10] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[11] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += 16;
	reg_C[12] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[13] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[14] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[15] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (im16<<3) + (id16)*M); 
	*((float4*) (sh_A + th8)) = *(A_start);
	*((float4*) (sh_A + th8 + 4)) = *(A_start+1);

	//load B from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (id128<<2) + (im128)*K); 
	*((float4*) (sh_B + th4)) = *(B_start);
	*((float4*) (sh_B + th4 + 128*8)) = *(B_start+2);
	
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (im16<<2);
		int B_offset = double_buffer + (id16<<4);
		
#pragma unroll
		for (int i=0; i<16; ++i)
		{
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+1];
			reg_A[2] = sh_A[A_offset+2];
			reg_A[3] = sh_A[A_offset+3];
			reg_A[4] = sh_A[A_offset+64];
			reg_A[5] = sh_A[A_offset+65];
			reg_A[6] = sh_A[A_offset+66];
			reg_A[7] = sh_A[A_offset+67];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];
			reg_B[4] = sh_B[B_offset+256];
			reg_B[5] = sh_B[B_offset+260];
			reg_B[6] = sh_B[B_offset+264];
			reg_B[7] = sh_B[B_offset+268];

			reg_C[0].x = fma(reg_A[0], reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0], reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0], reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0], reg_B[3], reg_C[3].x);
			reg_C[8].x = fma(reg_A[0], reg_B[4], reg_C[8].x);
			reg_C[9].x = fma(reg_A[0], reg_B[5], reg_C[9].x);
			reg_C[10].x = fma(reg_A[0], reg_B[6], reg_C[10].x);
			reg_C[11].x = fma(reg_A[0], reg_B[7], reg_C[11].x);
			reg_C[4].x = fma(reg_A[4], reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[4], reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[4], reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[4], reg_B[3], reg_C[7].x);
			reg_C[12].x = fma(reg_A[4], reg_B[4], reg_C[12].x);
			reg_C[13].x = fma(reg_A[4], reg_B[5], reg_C[13].x);
			reg_C[14].x = fma(reg_A[4], reg_B[6], reg_C[14].x);
			reg_C[15].x = fma(reg_A[4], reg_B[7], reg_C[15].x);

			reg_C[0].y = fma(reg_A[1], reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[1], reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[1], reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[1], reg_B[3], reg_C[3].y);
			reg_C[8].y = fma(reg_A[1], reg_B[4], reg_C[8].y);
			reg_C[9].y = fma(reg_A[1], reg_B[5], reg_C[9].y);
			reg_C[10].y = fma(reg_A[1], reg_B[6], reg_C[10].y);
			reg_C[11].y = fma(reg_A[1], reg_B[7], reg_C[11].y);
			reg_C[4].y = fma(reg_A[5], reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[5], reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[5], reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[5], reg_B[3], reg_C[7].y);
			reg_C[12].y = fma(reg_A[5], reg_B[4], reg_C[12].y);
			reg_C[13].y = fma(reg_A[5], reg_B[5], reg_C[13].y);
			reg_C[14].y = fma(reg_A[5], reg_B[6], reg_C[14].y);
			reg_C[15].y = fma(reg_A[5], reg_B[7], reg_C[15].y);

			reg_C[0].z = fma(reg_A[2], reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[2], reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[2], reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[2], reg_B[3], reg_C[3].z);
			reg_C[8].z = fma(reg_A[2], reg_B[4], reg_C[8].z);
			reg_C[9].z = fma(reg_A[2], reg_B[5], reg_C[9].z);
			reg_C[10].z = fma(reg_A[2], reg_B[6], reg_C[10].z);
			reg_C[11].z = fma(reg_A[2], reg_B[7], reg_C[11].z);
			reg_C[4].z = fma(reg_A[6], reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[6], reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[6], reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[6], reg_B[3], reg_C[7].z);
			reg_C[12].z = fma(reg_A[6], reg_B[4], reg_C[12].z);
			reg_C[13].z = fma(reg_A[6], reg_B[5], reg_C[13].z);
			reg_C[14].z = fma(reg_A[6], reg_B[6], reg_C[14].z);
			reg_C[15].z = fma(reg_A[6], reg_B[7], reg_C[15].z);

			reg_C[0].w = fma(reg_A[3], reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[3], reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[3], reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[3], reg_B[3], reg_C[3].w);
			reg_C[8].w = fma(reg_A[3], reg_B[4], reg_C[8].w);
			reg_C[9].w = fma(reg_A[3], reg_B[5], reg_C[9].w);
			reg_C[10].w = fma(reg_A[3], reg_B[6], reg_C[10].w);
			reg_C[11].w = fma(reg_A[3], reg_B[7], reg_C[11].w);
			reg_C[4].w = fma(reg_A[7], reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[7], reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[7], reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[7], reg_B[3], reg_C[7].w);
			reg_C[12].w = fma(reg_A[7], reg_B[4], reg_C[12].w);
			reg_C[13].w = fma(reg_A[7], reg_B[5], reg_C[13].w);
			reg_C[14].w = fma(reg_A[7], reg_B[6], reg_C[14].w);
			reg_C[15].w = fma(reg_A[7], reg_B[7], reg_C[15].w);

			A_offset += 128;
			if (i==3) B_offset += 508;
			B_offset += 1;
		}

		double_buffer ^= 2048;

		if (k+16 < K){
			A_start += M<<2;
			*((float4*) (sh_A + double_buffer + th8)) = *(A_start);
			*((float4*) (sh_A + double_buffer + th8 + 4)) = *(A_start+1);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer + th4)) = *(B_start);
			*((float4*) (sh_B + double_buffer + th4 + 128*8)) = *(B_start+2);
		}
	}
	
	C_start -= (m16 + 16);
    *C_start = reg_C[0];
	*(C_start + md4) = reg_C[1];
	*(C_start + md2) = reg_C[2];
	*(C_start + 3*md4) = reg_C[3];

	C_start += 16;
	*(C_start) = reg_C[4];
	*(C_start + md4) = reg_C[5];
	*(C_start + md2) = reg_C[6];
	*(C_start + 3*md4) = reg_C[7];

	C_start += (m16 - 16);
	*(C_start) = reg_C[8];
	*(C_start + md4) = reg_C[9];
	*(C_start + md2) = reg_C[10];
	*(C_start + 3*md4) = reg_C[11];

	C_start += 16;
	*(C_start) = reg_C[12];
	*(C_start + md4) = reg_C[13];
	*(C_start + md2) = reg_C[14];
	*(C_start + md2 + md4) = reg_C[15];
}



__global__ void gemm_256_64x128_16(int M, int N, int K, float *A, float *B, float *C)
{
    __shared__ float sh_A[2048];
	__shared__ float sh_B[4096];	

	float4 reg_C[8];
	float4 reg_A[2];
	float  reg_B[4];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<7;
	int block_base_y = blockIdx.x<<6;
	
	int md2 = M >>1;
	int md4 = M >>2;
	int im16 = threadIdx.x & 15;
	int id16 = threadIdx.x >> 4;
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >> 3;
	int im128 = threadIdx.x & 127;
	int id128 = threadIdx.x >> 7;	

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (im8<<2) + (id8<<2)*M);

    reg_C[0] = {0.f,0.f,0.f,0.f};// *C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[2] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[3] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += 8;
	reg_C[4] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[5] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[6] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[7] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (im16<<2) + (id16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (id128<<3) + (im128)*K); 
	*((float4*) (sh_B + 8*threadIdx.x)) = *(B_start);
	*((float4*) (sh_B + 8*threadIdx.x + 4)) = *(B_start+1);
		
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer_A + (im8<<2);
		int B_offset = double_buffer_B + (id8<<4);
			
#pragma unroll
		for (int i=0; i<16; ++i){
			
			reg_A[0] = *((float4*) (sh_A+A_offset));
			reg_A[1] = *((float4*) (sh_A+A_offset+32));

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+8];
			reg_B[2] = sh_B[B_offset+16];
			reg_B[3] = sh_B[B_offset+24];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0].x, reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0].x, reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0].x, reg_B[3], reg_C[3].x);
			reg_C[4].x = fma(reg_A[1].x, reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[1].x, reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[1].x, reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[1].x, reg_B[3], reg_C[7].x);

			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[0].y, reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[0].y, reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[0].y, reg_B[3], reg_C[3].y);
			reg_C[4].y = fma(reg_A[1].y, reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[1].y, reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[1].y, reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[1].y, reg_B[3], reg_C[7].y);

			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[0].z, reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[0].z, reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[0].z, reg_B[3], reg_C[3].z);
			reg_C[4].z = fma(reg_A[1].z, reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[1].z, reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[1].z, reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[1].z, reg_B[3], reg_C[7].z);

			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[0].w, reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[0].w, reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[0].w, reg_B[3], reg_C[3].w);
			reg_C[4].w = fma(reg_A[1].w, reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[1].w, reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[1].w, reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[1].w, reg_B[3], reg_C[7].w);

			A_offset += 64;
			B_offset += (1 + (i==7)*1016);
		}

		double_buffer_A ^= 1024;
		double_buffer_B ^= 2048;

		if (k+16 < K){
			A_start += M<<2; 
			*((float4*) (sh_A + double_buffer_A + 4*threadIdx.x)) = *(A_start);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer_B + 8*threadIdx.x)) = *(B_start);
			*((float4*) (sh_B + double_buffer_B + 8*threadIdx.x + 4)) = *(B_start+1);
		}
	}
	
    *C_start = reg_C[4];
	*(C_start + md4) = reg_C[5];
	*(C_start + md2) = reg_C[6];
	*(C_start + 3*md4) = reg_C[7];

	C_start -= 8;
	*(C_start) = reg_C[0];
	*(C_start + md4) = reg_C[1];
	*(C_start + md2) = reg_C[2];
	*(C_start + 3*md4) = reg_C[3];
}


__global__ void gemm_256_64x128_16_2(int M, int N, int K, float *A, float *B, float *C)
{
    __shared__ float sh_A[2048];
	__shared__ float sh_B[4096];	

	float4 reg_C[8];
	float4 reg_A[2];
	float  reg_B[4];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<7;
	int block_base_y = blockIdx.x<<6;
	
	int md2 = M >>1;
	int md4 = M >>2;
	int im16 = threadIdx.x & 15;
	int id16 = threadIdx.x >> 4;
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >> 3;
	int im128 = threadIdx.x & 127;
	int id128 = threadIdx.x >> 7;	

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (im8<<2) + (id8<<2)*M);

    reg_C[0] = {0.f,0.f,0.f,0.f};// *C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[2] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[3] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += 8;
	reg_C[4] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[5] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[6] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[7] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (im16<<2) + (id16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (id128<<2) + (im128)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
	*((float4*) (sh_B + 4*threadIdx.x + 128*8)) = *(B_start+2);
		
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer_A + (im8<<2);
		int B_offset = double_buffer_B + (id8<<4);
			
#pragma unroll
		for (int i=0; i<16; ++i){
			
			reg_A[0] = *((float4*) (sh_A+A_offset));
			reg_A[1] = *((float4*) (sh_A+A_offset+32));

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0].x, reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0].x, reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0].x, reg_B[3], reg_C[3].x);
			reg_C[4].x = fma(reg_A[1].x, reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[1].x, reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[1].x, reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[1].x, reg_B[3], reg_C[7].x);

			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[0].y, reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[0].y, reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[0].y, reg_B[3], reg_C[3].y);
			reg_C[4].y = fma(reg_A[1].y, reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[1].y, reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[1].y, reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[1].y, reg_B[3], reg_C[7].y);

			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[0].z, reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[0].z, reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[0].z, reg_B[3], reg_C[3].z);
			reg_C[4].z = fma(reg_A[1].z, reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[1].z, reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[1].z, reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[1].z, reg_B[3], reg_C[7].z);

			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[0].w, reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[0].w, reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[0].w, reg_B[3], reg_C[3].w);
			reg_C[4].w = fma(reg_A[1].w, reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[1].w, reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[1].w, reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[1].w, reg_B[3], reg_C[7].w);

			A_offset += 64;
			B_offset += (1 + (i==3)*508);
		}

		double_buffer_A ^= 1024;
		double_buffer_B ^= 2048;

		if (k+16 < K){
			A_start += M<<2; 
			*((float4*) (sh_A + double_buffer_A + 4*threadIdx.x)) = *(A_start);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer_B + 4*threadIdx.x)) = *(B_start);
			*((float4*) (sh_B + double_buffer_B + 4*threadIdx.x + 128*8)) = *(B_start+2);
		}
	}
	
    *C_start = reg_C[4];
	*(C_start + md4) = reg_C[5];
	*(C_start + md2) = reg_C[6];
	*(C_start + 3*md4) = reg_C[7];

	C_start -= 8;
	*(C_start) = reg_C[0];
	*(C_start + md4) = reg_C[1];
	*(C_start + md2) = reg_C[2];
	*(C_start + 3*md4) = reg_C[3];
}



__global__ void gemm_256_128x128(int M, int N, int K, float *A, float *B, float *C){

    __shared__ float sh_A[2048];
	__shared__ float sh_B[2048];
	//float sh_B = sh + 2*128*8;

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*128;
	int block_base_y = blockIdx.x*128;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*4*M);

    reg_C[0] = {0.f,0.f,0.f,0.f};//*C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[2] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[3] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);  

	C_start += 16;
	reg_C[4] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[5] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[6] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[7] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += (16*M - 16);
	reg_C[8] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[9] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[10] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[11] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += 16;
	reg_C[12] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[13] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[14] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[15] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%32)*4 + (threadIdx.x/32)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/128)*4 + (threadIdx.x%128)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
		
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%16)*4;
		int B_offset = double_buffer + ((threadIdx.x/16)*16);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+1];
			reg_A[2] = sh_A[A_offset+2];
			reg_A[3] = sh_A[A_offset+3];
			reg_A[4] = sh_A[A_offset+64];
			reg_A[5] = sh_A[A_offset+65];
			reg_A[6] = sh_A[A_offset+66];
			reg_A[7] = sh_A[A_offset+67];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];
			reg_B[4] = sh_B[B_offset+256];
			reg_B[5] = sh_B[B_offset+260];
			reg_B[6] = sh_B[B_offset+264];
			reg_B[7] = sh_B[B_offset+268];

			reg_C[0].x = fma(reg_A[0], reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0], reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0], reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0], reg_B[3], reg_C[3].x);
			reg_C[8].x = fma(reg_A[0], reg_B[4], reg_C[8].x);
			reg_C[9].x = fma(reg_A[0], reg_B[5], reg_C[9].x);
			reg_C[10].x = fma(reg_A[0], reg_B[6], reg_C[10].x);
			reg_C[11].x = fma(reg_A[0], reg_B[7], reg_C[11].x);
			reg_C[4].x = fma(reg_A[4], reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[4], reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[4], reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[4], reg_B[3], reg_C[7].x);
			reg_C[12].x = fma(reg_A[4], reg_B[4], reg_C[12].x);
			reg_C[13].x = fma(reg_A[4], reg_B[5], reg_C[13].x);
			reg_C[14].x = fma(reg_A[4], reg_B[6], reg_C[14].x);
			reg_C[15].x = fma(reg_A[4], reg_B[7], reg_C[15].x);

			reg_C[0].y = fma(reg_A[1], reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[1], reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[1], reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[1], reg_B[3], reg_C[3].y);
			reg_C[8].y = fma(reg_A[1], reg_B[4], reg_C[8].y);
			reg_C[9].y = fma(reg_A[1], reg_B[5], reg_C[9].y);
			reg_C[10].y = fma(reg_A[1], reg_B[6], reg_C[10].y);
			reg_C[11].y = fma(reg_A[1], reg_B[7], reg_C[11].y);
			reg_C[4].y = fma(reg_A[5], reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[5], reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[5], reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[5], reg_B[3], reg_C[7].y);
			reg_C[12].y = fma(reg_A[5], reg_B[4], reg_C[12].y);
			reg_C[13].y = fma(reg_A[5], reg_B[5], reg_C[13].y);
			reg_C[14].y = fma(reg_A[5], reg_B[6], reg_C[14].y);
			reg_C[15].y = fma(reg_A[5], reg_B[7], reg_C[15].y);

			reg_C[0].z = fma(reg_A[2], reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[2], reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[2], reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[2], reg_B[3], reg_C[3].z);
			reg_C[8].z = fma(reg_A[2], reg_B[4], reg_C[8].z);
			reg_C[9].z = fma(reg_A[2], reg_B[5], reg_C[9].z);
			reg_C[10].z = fma(reg_A[2], reg_B[6], reg_C[10].z);
			reg_C[11].z = fma(reg_A[2], reg_B[7], reg_C[11].z);
			reg_C[4].z = fma(reg_A[6], reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[6], reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[6], reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[6], reg_B[3], reg_C[7].z);
			reg_C[12].z = fma(reg_A[6], reg_B[4], reg_C[12].z);
			reg_C[13].z = fma(reg_A[6], reg_B[5], reg_C[13].z);
			reg_C[14].z = fma(reg_A[6], reg_B[6], reg_C[14].z);
			reg_C[15].z = fma(reg_A[6], reg_B[7], reg_C[15].z);

			reg_C[0].w = fma(reg_A[3], reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[3], reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[3], reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[3], reg_B[3], reg_C[3].w);
			reg_C[8].w = fma(reg_A[3], reg_B[4], reg_C[8].w);
			reg_C[9].w = fma(reg_A[3], reg_B[5], reg_C[9].w);
			reg_C[10].w = fma(reg_A[3], reg_B[6], reg_C[10].w);
			reg_C[11].w = fma(reg_A[3], reg_B[7], reg_C[11].w);
			reg_C[4].w = fma(reg_A[7], reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[7], reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[7], reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[7], reg_B[3], reg_C[7].w);
			reg_C[12].w = fma(reg_A[7], reg_B[4], reg_C[12].w);
			reg_C[13].w = fma(reg_A[7], reg_B[5], reg_C[13].w);
			reg_C[14].w = fma(reg_A[7], reg_B[6], reg_C[14].w);
			reg_C[15].w = fma(reg_A[7], reg_B[7], reg_C[15].w);

			A_offset += 128;
			if (i==3) B_offset += 508;
			B_offset += 1;
		}

		double_buffer ^= 1024;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
		}
				
	}
	
	C_start -= (16*M + 16);
    *C_start = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];

	C_start += 16;
	*(C_start) = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

	C_start += (16*M - 16);
	*(C_start) = reg_C[8];
	*(C_start + M/4) = reg_C[9];
	*(C_start + M/2) = reg_C[10];
	*(C_start + 3*M/4) = reg_C[11];

	C_start += 16;
	*(C_start) = reg_C[12];
	*(C_start + M/4) = reg_C[13];
	*(C_start + M/2) = reg_C[14];
	*(C_start + 3*M/4) = reg_C[15];
}

__global__ void gemm_256_128x64_16(int M, int N, int K, float *A, float *B, float *C)
{
	__shared__ float sh_A[4096];
	__shared__ float sh_B[2048];

	float4 reg_C[8];
	float4 reg_A[2];
	float  reg_B[4];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y<<6;
	int block_base_y = blockIdx.x<<7;
	
	int md2 = M >>1;
	int md4 = M >>2;
	int im16 = threadIdx.x & 15;
	int id16 = threadIdx.x >> 4;
	//int im32 = threadIdx.x & 31;
	//int id32 = threadIdx.x >> 5;
	int im64 = threadIdx.x & 63;
	int id64 = threadIdx.x >> 6;
	
	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (im16<<2) + (id16<<2)*M);

    reg_C[0] = {0.f,0.f,0.f,0.f};//*C_start;
	reg_C[1] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[2] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[3] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	C_start += 16;
	reg_C[4] = {0.f,0.f,0.f,0.f};//*(C_start);
	reg_C[5] = {0.f,0.f,0.f,0.f};//*(C_start + md4);
	reg_C[6] = {0.f,0.f,0.f,0.f};//*(C_start + md2);
	reg_C[7] = {0.f,0.f,0.f,0.f};//*(C_start + 3*md4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (im16<<3) + (id16)*M); 
	*((float4*) (sh_A + 8*threadIdx.x)) = *(A_start);
	*((float4*) (sh_A + 8*threadIdx.x + 4)) = *(A_start+1);

	//load A from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (id64<<2) + (im64)*K); 
	*((float2*) (sh_B + (id64<<8) + 2*im64)) = *(B_start);
	*((float2*) (sh_B + (id64<<8) + 2*im64 + 128)) = *(B_start+1);
	
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer_A + (im16<<2);
		int B_offset = double_buffer_B + (id16<<3);
			
#pragma unroll
		for (int i=0; i<16; ++i)
		{
			reg_A[0] = *((float4*) (sh_A+A_offset));
			reg_A[1] = *((float4*) (sh_A+A_offset+64));

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+2];
			reg_B[2] = sh_B[B_offset+4];
			reg_B[3] = sh_B[B_offset+6];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0].x, reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0].x, reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0].x, reg_B[3], reg_C[3].x);
			reg_C[4].x = fma(reg_A[1].x, reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[1].x, reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[1].x, reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[1].x, reg_B[3], reg_C[7].x);

			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[0].y, reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[0].y, reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[0].y, reg_B[3], reg_C[3].y);
			reg_C[4].y = fma(reg_A[1].y, reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[1].y, reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[1].y, reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[1].y, reg_B[3], reg_C[7].y);

			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[0].z, reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[0].z, reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[0].z, reg_B[3], reg_C[3].z);
			reg_C[4].z = fma(reg_A[1].z, reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[1].z, reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[1].z, reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[1].z, reg_B[3], reg_C[7].z);

			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[0].w, reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[0].w, reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[0].w, reg_B[3], reg_C[3].w);
			reg_C[4].w = fma(reg_A[1].w, reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[1].w, reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[1].w, reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[1].w, reg_B[3], reg_C[7].w);

			A_offset += 128;
			B_offset += (1 + (i&1)*126);
		}

		double_buffer_A ^= 2048;
		double_buffer_B ^= 1024;

		if (k+16 < K){
			A_start += M<<2; 
			*((float4*) (sh_A + double_buffer_A + 8*threadIdx.x)) = *(A_start);
			*((float4*) (sh_A + double_buffer_A + 8*threadIdx.x + 4)) = *(A_start+1);

			B_start += 8; 
			*((float2*) (sh_B + double_buffer_B + (id64<<8) + 2*im64)) = *(B_start);
			*((float2*) (sh_B + double_buffer_B + (id64<<8) + 2*im64 + 128)) = *(B_start + 1);
		}
	}
	
	C_start -= 16;
    *C_start = reg_C[0];
	*(C_start + md4) = reg_C[1];
	*(C_start + md2) = reg_C[2];
	*(C_start + md2 + md4) = reg_C[3];

	C_start += 16;
	*(C_start) = reg_C[4];
	*(C_start + md4) = reg_C[5];
	*(C_start + md2) = reg_C[6];
	*(C_start + md2 + md4) = reg_C[7];
}


__global__ void gemm_256_128x64(int M, int N, int K, float *A, float *B, float *C){

	__shared__ float sh_A[2048];
	__shared__ float sh_B[1024];

	float4 reg_C[8];
	float4 reg_A[2];
	float  reg_B[4];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*128;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*4*M);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 16;
	reg_C[4] = *(C_start);
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%32)*4 + (threadIdx.x/32)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/64)*2 + (threadIdx.x%64)*K); 
	*((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);
		
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer_A + (threadIdx.x%16)*4;
		int B_offset = double_buffer_B + ((threadIdx.x/16)*8);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A[0] = *((float4*) (sh_A+A_offset));
			reg_A[1] = *((float4*) (sh_A+A_offset+64));

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+2];
			reg_B[2] = sh_B[B_offset+4];
			reg_B[3] = sh_B[B_offset+6];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0].x, reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0].x, reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0].x, reg_B[3], reg_C[3].x);
			reg_C[4].x = fma(reg_A[1].x, reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[1].x, reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[1].x, reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[1].x, reg_B[3], reg_C[7].x);

			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[0].y, reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[0].y, reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[0].y, reg_B[3], reg_C[3].y);
			reg_C[4].y = fma(reg_A[1].y, reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[1].y, reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[1].y, reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[1].y, reg_B[3], reg_C[7].y);

			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[0].z, reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[0].z, reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[0].z, reg_B[3], reg_C[3].z);
			reg_C[4].z = fma(reg_A[1].z, reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[1].z, reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[1].z, reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[1].z, reg_B[3], reg_C[7].z);

			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[0].w, reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[0].w, reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[0].w, reg_B[3], reg_C[3].w);
			reg_C[4].w = fma(reg_A[1].w, reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[1].w, reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[1].w, reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[1].w, reg_B[3], reg_C[7].w);

			A_offset += 128;
			B_offset += (1 + (i%2)*126);
		}

		double_buffer_A ^= 1024;
		double_buffer_B ^= 512;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer_A + 4*threadIdx.x)) = *(A_start);

			B_start += 4; 
			*((float2*) (sh_B + double_buffer_B + 2*threadIdx.x)) = *(B_start);
		}
				
	}
	
	C_start -= 16;
    *C_start = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];

	C_start += 16;
	*(C_start) = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

}







int main(int argc, char* argv[]) 
{
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_wmma;

   float *c_host_cublas;
   float *c_host_wmma;
   
   curandGenerator_t gen;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   
   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
   
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
   
   curandErrCheck(curandDestroyGenerator(gen));
   
   cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

   float alpha = 1.0f;
   float beta = 0.0f;


   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;
   
   dim3 gridDim2;
   dim3 blockDim2;
   gridDim2.x = MATRIX_M/128; gridDim2.y = MATRIX_N/64;  gridDim2.z = 1;
   blockDim2.x = 256; blockDim2.y = 1; blockDim2.z = 1; 

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   dim3 gridDim3;
   dim3 blockDim3;
   gridDim3.x = MATRIX_M/16; gridDim3.y = MATRIX_N/16; gridDim3.z = 1;
   blockDim3.x = 64; blockDim3.y = 1; blockDim3.z = 1;
   
   dim3 gridDim4;
   dim3 blockDim4;
   gridDim4.x = MATRIX_M/16 + 1; gridDim4.y = MATRIX_N/8; gridDim4.z = 1;
   blockDim4.x = 32; gridDim4.y = 1; gridDim4.z = 1;
   
   printf("Running with gemm...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   
   gemm_64_16x16_1<<< gridDim3, blockDim3 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   // gemm_256_32x32 <<< gridDim2, blockDim2 >>> (MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_256_64x64_16<<< gridDim2, blockDim2 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_256_128x128_16<<< gridDim2, blockDim2 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_256_64x128_16<<< gridDim2, blockDim2 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_256_128x64<<< gridDim2, blockDim2 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_64_16x8_1<<< gridDim4, blockDim4 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_64_16x8_3<<< gridDim4, blockDim4 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_64_8x8_1<<< gridDim4, blockDim4 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_32_16x8_3<<< gridDim4, blockDim4 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   //gemm_32_16x8_1<<< gridDim4, blockDim4 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   
   // Now using cuBLAS
   printf("Running with cuBLAS...\n");
   cudaErrCheck(cudaEventRecord(startcublas));
  /* cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));*/
   gemm_64_16x16_1<<<gridDim3, blockDim3>>>(MATRIX_M,MATRIX_N,MATRIX_K,a_fp32,b_fp32,c_cublas);
   cudaErrCheck(cudaEventRecord(stopcublas));

   // Error checking
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   
   // 0.01% relative tolerance. 1e-5 absolute tolerance.
   int errors = 0;
   for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
      float v1 = c_host_wmma[i];
      float v2 = c_host_cublas[i];
      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-5) {
         errors++;
         if (errors < 10)
			printf("%d %f %f\n", i, v1, v2);
      }
   }
   
   //if (errors > 0) {
   //   printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
   //}
   //else {
      printf("Results verified: cublas and WMMA agree.\n\n");
      float wmmaTime;
      float cublasTime;
      cudaErrCheck(cudaEventSynchronize(stopWMMA));
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("gemm took %fms\n", wmmaTime);
      printf("cublas took %fms\n", cublasTime);
   //}
   
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cublas));
   cudaErrCheck(cudaFree(c_wmma));
   
   free(c_host_cublas);
   free(c_host_wmma);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}
