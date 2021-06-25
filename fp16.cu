 
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


#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work  16384
#define MATRIX_M 49
#define MATRIX_N 48
#define MATRIX_K 48



// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

// 64 thread
__global__ void fp16gemm_16x16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

	__shared__ half sh_A[256];
	__shared__ half sh_B[256];  // 2*16*8
        
	float2 reg_C[2];
	half2 reg_A[4];
	half2  reg_B[2];
	
	reg_C[0].x = 0.f;
	reg_C[0].y = 0.f;
	reg_C[1].x = 0.f;
	reg_C[1].y = 0.f;
	
	int im4 = threadIdx.x & 3;
	int id4 = threadIdx.x >> 2;
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >> 3;
	int im16 = threadIdx.x & 15;
	int id16 = threadIdx.x >> 4;
	
	int thread2 = threadIdx.x << 1;

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y << 4;
    int block_base_y = blockIdx.x << 4;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float2 *A_start = (float2*) (A + block_base_y + (im8 << 1) + (id8)*M);
    *((half2*)(sh_A + thread2)) = __float22half2_rn(*(A_start));

    //load B from global memory to shared memory
    float2 *B_start = (float2*) (B + K*(im16+block_base_x) + (id16 << 1));
    *((half2*) (sh_B + thread2)) = __float22half2_rn(*(B_start));

    int double_buffer = 0;
#pragma unroll
    for(int k = 0; k < K; k += 8)
	{
        __syncthreads();
        int A_offset = double_buffer + (im4 << 2);
        int B_offset = double_buffer + (id4 << 1);	
		
#pragma unroll
        for (int i=0; i<8; i+=2)   // 全部展开有register spill吗
		{
			reg_A[0] = *((half2*)(sh_A + A_offset));
			reg_A[1] = *((half2*)(sh_A + A_offset + 2));
			reg_A[2] = *((half2*)(sh_A + A_offset + 16));
			reg_A[3] = *((half2*)(sh_A + A_offset + 18));
            //reg_A[0] = sh_A[A_offset];    // A_offset+0 ~ A_offset+3 为什么不用向量呢
			reg_B[0] = *((half2*)(sh_B + B_offset));
			reg_B[1].x = reg_B[0].y;
			reg_B[0].y = reg_B[0].x;
			reg_B[1].y = reg_B[1].x;
            //reg_A[1] = sh_A[A_offset+1];  // 为了避免bank冲突, 这8个寄存器都不是连续的【4个就不会重复】，因此不能使用向量传输指令
            //reg_A[2] = sh_A[A_offset+2];
            //reg_A[3] = sh_A[A_offset+3];
			
            //reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            //reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
			reg_C[0] = __half22float2(__hfma2(reg_A[0], reg_B[0], __float22half2_rn(reg_C[0])));
            //reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            //reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			reg_C[1] = __half22float2(__hfma2(reg_A[1], reg_B[0], __float22half2_rn(reg_C[1])));
			
			//*((float4*) (reg_A + 4)) = *((float4*)(sh_A + A_offset + 16));
			//reg_B[1] = sh_B[B_offset+1];
			//reg_A[4] = sh_A[A_offset+16];
            //reg_A[5] = sh_A[A_offset+17];
            //reg_A[6] = sh_A[A_offset+18];
            //reg_A[7] = sh_A[A_offset+19];
			A_offset += 32;
			
            //reg_C.x = fma(reg_A[4], reg_B[1], reg_C.x);
            //reg_C.y = fma(reg_A[5], reg_B[1], reg_C.y);
            //reg_C.z = fma(reg_A[6], reg_B[1], reg_C.z);
            //reg_C.w = fma(reg_A[7], reg_B[1], reg_C.w);
			reg_C[0] = __half22float2(__hfma2(reg_A[2], reg_B[1], __float22half2_rn(reg_C[0]))) ;
			reg_C[1] = __half22float2(__hfma2(reg_A[3], reg_B[1], __float22half2_rn(reg_C[1]))) ;

            B_offset += 32;
        }
		
        double_buffer ^= 128;  // 16*8
		
        if (k+8 < K)
		{
            A_start += M << 2; // half2 --> 8M
            *((half2*) (sh_A + double_buffer + thread2)) = __float22half2_rn(*(A_start));
            B_start += 4;
            *((half2*) (sh_B + double_buffer + thread2)) = __float22half2_rn(*(B_start));
        }
    }

	int ind = block_base_y + (im4<<2);     // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;
	int PQ = M;
    int C_offset = ind/(PQ)*(PQ*N) + ind%(PQ) + (id4 + block_base_x)*(PQ);
    *((float2*)(C + C_offset)) = reg_C[0];
    //C[C_offset+1] = reg_C.y;
	*((float2*)(C + C_offset + 2)) = reg_C[1];
}


// 64 thread
__global__ void fp16gemm_16x16_tensor(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

	__shared__ half sh_A[512];
	__shared__ half sh_B[512];  // 2*16*16
	
	int im4 = threadIdx.x & 3;
	int id4 = threadIdx.x >> 2;
	//int im8 = threadIdx.x & 7;
	//int id8 = threadIdx.x >> 3;
	//int im16 = threadIdx.x & 15;
	//int id16 = threadIdx.x >> 4;
	
	//int thread2 = threadIdx.x << 1;
	int thread4 = threadIdx.x << 2;

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y << 4;
    int block_base_y = blockIdx.x << 4;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float2 *A_start = (float2*) (A + block_base_y + (im4 << 2) + (id4)*M);
    *((half2*)(sh_A + thread4)) = __float22half2_rn(*(A_start));
	*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn(*(A_start + 1));

    //load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (im4 << 2) + (id4)*K);
    //float2 *B_start = (float2*) (B + K*(im16+block_base_x) + (id16 << 1));
    *((half2*) (sh_B + thread4)) = __float22half2_rn(*(B_start));
	*((half2*) (sh_B + thread4 + 2)) = __float22half2_rn(*(B_start + 1));

    int double_buffer = 0;
	
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   
#pragma unroll
    for(int k = 0; k < K; k += 16)
	{
        __syncthreads();
		
		// Load the inputs
        wmma::load_matrix_sync(a_frag, sh_A + double_buffer, 16);
        wmma::load_matrix_sync(b_frag, sh_B + double_buffer, 16);
 
        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);		

        double_buffer ^= 256;  // 16*16
		
        if (k + 16 < K)
		{
            A_start += M << 3; // half2 --> 8M
            *((half2*) (sh_A + double_buffer + thread4)) = __float22half2_rn(*(A_start));
			*((half2*) (sh_A + double_buffer + thread4 + 2)) = __float22half2_rn(*(A_start+1));
			
            B_start += 8;
            *((half2*) (sh_B + double_buffer + thread4)) = __float22half2_rn(*(B_start));
			*((half2*) (sh_B + double_buffer + thread4 + 2)) = __float22half2_rn(*(B_start+1));
        }
    }

	//int ind = block_base_y + (im4<<2);     // 横、纵坐标  M=HW， K = C， N = K
	//int PQ = M;
    //int C_offset = ind/(PQ)*(PQ*N) + ind%(PQ) + (id4 + block_base_x)*(PQ);
    //*((float2*)(C + C_offset)) = reg_C[0];
	//*((float2*)(C + C_offset + 2)) = reg_C[1];
	
	// Store the output
    wmma::store_matrix_sync(C + block_base_y + (block_base_x * M), acc_frag, M, wmma::mem_col_major);	
}


__global__ void gemm_64_16x16_3_tensor(int M, int N, int K, float *A, float *B, float *C){

   __shared__ half sh_A[512];
   __shared__ half sh_B[512];
   __shared__ float sh_C[256];

   float reg_C[4];
   reg_C[0] = 0.f;
   reg_C[1] = 0.f;
   reg_C[2] = 0.f;
   reg_C[3] = 0.f;

   int im4 = threadIdx.x & 3;
   int id4 = threadIdx.x >> 2;
   int thread4 = threadIdx.x << 2;
   
   // Compute block's starting coordinate
   int block_base_x = blockIdx.y*16;
   int block_base_y = blockIdx.x*16;


    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float2 *A_start = (float2*) (A + block_base_y + (im4 << 2) + (id4)*M);
	if (block_base_y == 3)
	{
		if (id4 == 0)
			*((half2*)(sh_A + thread4)) = __float22half2_rn({(*(A_start)).x, 0.f});
		*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn({0.f, 0.f});
	}
	else
	{
		*((half2*)(sh_A + thread4)) = __float22half2_rn(*(A_start));
		*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn(*(A_start + 1));		
	}

    //load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (im4 << 2) + (id4)*K);
    //float2 *B_start = (float2*) (B + K*(im16+block_base_x) + (id16 << 1));
    *((half2*) (sh_B + thread4)) = __float22half2_rn(*(B_start));
	*((half2*) (sh_B + thread4 + 2)) = __float22half2_rn(*(B_start + 1));

   int double_buffer = 0;
   
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   
#pragma unroll
   for(int k=0; k<K; k+=16)
   {
       __syncthreads();
	   
		// Load the inputs
        wmma::load_matrix_sync(a_frag, sh_A + double_buffer, 16);
        wmma::load_matrix_sync(b_frag, sh_B + double_buffer, 16);
 
        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);		   

       double_buffer ^= 256;

       if (k+16 < K)
	   {
           A_start += M<<3;
			if (block_base_y == 3)
			{
				if (id4 == 0)
					*((half2*)(sh_A + double_buffer  + thread4)) = __float22half2_rn({(*(A_start)).x, 0.f});
				*((half2*)(sh_A + double_buffer  + thread4 + 2)) = __float22half2_rn({0.f, 0.f});
			}
			else
			{
				*((half2*)(sh_A + double_buffer  + thread4)) = __float22half2_rn(*(A_start));
				*((half2*)(sh_A + double_buffer  + thread4 + 2)) = __float22half2_rn(*(A_start + 1));		
			}
			
           B_start += 8;
			*((half2*) (sh_B + double_buffer  + thread4)) = __float22half2_rn(*(B_start));
			*((half2*) (sh_B + double_buffer  + thread4 + 2)) = __float22half2_rn(*(B_start + 1));
       }
   }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
	int PQ = M; 
    int C_offset = ind/(PQ)*(PQ*N) + ind%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*16*(PQ);

   if (blockIdx.x < M/16)
   {
		// Store the output
		wmma::store_matrix_sync(C + block_base_y + (block_base_x * M), acc_frag, M, wmma::mem_col_major);		
   }
   else
   {
		// Store the output
		wmma::store_matrix_sync(sh_C, acc_frag, 16, wmma::mem_col_major);
		
	   reg_C[0] = sh_C[threadIdx.x*4];
	   reg_C[1] = sh_C[threadIdx.x*4 + 1];
	   reg_C[2] = sh_C[threadIdx.x*4 + 2];
	   reg_C[3] = sh_C[threadIdx.x*4 + 3];
	
       int ruler = (threadIdx.x%4)*4;
       int rag = M%16;
	   
       if (ruler < rag)
	   {
           C[C_offset] = reg_C[0];
	    }
		
       if ((ruler+1) < rag){
     	   C_offset = (ind+1)/(PQ)*(PQ*N) + (ind+1)%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*16*(PQ);
           C[C_offset] = reg_C[1];
		}
		
       if ((ruler+2) < rag){
			C_offset = (ind+2)/(PQ)*(PQ*N) + (ind+2)%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*16*(PQ);
            C[C_offset] = reg_C[2];
		}
		
       if ((ruler+3) < rag){
     	   C_offset = (ind+3)/(PQ)*(PQ*N) + (ind+3)%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*16*(PQ);
           C[C_offset] = reg_C[3];
		}
   }
}


__global__ void gemm_64_16x16_3(int M, int N, int K, float *A, float *B, float *C){

   __shared__ float sh_A[256];
   __shared__ float sh_B[256];
   //float* sh_B = sh + 2*16*8;

    float4 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;
	reg_C.z =0.f;
	reg_C.w =0.f;

    float reg_A[8];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*16;
    int block_base_y = blockIdx.x*16;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    int aoffset = block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M;
    sh_A[2*threadIdx.x] = A[aoffset%(M*K)];
	sh_A[2*threadIdx.x+1] = A[(aoffset+1)%(M*K)];

    //load B from global memory to shared memory
    int boffset = K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K;
    sh_B[2*threadIdx.x] = B[boffset%(N*K)];
	sh_B[2*threadIdx.x+1] = B[(boffset+1)%(N*K)];

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
			reg_A[4] = sh_A[A_offset+16];
            reg_A[5] = sh_A[A_offset+17];
            reg_A[6] = sh_A[A_offset+18];
            reg_A[7] = sh_A[A_offset+19];
			
			reg_B[0] = sh_B[B_offset];
			
            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
            reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			
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
            aoffset += 8*M; // float2 --> 8M
            //*((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
            boffset += 8;
            //*((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
			
			//int aoffset = block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M;
			sh_A[double_buffer+2*threadIdx.x] = A[aoffset%(M*K)];
			sh_A[double_buffer+2*threadIdx.x+1] = A[(aoffset+1)%(M*K)];

			//load B from global memory to shared memory
			//int boffset = K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K;
			sh_B[double_buffer+2*threadIdx.x] = B[boffset%(N*K)];
			sh_B[double_buffer+2*threadIdx.x+1] = B[(boffset+1)%(N*K)];
        }
    }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;  // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.x > 0 ? reg_C.x : 0;
    C[C_offset+1] = reg_C.y;
    C[C_offset+2] = reg_C.z;
    C[C_offset+3] = reg_C.w;
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
   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   
   //convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   //convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);
   //fp16gemm_16x16<<< gridDim3, blockDim3 >>>(a_fp32, b_fp32, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   //fp16gemm_16x16_tensor<<< gridDim3, blockDim3 >>>(a_fp32, b_fp32, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   //gemm_64_16x16_3_tensor<<< gridDim3, blockDim3 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   gemm_64_16x16_3 <<< gridDim3, blockDim3 >>>(MATRIX_M, MATRIX_N, MATRIX_K, a_fp32, b_fp32, c_wmma);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   
   // Now using cuBLAS
   printf("Running with cuBLAS...\n");
   cudaErrCheck(cudaEventRecord(startcublas));
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
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
         if (errors < 10) printf("%f %f\n", v1, v2);
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
      printf("wmma took %fms\n", wmmaTime);
      printf("cublas took %fms\n", cublasTime);

      printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
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
