#include "cuda_fp16.h"

__device__ void gemm_256_16x16(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*16*16;

	float reg_C;
	float reg_A;
	float reg_B;

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*16;
	int block_base_y = blockIdx.x*16;

	//Load C from global memory to register file
	float *C_start = (C + block_base_x*M + block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M);

    reg_C = *C_start;

	//load A from global memory to shared memory
	float *A_start = (A + block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M); 
	* (sh_A + threadIdx.x) = *(A_start);

	//load B from global memory to shared memory
	float *B_start = (B + K*block_base_x + (threadIdx.x/16) + (threadIdx.x%16)*K); 
	* (sh_B + threadIdx.x) = *(B_start);
		
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16){

		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%16);
		int B_offset = double_buffer + (threadIdx.x/16);
			
#pragma unroll
		for (int i=0; i<16; ++i)	{
			reg_A = sh_A[A_offset]; 
			reg_B = sh_B[B_offset]; 
			reg_C = fma(reg_A, reg_B, reg_C);

			A_offset += 16;
			B_offset += 16;
		}

		double_buffer ^= 256;

		if (k+16 < K){
			A_start += 16*M; 
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 16; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
				
	}
	*(C_start) = reg_C;
}

__device__ void gemm_256_16x16_w(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*16*16;

	float reg_C;
	float reg_A, reg_A2;
	float reg_B, reg_B2;

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y << 4;
	int block_base_y = blockIdx.x << 4;

	//Load C from global memory to register file
	float *C_start = (C + block_base_x*M + block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M);

    reg_C = *C_start;

	//load A from global memory to shared memory
	float *A_start = (A + block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M); 
	* (sh_A + threadIdx.x) = *(A_start);

	//load B from global memory to shared memory
	float *B_start = (B + K*block_base_x + (threadIdx.x/16) + (threadIdx.x%16)*K); 
	* (sh_B + threadIdx.x) = *(B_start);
		
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x & 15);  // threadIdx.x%16
		int B_offset = double_buffer + (threadIdx.x >> 4);  // threadIdx.x/16
		
#pragma unroll
		for (int i=0; i<8; i += 2)   // loop更大，有更多优化空间？
		{
			reg_A = sh_A[A_offset];  // 2个一起加载：需要解决后面的加16的问题
			reg_B = sh_B[B_offset];  // 共享内存访问 一个warp只访问0和1两个元素
			
			reg_C = fma(reg_A, reg_B, reg_C);  // 周期数
			
			A_offset += 16;
			B_offset += 16;
			
			reg_A2 = sh_A[A_offset];           // 可以用prefetch吗
			reg_B2 = sh_B[B_offset];
			
			reg_C = fma(reg_A2, reg_B2, reg_C);   // reg_C依赖
		}

		double_buffer ^= 256;

		if (k+16 < K)
		{
			A_start += M << 4;   // 16*M
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 16; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
	}
	
	*(C_start) = reg_C;
}

__device__ void gemm_256_32x32_nrb(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*32*8;

	//float4 reg_C;
	//float4 reg_A;
	//float  reg_B;

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*32;
	int block_base_y = blockIdx.x*32;

	//Load C from global memory to register file
	float *C_start = (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M); //(float4 *) 

    //reg_C = *C_start;

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
		for (int i=0; i<8; ++i)	
		{
			//reg_A = *((float4*) (sh_A + A_offset)); 
			//reg_B = sh_B[B_offset]; 

			*(C_start) = fma(*(sh_A + A_offset), sh_B[B_offset], *(C_start));
			*(C_start + 1) = fma(*(sh_A + A_offset + 1), sh_B[B_offset], *(C_start + 1));
			*(C_start + 2) = fma(*(sh_A + A_offset + 2), sh_B[B_offset], *(C_start + 2));
			*(C_start + 3) = fma(*(sh_A + A_offset + 3), sh_B[B_offset], *(C_start + 3));

			A_offset += 32;
			B_offset += 32;
		}
		
		double_buffer ^= 256;

		if (k+8 < K)
		{
			A_start += 8*M; 
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 8; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}

	}
	
	//*(C_start) = reg_C;

}

__device__ void gemm_256_32x32_mdb(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*32*8;

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
		
		double_buffer ^= 256;

		if (k+8 < K){
			A_start += 8*M; 
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 8; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
		
// 是不是跟pragma unroll相关		
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


__device__ void gemm_256_32x32_vl(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 512;  // 2*32*8

	float4 reg_C;
	//float4 reg_A;
	//float  reg_B;
	
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
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			//reg_A = *((float4*) (sh_A + A_offset)); 
			//reg_B = sh_B[B_offset]; 

			reg_C.x = fma(*(sh_A + A_offset), sh_B[B_offset], reg_C.x);
			reg_C.y = fma(*(sh_A + A_offset+1), sh_B[B_offset], reg_C.y);
			reg_C.z = fma(*(sh_A + A_offset+2), sh_B[B_offset], reg_C.z);
			reg_C.w = fma(*(sh_A + A_offset+3), sh_B[B_offset], reg_C.w);

			A_offset += 32;
			B_offset += 32;
		}

		double_buffer ^= 256;

		if (k+8 < K){
			A_start += M << 3; 
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 8; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
				
	}
	
	*(C_start) = reg_C;

}

__device__ void gemm_256_32x32(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 512;  // 2*32*8

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

	//load B from global memory to shared memory: gld_efficiency不高
	float *A_start = (A + block_base_y + (im32) + (id32)*M); 
	*(sh_A + threadIdx.x) = *(A_start); // 列存储变为按行存储的了

	//load A from global memory to shared memory
	float *B_start = (B + K*block_base_x + (id32) + (im32)*K); 
	*(sh_B + threadIdx.x) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (im8 << 2);
		int B_offset = double_buffer + id8;
			
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
			A_start += M << 3; 
			* (sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 8; 
			* (sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
				
	}
	*(C_start) = reg_C;

}

__device__ void gemm_256_32x32_orig(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*32*8;

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


__device__ void gemm_256_32x32_16k(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 1024;

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
	float2 *A_start = (float2*)(A + block_base_y + (threadIdx.x%16)*2 + (threadIdx.x/16)*M); 
	*((float2*)(sh_A + 2*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float *B_start = (B + K*block_base_x + (threadIdx.x/32) + (threadIdx.x%32)*K); 
	*(sh_B + threadIdx.x) = *(B_start);
	*(sh_B + threadIdx.x + 32) = *(B_start+1);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*4;
		int B_offset = double_buffer + (threadIdx.x/8);
			
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
			A_start += 8*M; 
			*((float2*)(sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);

			B_start += 16;
			*(sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
	}
	
	*(C_start) = reg_C;
}


__device__ void gemm_256_32x32_half(int M, int N, int K, float *A, float *B, float *C, half *sh){

	half *sh_A = sh;
	half *sh_B = sh + 512;  // 2*32*8

	half2 reg_C[2];
	half2 reg_A[2];
	half2  reg_B;
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >> 3;
	int im32 = threadIdx.x & 31;
	int id32 = threadIdx.x >> 5;

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y << 5;
	int block_base_y = blockIdx.x << 5;

	//Load C from global memory to register file
	float2 *C_start = (float2 *) (C + block_base_x*M + block_base_y + (im8<<2) + (id8)*M);

    reg_C[0] = __float22half2_rn(*C_start);
	reg_C[1] = __float22half2_rn(*(C_start+1));

	//load B from global memory to shared memory
	float *A_start = (A + block_base_y + (im32) + (id32)*M); 
	*(sh_A + threadIdx.x) = __float2half(*(A_start));

	//load A from global memory to shared memory
	float *B_start = (B + K*block_base_x + (id32) + (im32)*K); 
	*(sh_B + threadIdx.x) = __float2half(*(B_start));

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8)
	{
		__syncthreads();
		int A_offset = double_buffer + (im8 << 2);
		int B_offset = double_buffer + id8;
			
#pragma unroll
		for (int i=0; i<8; ++i)	
		{
			reg_A[0] = *((half2*) (sh_A + A_offset)); 
			reg_B.x = sh_B[B_offset]; 
			reg_B.x = reg_B.y; 

			reg_C[0] = __hfma2(reg_A[0], reg_B, reg_C[0]);
			//reg_C.y = hfma(reg_A[0].y, reg_B, reg_C.y);
			
			reg_A[1] = *((half2*) (sh_A + A_offset + 2)); 
			//reg_C.z = hfma(reg_A[1].x, reg_B, reg_C.z);
			//reg_C.w = hfma(reg_A[1].y, reg_B, reg_C.w);
			reg_C[1] = __hfma2(reg_A[1], reg_B, reg_C[1]);
			
			A_offset += 32;
			B_offset += 32;
		}

		double_buffer ^= 256;

		if (k+8 < K){
			A_start += M << 3; 
			*(sh_A + double_buffer + threadIdx.x) = *(A_start);

			B_start += 8; 
			*(sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
	}
	//*(C_start) = reg_C;
	
	*(C_start) = __half22float2(reg_C[0]);
	*(C_start+1) = __half22float2(reg_C[1]);
}



__device__ void gemm_256_64x64(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*64*8;

	float4 reg_C[4];
	float4 reg_A[2];
	float  reg_B[2];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M);
    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + 8);
	reg_C[2] = *(C_start + 8*M);
	reg_C[3] = *(C_start + 8 + 8*M);
	
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

__device__ void gemm_256_64x64(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*64*8;

	float4 reg_C[4];
	float4 reg_A[2];
	float  reg_B[2];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M);
    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + 8);
	reg_C[2] = *(C_start + 8*M);
	reg_C[3] = *(C_start + 8 + 8*M);
	
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


__device__ void gemm_256_64x64_16(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*64*16;

	float4 reg_C[4];
	float4 reg_A[2];
	float  reg_B[2];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M);
    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + 8);
	reg_C[2] = *(C_start + 8*M);
	reg_C[3] = *(C_start + 8 + 8*M);
	
	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load B from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/64)*4 + (threadIdx.x%64)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
	
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*4;
		int B_offset = double_buffer + (threadIdx.x/8)*2;

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
			if (i%4) B_offset += 252;
		}

		double_buffer ^= 512;

		if (k+16 < K)
		{
			A_start += 4*M; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
		}
	}
	
	*(C_start) = reg_C[0];
	*(C_start + 8) = reg_C[1];
	*(C_start + 8*M) = reg_C[2];
	*(C_start + 8 + 8*M) = reg_C[3];
}




__device__ void gemm_256_128x64(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*128*8;

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


__device__ void gemm_256_128x64_16(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*128*16;

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

__device__ void gemm_256_64x128(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*64*8;

	float4 reg_C[8];
	float4 reg_A[2];
	float  reg_B[4];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*128;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*4*M);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 8;
	reg_C[4] = *(C_start);
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float2 *A_start = (float2*) (A + block_base_y + (threadIdx.x%32)*2 + (threadIdx.x/32)*M); 
	*((float2*) (sh_A + 2*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/128)*4 + (threadIdx.x%128)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
		
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer_A + (threadIdx.x%8)*4;
		int B_offset = double_buffer_B + ((threadIdx.x/8)*16);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
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

		double_buffer_A ^= 512;
		double_buffer_B ^= 1024;

		if (k+8 < K){
			A_start += 4*M; 
			*((float2*) (sh_A + double_buffer_A + 2*threadIdx.x)) = *(A_start);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer_B + 4*threadIdx.x)) = *(B_start);
		}
				
	}
    *C_start = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

	C_start -= 8;
	*(C_start) = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];

}

__device__ void gemm_256_64x128_16(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*64*16;

	float4 reg_C[8];
	float4 reg_A[2];
	float  reg_B[4];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*128;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*4*M);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 8;
	reg_C[4] = *(C_start);
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/128)*8 + (threadIdx.x%128)*K); 
	*((float4*) (sh_B + 8*threadIdx.x)) = *(B_start);
	*((float4*) (sh_B + 8*threadIdx.x + 4)) = *(B_start+1);
		
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{

		__syncthreads();
		int A_offset = double_buffer_A + (threadIdx.x%8)*4;
		int B_offset = double_buffer_B + ((threadIdx.x/8)*16);
			
#pragma unroll
		for (int i=0; i<16; ++i)	{
			
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
			B_offset += (1 + (i==7)*1014);
		}

		double_buffer_A ^= 1024;
		double_buffer_B ^= 2048;

		if (k+16 < K){
			A_start += 4*M; 
			*((float4*) (sh_A + double_buffer_A + 4*threadIdx.x)) = *(A_start);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer_B + 8*threadIdx.x)) = *(B_start);
			*((float4*) (sh_B + double_buffer_B + 8*threadIdx.x + 4)) = *(B_start+1);
		}
				
	}
	
    *C_start = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

	C_start -= 8;
	*(C_start) = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];
}

__device__ void gemm_256_128x128(int M, int N, int K, float *A, float *B, float *C, float *sh){

    float *sh_A = sh;
	float *sh_B = sh + 2*128*8;

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*128;
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

	C_start += (16*M - 16);
	reg_C[8] = *(C_start);
	reg_C[9] = *(C_start + M/4);
	reg_C[10] = *(C_start + M/2);
	reg_C[11] = *(C_start + 3*M/4);

	C_start += 16;
	reg_C[12] = *(C_start);
	reg_C[13] = *(C_start + M/4);
	reg_C[14] = *(C_start + M/2);
	reg_C[15] = *(C_start + 3*M/4);

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




__device__ void gemm_256_128x128_16(int M, int N, int K, float *A, float *B, float *C, float *sh){

    float *sh_A = sh;
	float *sh_B = sh + 2*128*16;

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*128;
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

	C_start += (16*M - 16);
	reg_C[8] = *(C_start);
	reg_C[9] = *(C_start + M/4);
	reg_C[10] = *(C_start + M/2);
	reg_C[11] = *(C_start + 3*M/4);

	C_start += 16;
	reg_C[12] = *(C_start);
	reg_C[13] = *(C_start + M/4);
	reg_C[14] = *(C_start + M/2);
	reg_C[15] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%16)*8 + (threadIdx.x/16)*M); 
	*((float4*) (sh_A + 8*threadIdx.x)) = *(A_start);
	*((float4*) (sh_A + 8*threadIdx.x + 4)) = *(A_start+1);

	//load B from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/128)*8 + (threadIdx.x%128)*K); 
	*((float4*) (sh_B + 8*threadIdx.x)) = *(B_start);
	*((float4*) (sh_B + 8*threadIdx.x + 4)) = *(B_start+1);
	
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=16)
	{
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%16)*4;
		int B_offset = double_buffer + ((threadIdx.x/16)*16);
		
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
			if (i==7) B_offset += 1014;
			B_offset += 1;
		}

		double_buffer ^= 2048;

		if (k+16 < K){
			A_start += 4*M;
			*((float4*) (sh_A + double_buffer + 8*threadIdx.x)) = *(A_start);
			*((float4*) (sh_A + double_buffer + 8*threadIdx.x + 4)) = *(A_start+1);

			B_start += 4; 
			*((float4*) (sh_B + double_buffer + 8*threadIdx.x)) = *(B_start);
			*((float4*) (sh_B + double_buffer + 8*threadIdx.x + 4)) = *(B_start+1);
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
