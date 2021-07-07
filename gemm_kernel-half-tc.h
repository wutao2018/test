/*
 * gemm_kernel.h
 *
 *  Created on: Nov 5, 2018
 *      Author: cambricon
 */

#ifndef GEMM_KERNEL_H_
#define GEMM_KERNEL_H_

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

//(N*P*Q)%16==0 && (P*Q)%4==0
__device__ void gemm_64_16x16_1(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

	float* sh_A = sh;
    float* sh_B = sh + 2*16*8;

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
            A_start += 4*M; // float2 --> 8M
            *((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
            B_start += 4;
            *((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
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

// 8x8 gemm with 64 thread
__device__ void gemm_64_8x8_1(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

	float* sh_A = sh;
    float* sh_B = sh + 2*8*8;

    float reg_C = 0.f;

    float reg_A[8];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*8;
    int block_base_y = blockIdx.x*8;

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
            A_start += 4*M; // float2 --> 8M
            *((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
            B_start += 4;
            *((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
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


// 64 thread  --  new
__device__ void fp16gemm_16x16_1(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float* sh) {

	//__shared__ half sh_A[256];
	//__shared__ half sh_B[256];  // 2*16*8
	
	half* sh_A = (half*)sh;
    half* sh_B = (half*)(sh + 2*16*8);
        
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
			A_offset += 32;
			
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


__device__ void fp16gemm_16x16_tensor(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float* sh) {

	//__shared__ half sh_A[512];
	//__shared__ half sh_B[512];  // 2*16*16
	
	half* sh_A = (half*)sh;
    half* sh_B = (half*)(sh + 256);	
	
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
	
	// Store the output
    wmma::store_matrix_sync(C + block_base_y + (block_base_x * M), acc_frag, M, wmma::mem_col_major);	
}


/*
// 64 thread
__device__ void fp16gemm_16x16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

	__shared__ half sh_A[256];
	__shared__ half sh_B[256];  // 2*32*8
        
	float2 reg_C[2];
	half2 reg_A[4];
	half2  reg_B[2];
	
	reg_C[0].x = 0.f;
	reg_C[0].y = 0.f;
	reg_C[1].x = 0.f;
	reg_C[1].y = 0.f;
	
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >> 3;
	int im16 = threadIdx.x & 15;
	int id16 = threadIdx.x >> 4;

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*16;
    int block_base_y = blockIdx.x*16;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    half2 *A_start = (half2*)((float2*)(A + block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M));
    *((half2*) (sh_A + 2*threadIdx.x)) = *(A_start);

    //load B from global memory to shared memory
    half2 *B_start = (half2*)((float2*)(B + K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K));
    *((half2*) (sh_B + 2*threadIdx.x)) = *(B_start);

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
			reg_C[0] += (__half22float2)__hmul2(reg_A[0], reg_B[0]);
            //reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            //reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			reg_C[1] += (__half22float2)__hmul2(reg_A[1], reg_B[0]);
			
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
			reg_C[0] += (__half22float2)__hmul2(reg_A[2], reg_B[1]);
			reg_C[1] += (__half22float2)__hmul2(reg_A[3], reg_B[1]);

            B_offset += 32;
        }
		
        double_buffer ^= 128;  // 16*8
		
        if (k+8 < K)
		{
            A_start += 4*M; // half2 --> 8M
            *((half2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
            B_start += 4;
            *((half2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
        }
    }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;  // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    *((float2*)(C + C_offset)) = reg_C[0];
    //C[C_offset+1] = reg_C.y;
	*((float2*)(C + C_offset + 2)) = reg_C[1];
}
*/


__device__ void gemm_64_16x16_1M(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

	float* sh_A = sh;
    float* sh_B = sh + 2*16*8;

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
			*((float4*) reg_A) = *((float4*)(sh_A + A_offset));
            //reg_A[0] = sh_A[A_offset];    // A_offset+0 ~ A_offset+3 为什么不用向量呢
			reg_B[0] = sh_B[B_offset];
            //reg_A[1] = sh_A[A_offset+1];  // 为了避免bank冲突, 这8个寄存器都不是连续的【4个就不会重复】，因此不能使用向量传输指令
            //reg_A[2] = sh_A[A_offset+2];
            //reg_A[3] = sh_A[A_offset+3];
			
			
			
			/*
			ld.shared.f32 	%f21, [%r47+1024];
			ld.shared.f32 	%f22, [%r44];
			fma.rn.f32 	%f23, %f22, %f21, %f96;
			
			asm volatile ("{\t\n"
			// registers to store input operands
			".reg .f32 a1,b2,a3,a4;\n\t"
			".reg .f32 b1,b2;\n\t"
			// loading with vectorized, 128-bit instructions
			"ld.shared.f32 a1,[%0];\n\t"
			"ld.shared.f32 b1,[%1];\n\t"
			// core math operations
			"fma.rn.f32 %2,a1,b1,%2;\n\t"
			:: "l"(sh_A+A_offset),"l"(sh_B+B_offset), ="f"(reg_C.x) : "memory" );
			*/
			
            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); // reg_C.x = reg_A[0]*reg_B[0] + reg_A[4]*reg_B[1]
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
            reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			
			
			*((float4*) (reg_A + 4)) = *((float4*)(sh_A + A_offset + 16));
			reg_B[1] = sh_B[B_offset+1];
			//reg_A[4] = sh_A[A_offset+16];
            //reg_A[5] = sh_A[A_offset+17];
            //reg_A[6] = sh_A[A_offset+18];
            //reg_A[7] = sh_A[A_offset+19];
			A_offset += 32;
			
            reg_C.x = fma(reg_A[4], reg_B[1], reg_C.x);
            reg_C.y = fma(reg_A[5], reg_B[1], reg_C.y);
            reg_C.z = fma(reg_A[6], reg_B[1], reg_C.z);
            reg_C.w = fma(reg_A[7], reg_B[1], reg_C.w);

            B_offset += 32;
        }
		
        double_buffer ^= 128;
		
        if (k+8 < K)
		{
            A_start += 4*M; // float2 --> 8M
            *((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
            B_start += 4;
            *((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
        }
    }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;  // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.x;
    C[C_offset+1] = reg_C.y;
    C[C_offset+2] = reg_C.z;
    C[C_offset+3] = reg_C.w;
}

using namespace nvcuda;

//(N*P*Q)%16==0 && (P*Q)%4==0  每个线程读一部分数据，32个线程一起完成warp层级GEMM计算需要的矩阵A,B和C
__device__ void gemm_32_16x16_tc(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

/*
	half* sh_A = (half*)sh;  // 共享内存有没有用：double buffer
    half* sh_B = (half*)(sh + 2*16*8);

	// Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = N;
    
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
	
    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M);
    *((half2*) (sh_A + 2*threadIdx.x)) = *(A_start);

    //load B from global memory to shared memory
    float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K);
    *((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);

	// Load data
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;  // 初值  wmma::fill_fragment(acc_frag, 0.0f);
    // wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	int aRow = warpM * 16;
	int bCol = warpN * 16;
	// Loop over the K-dimension
    for (int i = 0; i < K; i += 16) 
	{    
        int aCol = i;
        int bRow = i;        
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) 
		{
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, B + bRow + bCol * ldb, ldb);
 
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag); 
		}
    }
	
	// Store the output
    wmma::store_matrix_sync(C + aRow + bCol * ldc, acc_frag, ldc, wmma::mem_col_major);	
	*/
}


//(N*P*Q)%16==0 && (P*Q)%4!=0
__device__ void gemm_64_16x16_2(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

	float* sh_A = sh;
    float* sh_B = sh + 2*16*8;

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

    //load A from global memory to shared memory
    float2 *A_start = (float2*) (A + block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M);
    *((float2*) (sh_A + 2*threadIdx.x)) = *(A_start);

    //load A from global memory to shared memory
    float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K);
    *((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);

    int double_buffer = 0;
#pragma unroll
    for(int k=0; k<K; k+=8){
        __syncthreads();
        int A_offset = double_buffer + (threadIdx.x%4)*4;
        int B_offset = double_buffer + ((threadIdx.x/4)*2);

#pragma unroll
        for (int i=0; i<8; i+=2)
		{
            reg_A[0] = sh_A[A_offset];
            reg_A[1] = sh_A[A_offset+1];
            reg_A[2] = sh_A[A_offset+2];
            reg_A[3] = sh_A[A_offset+3];
            reg_A[4] = sh_A[A_offset+16];
            reg_A[5] = sh_A[A_offset+17];
            reg_A[6] = sh_A[A_offset+18];
            reg_A[7] = sh_A[A_offset+19];

            reg_B[0] = sh_B[B_offset];
            reg_B[1] = sh_B[B_offset+1];

            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x);
            reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
            reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
            reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
            reg_C.x = fma(reg_A[4], reg_B[1], reg_C.x);
            reg_C.y = fma(reg_A[5], reg_B[1], reg_C.y);
            reg_C.z = fma(reg_A[6], reg_B[1], reg_C.z);
            reg_C.w = fma(reg_A[7], reg_B[1], reg_C.w);

            A_offset += 32;
            B_offset += 32;
        }

        double_buffer ^= 128;

        if (k+8 < K){
            A_start += 4*M;
            *((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
            B_start += 4;
            *((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
        }
    }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.x;
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 ;
    C_offset = (ind+1)/(P*Q)*(P*Q*N) + (ind+1)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.y;
    C_offset = (ind+2)/(P*Q)*(P*Q*N) + (ind+2)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.z;
    C_offset = (ind+3)/(P*Q)*(P*Q*N) + (ind+3)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.w;
}

//(N*P*Q%16!=0)
__device__ void gemm_64_16x16_3(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

   float* sh_A = sh;
   float* sh_B = sh + 2*16*8;

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

// half precision
__device__ void fp16gemm_16x16_3(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float* sh) {

	//__shared__ half sh_A[256];
	//__shared__ half sh_B[256];  // 2*16*8
	
	half* sh_A = (half*)sh;
    half* sh_B = (half*)(sh + 16*8);
        
	float2 reg_C[2];
	half2 reg_A[4];
	half2  reg_B[2];
	
	reg_C[0] = {0.f, 0.f};
	reg_C[1] = {0.f, 0.f};
	
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
    int aoffset = block_base_y + (im8 << 1) + (id8)*M;
	if(blockIdx.x == 3)
	{
		if (im8 == 0)
		{
			*(sh_A + thread2) = __float2half_rn(A[aoffset]);
		}
		else
		{
			*(sh_A + thread2) = __float2half_rn(0.f);
		}
		
		*(sh_A + thread2 + 1) = __float2half_rn( 0.f );
	}
	else
	{
		*(sh_A + thread2) = __float2half_rn( A[aoffset] );
		*(sh_A + thread2 + 1) = __float2half_rn( A[aoffset+1] );
	}

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
			A_offset += 32;
			
			reg_C[0] = __half22float2(__hfma2(reg_A[2], reg_B[1], __float22half2_rn(reg_C[0]))) ;
			reg_C[1] = __half22float2(__hfma2(reg_A[3], reg_B[1], __float22half2_rn(reg_C[1]))) ;

            B_offset += 32;
        }
		
        double_buffer ^= 128;  // 16*8
		
        if (k+8 < K)
		{
            aoffset += M << 3; // half2 --> 8M
            //*((half2*) (sh_A + double_buffer + thread2)) = __float22half2_rn(*(A_start));
			//*(sh_A + double_buffer + thread2) = __float2half_rn( A[A_offset%(M*K)] );
			//*(sh_A + double_buffer + thread2 + 1) = __float2half_rn( A[(A_offset+1)%(M*K)] );
			if(blockIdx.x == 3)
			{
				if (im8 == 0)
				{
					*(sh_A + double_buffer + thread2) = __float2half_rn(A[aoffset]);
				}
				else
				{
					*(sh_A + double_buffer + thread2) = __float2half_rn(0.f);
				}
				
				*(sh_A + double_buffer + thread2 + 1) = __float2half_rn(0.f);
			}
			else
			{
				*(sh_A + double_buffer + thread2) = __float2half_rn( A[aoffset] );
				*(sh_A + double_buffer + thread2 + 1) = __float2half_rn( A[aoffset+1] );
			}
			
            B_start += 4;
            *((half2*) (sh_B + double_buffer + thread2)) = __float22half2_rn(*(B_start));
        }
    }

	int ind = block_base_y + (im4<<2);     // 横、纵坐标  M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;
	int PQ = M;
    int C_offset = ind/(PQ)*(PQ*N) + ind%(PQ) + (id4 + block_base_x)*(PQ);

    if (blockIdx.x < M/16)
    {
		// Store the output
		*(C + C_offset) = reg_C[0].x;  // C_offset计算
		*(C + C_offset + 1) = reg_C[0].y;
		//C[C_offset+1] = reg_C.y;
		*(C + C_offset + 2) = reg_C[1].x;
		*(C + C_offset + 3) = reg_C[1].y;
    }
    else
    {
       int ruler = (threadIdx.x%4)*4;
       int rag = M%16;
	   
       if (ruler < rag)
	   {
           C[C_offset] = reg_C[0].x;
	   }
    }
}


__device__ void gemm_64_16x16_3_tensor(int M, int N, int K,  int P, int Q, float *A, float *B, float *C, float* sh){

   //__shared__ half sh_A[512];
   //__shared__ half sh_B[512];
   //__shared__ float sh_C[256];
   
   half* sh_A = (half*)sh;
   half* sh_B = (half*)(sh + 256);
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
   int block_base_x = blockIdx.y<<4;
   int block_base_y = blockIdx.x<<4;

    //load A from global memory to shared memory  sgemm中A和B是分别用两个warp载入的
    float *A_start = (A + block_base_y + (im4 << 2) + (id4)*M);
	if (blockIdx.x == 3)
	{
		if (im4 == 0)
		{
			*(sh_A + thread4) = __float2half_rn(*(A_start));
		}
		else
		{
			*(sh_A + thread4) = __float2half_rn(0.f);
		}
		
		//*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn({0.f, 0.f});
		*(sh_A + thread4 + 1) = __float2half_rn(0.f);
		*(sh_A + thread4 + 2) = __float2half_rn(0.f);
		*(sh_A + thread4 + 3) = __float2half_rn(0.f);
	}
	else
	{
		//*((half2*)(sh_A + thread4)) = __float22half2_rn(*(A_start));
		//*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn(*(A_start + 1));
		*(sh_A + thread4) = __float2half_rn(*(A_start));
		*(sh_A + thread4 + 1) = __float2half_rn(*(A_start+1));
		*(sh_A + thread4 + 2) = __float2half_rn(*(A_start+2));
		*(sh_A + thread4 + 3) = __float2half_rn(*(A_start+3));
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
           A_start += M<<4;
			if (blockIdx.x == 3)
			{
				if (im4 == 0)
				{
					*(sh_A + double_buffer + thread4) = __float2half_rn(*(A_start));
				}
				else
				{
					*(sh_A + double_buffer + thread4) = __float2half_rn(0.f);
				}
				
				//*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn({0.f, 0.f});
				*(sh_A + double_buffer + thread4 + 1) = __float2half_rn(0.f);
				*(sh_A + double_buffer + thread4 + 2) = __float2half_rn(0.f);
				*(sh_A + double_buffer + thread4 + 3) = __float2half_rn(0.f);
			}
			else
			{
				//*((half2*)(sh_A + thread4)) = __float22half2_rn(*(A_start));
				//*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn(*(A_start + 1));
				*(sh_A + double_buffer + thread4) = __float2half_rn(*(A_start));
				*(sh_A + double_buffer + thread4 + 1) = __float2half_rn(*(A_start+1));
				*(sh_A + double_buffer + thread4 + 2) = __float2half_rn(*(A_start+2));
				*(sh_A + double_buffer + thread4 + 3) = __float2half_rn(*(A_start+3));
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
	   //reg_C[1] = sh_C[threadIdx.x*4 + 1];
	   //reg_C[2] = sh_C[threadIdx.x*4 + 2];
	   //reg_C[3] = sh_C[threadIdx.x*4 + 3];
	
       int ruler = (threadIdx.x%4)*4;
       int rag = M%16;
	   
       if (ruler < rag)
	   {
           C[C_offset] = reg_C[0];
	   }
   }
}


__global__ void gemm_2(int M1, int M2, int N1, int N2, int K1, int K2, int P, int Q, float *A1, float *A2, float *B1, float *B2, float *C1, float *C2){

	int id = blockIdx.z;

    extern __shared__ float sh[];

    int M = (id==0)?(M1):(M2);
    int N = (id==0)?(N1):(N2);
    int K = (id==0)?(K1):(K2);
    float *A = (id==0)?(A1):(A2);
    float *B = (id==0)?(B1):(B2);
    float *C = (id==0)?(C1):(C2);

    if (blockIdx.x*16 < (M + (M%16!=0)*16) && blockIdx.y*16 < (N + (N%16!=0)*16)){
   		if (M%16==0 && P%2==0){
   			//(N*P*Q)%16==0 && (P*Q)%4==0
   			gemm_64_16x16_1(M, N, K, P, Q, A, B, C, sh);
   		}
   		else if (M%16==0){
    		//(N*P*Q)%16==0 && (P*Q)%4!=0
   			gemm_64_16x16_2(M, N, K, P, Q, A, B, C, sh);
   		}
   		else{
   			//(N*P*Q%16!=0)
   			gemm_64_16x16_3(M, N, K, P, Q, A, B, C, sh);
    	}
    }
}



__global__ void gemm_4(int M, int N1, int N2, int N3, int N4, 
					   int K, int P, int Q, 
					   float *A1, float *A2, 
					   float *B1, float *B2, float *B3, float *B4,
					   float *C1, float *C2, float *C3, float *C4)
{
	int id = blockIdx.z;
    extern __shared__ float sh[];

    int N;
    float *A, *B, *C;

    switch(id)
	{
		case 0:
			N = N1;
			A = A1;
			B = B1;
			C = C1;
			break;
		case 1:
			N = N2;
			A = A1;
			B = B2;
			C = C2;
			break;
		case 2:
			N = N3;
			A = A1;
			B = B3;
			C = C3;
			break;
		case 3:
			N = N4;
			A = A2;
			B = B4;
			C = C4;
			break;
    }

    if (blockIdx.x*16 < (M + (M%16!=0)*16) && blockIdx.y*16 < (N + (N%16!=0)*16))  // 剔除空的Block
	{
		// N是不是16的整数倍
   		if (M%16 == 0 && P%2 == 0)
		{
   			//(N*P*Q)%16==0 && (P*Q)%4==0
   			gemm_64_16x16_1(M, N, K, P, Q, A, B, C, sh);
			//gemm_32_16x16_tc(M, N, K, P, Q, A, B, C, sh);
			//fp16gemm_16x16_1(M, N, K, P, Q, A, B, C, sh);  // half
			//fp16gemm_16x16_tensor(M, N, K, P, Q, A, B, C, sh);  // tensor
   		}
   		else if (M%16 == 0)
		{
    		//(N*P*Q)%16==0 && (P*Q)%4!=0
   			gemm_64_16x16_2(M, N, K, P, Q, A, B, C, sh);
   		}
   		else
		{
   			//(N*P*Q%16!=0)
   			gemm_64_16x16_3(M, N, K, P, Q, A, B, C, sh);
			// half
			//gemm_64_16x16_3_tensor(M, N, K, P, Q, A, B, C, sh); // tensor
    	}
    }
}


__device__ void gemm_64_16x8_1(int M, int N, int K, float *A, float *B, float *C, float* sh){

	//__shared__ float sh_A[256];
    //__shared__ float sh_B[128];
    float* sh_A = sh;
    float* sh_B = (sh + 256);
   
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


__device__ void gemm_64_16x8_3(int M, int N, int K, float *A, float *B, float *C, float* sh){

	//__shared__ float sh_A[256];
    //__shared__ float sh_B[128];
   float* sh_A = sh;
   float* sh_B = (sh + 256);	

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



__global__ void gemm_4_2(int M, int N1, int N2, int N3, int N4, 
					   int K, int P, int Q, 
					   float *A1, float *A2, 
					   float *B1, float *B2, float *B3, float *B4,
					   float *C1, float *C2, float *C3, float *C4)
{
	int id = blockIdx.z;
    extern __shared__ float sh[];

    int N;
    float *A, *B, *C;

    switch(id)
	{
		case 0:
			N = N1;
			A = A1;
			B = B1;
			C = C1;
			break;
		case 1:
			N = N2;
			A = A1;
			B = B2;
			C = C2;
			break;
		case 2:
			N = N3;
			A = A1;
			B = B3;
			C = C3;
			break;
		case 3:
			N = N4;
			A = A2;
			B = B4;
			C = C4;
			break;
    }

    if (blockIdx.x*16 < (M + (M%16!=0)*16) && blockIdx.y*16 < (N + (N%16!=0)*16))  // 剔除空的Block
	{
		// N是不是16的整数倍
   		if (M%16 == 0 && P%2 == 0)
		{
   			//(N*P*Q)%16==0 && (P*Q)%4==0
   			//gemm_64_16x16_1(M, N, K, P, Q, A, B, C, sh);
			//gemm_32_16x16_tc(M, N, K, P, Q, A, B, C, sh);
			//fp16gemm_16x16_1(M, N, K, P, Q, A, B, C, sh);  // half
			//fp16gemm_16x16_tensor(M, N, K, P, Q, A, B, C, sh);  // tensor
			gemm_64_16x8_1(M, N, K, P, Q, A, B, C, sh); 
   		}
   		else if (M%16 == 0)
		{
    		//(N*P*Q)%16==0 && (P*Q)%4!=0
   			gemm_64_16x16_2(M, N, K, P, Q, A, B, C, sh);
   		}
   		else
		{
   			//(N*P*Q%16!=0)
   			//gemm_64_16x16_3(M, N, K, P, Q, A, B, C, sh);
			//fp16gemm_16x16_3(M, N, K, P, Q, A, B, C, sh); // half
			//gemm_64_16x16_3_tensor(M, N, K, P, Q, A, B, C, sh); // tensor
			gemm_64_16x8_3(M, N, K, P, Q, A, B, C, sh); 
    	}
    }
}



#endif /* GEMM_KERNEL_H_ */
