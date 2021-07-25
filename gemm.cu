/*
 * gemm_kernel.h
 *
 */

#ifndef GEMM_KERNEL_H_
#define GEMM_KERNEL_H_

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;


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

    //load A from global memory to shared memory
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
			
            reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x); 
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

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;  // M=HW， K = C， N = K
	// blockIdx.x*16 < (M + (0)*16) ;  M%16 == 0 && P%2 == 0 ;   relu = max(0, x)
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.x;
    C[C_offset+1] = reg_C.y;
    C[C_offset+2] = reg_C.z;
    C[C_offset+3] = reg_C.w;
}

// 64 thread, 
__device__ void fp16gemm_16x16_tensor(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float* sh) {
	
	half* sh_A = (half*)sh;
    half* sh_B = (half*)(sh + 256);	
	
	int im8 = threadIdx.x & 7;
	int id8 = threadIdx.x >> 3;
	
	int thread2 = threadIdx.x << 1;

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y << 4;
    int block_base_y = blockIdx.x << 4;

    //load A from global memory to shared memory 
    float2 *A_start = (float2*) (A + block_base_y + (im8 << 1) + (id8)*M);
    *((half2*)(sh_A + thread2)) = __float22half2_rn(*(A_start));
	*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn(*(A_start + 4*M));

    //load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (im8 << 1) + (id8)*K);
    *((half2*) (sh_B + thread2)) = __float22half2_rn(*(B_start));
	*((half2*) (sh_B + thread2 + 128)) = __float22half2_rn(*(B_start + 4*K));

    int double_buffer = 0;
	
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   
#pragma unroll
    for(int k = 0; k < K; k += 16)
	{
        __syncthreads();
	
		if (thread.Idx.x < 32)
		{
			// Load the inputs
			wmma::load_matrix_sync(a_frag, sh_A + double_buffer, 16);
			wmma::load_matrix_sync(b_frag, sh_B + double_buffer, 16);
	 
			// Perform the matrix multiplication
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);		
		}
        double_buffer ^= 256;  // 16*16
		
        if (k + 16 < K)
		{
            A_start += M << 3; // half2 --> 8M
            *((half2*) (sh_A + double_buffer + thread2)) = __float22half2_rn(*(A_start));
			*((half2*) (sh_A + double_buffer + thread2 + 128)) = __float22half2_rn(*(A_start+4*M));
			
            B_start += 8;
            *((half2*) (sh_B + double_buffer + thread2)) = __float22half2_rn(*(B_start));
			*((half2*) (sh_B + double_buffer + thread2 + 128)) = __float22half2_rn(*(B_start+4*K));
        }
    }
	
	// Store the output
    if (threadIdx.x < 32)
    {
		for (int i=0; i < acc_frag.num_elements; i++)
			acc_frag.x[i] = acc_frag.x[i] > 0.f ? acc_frag.x[i]:0.f;
		wmma::store_matrix_sync(C + block_base_y + (block_base_x * M), acc_frag, M, wmma::mem_col_major);	
	}
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
   int A_offset = block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M; 
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
       for (int i=0; i<8; i+=2){  

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


__device__ void gemm_64_16x16_3_tensor(int M, int N, int K,  int P, int Q, float *A, float *B, float *C, float* sh){
   
   half* sh_A = (half*)sh;
   half* sh_B = (half*)(sh + 256);
   __shared__ float sh_C[256];   

   float reg_C[4];
   //reg_C[0 = {0.f, 0.f,0.f,0.f};

   int im8 = threadIdx.x & 7;
   int id8 = threadIdx.x >> 3;
   int thread2 = threadIdx.x << 1;
   
   // Compute block's starting coordinate
   int block_base_x = blockIdx.y<<4;
   int block_base_y = blockIdx.x<<4;

    //load A from global memory to shared memory 
    float *A_start = (A + block_base_y + (im8 << 1) + (id8)*M);
	if (blockIdx.x == 3)
	{
		if (im8 == 0)
		{
			*(sh_A + thread2) = __float2half_rn(*(A_start));
			*(sh_A + thread2+128) = __float2half_rn(*(A_start+8*M));
		}
		
		*(sh_A + thread2 + 1) = __float2half_rn(0.f);
		*(sh_A + thread2 + 129) = __float2half_rn(0.f);
	}
	else
	{
		*(sh_A + thread2) = __float2half_rn(*(A_start));
		*(sh_A + thread2 + 1) = __float2half_rn(*(A_start+1));
		*(sh_A + thread2 + 128) = __float2half_rn(*(A_start+8*M));
		*(sh_A + thread2 + 129) = __float2half_rn(*(A_start+8*M+1));
	}

    //load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (im8 << 1) + (id8)*K);
    *((half2*) (sh_B + thread2)) = __float22half2_rn(*(B_start));
	*((half2*) (sh_B + thread2 + 128)) = __float22half2_rn(*(B_start + 4*K));

   int double_buffer = 0;
   
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   
#pragma unroll
   for(int k=0; k<K; k+=16)
   {
       __syncthreads();
	   
		// Load the inputs
		if(threadIdx.x < 32){
			wmma::load_matrix_sync(a_frag, sh_A + double_buffer, 16);
			wmma::load_matrix_sync(b_frag, sh_B + double_buffer, 16);
	 
			// Perform the matrix multiplication
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);		   
		}
		
       double_buffer ^= 256;

       if (k+16 < K)
	   {
           A_start += M<<4;
			if (blockIdx.x == 3)
			{
				if (im8 == 0)
				{
					*(sh_A + double_buffer + thread2) = __float2half_rn(*(A_start));
					*(sh_A + double_buffer + thread2+128) = __float2half_rn(*(A_start+8*M));
				}
				
				*(sh_A + double_buffer + thread2 + 1) = __float2half_rn(0.f);
				*(sh_A + double_buffer + thread2 + 129) = __float2half_rn(0.f);
			}
			else
			{
				*(sh_A + double_buffer + thread2) = __float2half_rn(*(A_start));
				*(sh_A + double_buffer + thread2 + 1) = __float2half_rn(*(A_start+1));
				*(sh_A + double_buffer + thread2 + 128) = __float2half_rn(*(A_start+8*M));
				*(sh_A + double_buffer + thread2 + 129) = __float2half_rn(*(A_start+8*M+1));
			}
			
            B_start += 8;
			*((half2*) (sh_B + double_buffer  + thread2)) = __float22half2_rn(*(B_start));
			*((half2*) (sh_B + double_buffer  + thread2 + 128)) = __float22half2_rn(*(B_start + 4*K));
       }
   }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
	int PQ = M; 
    int C_offset = ind/(PQ)*(PQ*N) + ind%(PQ) + (threadIdx.x/4)*(PQ) + blockIdx.y*16*(PQ);

   if (blockIdx.x < M/16)
   {
		// Store the output
		if(threadIdx.x < 32)
		{
			for (int i=0; i < acc_frag.num_elements; i++)
				acc_frag.x[i] = acc_frag.x[i] > 0.f ? acc_frag.x[i]:0.f;
			wmma::store_matrix_sync(C + block_base_y + (block_base_x * M), acc_frag, M, wmma::mem_col_major);
		}
   }
   else
   {
		// Store the output
		if(threadIdx.x < 32)
			wmma::store_matrix_sync(sh_C, acc_frag, 16, wmma::mem_col_major);
		
		__syncthreads();
	   
        if (threadIdx.x%4 == 0)
	    {
	       reg_C[0] = sh_C[threadIdx.x*4];
           C[C_offset] = reg_C[0] > 0.f ? reg_C[0]:0.f;
	    }
   }
}


__device__ void gemm_64_16x16_3_tensor0(int M, int N, int K,  int P, int Q, float *A, float *B, float *C, float* sh){
   
   half* sh_A = (half*)sh;
   half* sh_B = (half*)(sh + 256);
   __shared__ float sh_C[256];   

   float4 reg_C;
   //reg_C = {0.f, 0.f,0.f,0.f};

   int im8 = threadIdx.x & 7;
   int id8 = threadIdx.x >> 3;
   int thread2 = threadIdx.x << 1;
   //int nb = N%16;
   
   // Compute block's starting coordinate
   int block_base_x = blockIdx.y<<4;
   int block_base_y = blockIdx.x<<4;

    //load A from global memory to shared memory 
    //float2 *A_start = (float2*) (A + block_base_y + (im8 << 1) + (id8)*M);
    //*((half2*)(sh_A + thread2)) = __float22half2_rn(*(A_start));
	//*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn(*(A_start + 4*M));

    //load A from global memory to shared memory 
    float2 *A_start = (float2*)(A + block_base_y + (im8 << 1) + (id8)*M);
	if (blockIdx.x == 12)
	{
		if ((im8 >> 1) == 0)  //((im8 == 0) || (im8 == 1))
		{
			//*(sh_A + thread2) = __float2half_rn(*(A_start));
			//*(sh_A + thread2+128) = __float2half_rn(*(A_start+8*M));
			*((half2*)(sh_A + thread2)) = __float22half2_rn(*(A_start));
			*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn(*(A_start + 4*M));
		}
		else
		{
			*((half2*)(sh_A + thread2)) = __float22half2_rn({0.f,0.f});
			*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn({0.f,0.f});			
		}
		//*(sh_A + thread2 + 1) = __float2half_rn(0.f);
		//*(sh_A + thread2 + 129) = __float2half_rn(0.f);
	}
	else
	{
		//*(sh_A + thread2) = __float2half_rn(*(A_start));
		//*(sh_A + thread2 + 1) = __float2half_rn(*(A_start+1));
		//*(sh_A + thread2 + 128) = __float2half_rn(*(A_start+8*M));
		//*(sh_A + thread2 + 129) = __float2half_rn(*(A_start+8*M+1));
		*((half2*)(sh_A + thread2)) = __float22half2_rn(*(A_start));
		*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn(*(A_start + 4*M));		
	}

    //load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (im8 << 1) + (id8)*K);
	if ((24 == N) && (blockIdx.y == 1))
	{
		//if (blockIdx.y == 1)
		//{
			*((half2*) (sh_B + thread2)) = __float22half2_rn(*(B_start));
			*((half2*) (sh_B + thread2 + 128)) = __float22half2_rn({0.f, 0.f});		
		//}
		//else
		//{
		//	*((half2*) (sh_B + thread2)) = __float22half2_rn(*(B_start));
		//	*((half2*) (sh_B + thread2 + 128)) = __float22half2_rn(*(B_start + 4*K));			
		//}
	}
	else
	{
		*((half2*) (sh_B + thread2)) = __float22half2_rn(*(B_start));
		*((half2*) (sh_B + thread2 + 128)) = __float22half2_rn(*(B_start + 4*K));
	}

   int double_buffer = 0;
   
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   
#pragma unroll
   for(int k=0; k<K; k+=16)
   {
       __syncthreads();
	   
		// Load the inputs
		if(threadIdx.x < 32){
			wmma::load_matrix_sync(a_frag, sh_A + double_buffer, 16);
			wmma::load_matrix_sync(b_frag, sh_B + double_buffer, 16);
	 
			// Perform the matrix multiplication
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);		   
		}
		
       double_buffer ^= 256;

       if (k+16 < K)
	   {
           A_start += M<<3;
		   //A_start += M<<4;
			if (blockIdx.x == 12)
			{
				if ((im8 >> 1) == 0)
				{
					//*(sh_A + double_buffer + thread2) = __float2half_rn(*(A_start));
					//*(sh_A + double_buffer + thread2+128) = __float2half_rn(*(A_start+8*M));
					*((half2*)(sh_A + thread2)) = __float22half2_rn(*(A_start));
					*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn(*(A_start + 4*M));					
				}
				else
				{
					*((half2*)(sh_A + thread2)) = __float22half2_rn({0.f, 0.f});
					*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn({0.f, 0.f});					
				}
				
				//*(sh_A + double_buffer + thread2 + 1) = __float2half_rn(0.f);
				//*(sh_A + double_buffer + thread2 + 129) = __float2half_rn(0.f);
			}
			else
			{
				//*(sh_A + double_buffer + thread2) = __float2half_rn(*(A_start));
				//*(sh_A + double_buffer + thread2 + 1) = __float2half_rn(*(A_start+1));
				//*(sh_A + double_buffer + thread2 + 128) = __float2half_rn(*(A_start+8*M));
				//*(sh_A + double_buffer + thread2 + 129) = __float2half_rn(*(A_start+8*M+1));
				*((half2*)(sh_A + thread2)) = __float22half2_rn(*(A_start));
				*((half2*)(sh_A + thread2 + 128)) = __float22half2_rn(*(A_start + 4*M));				
			}
			
            B_start += 8;
			if ((24 == N) && (blockIdx.y == 1))
			{
				*((half2*) (sh_B + double_buffer  + thread2)) = __float22half2_rn(*(B_start));
				*((half2*) (sh_B + double_buffer  + thread2 + 128)) = __float22half2_rn({0.f, 0.f});
			}
			else
			{
				*((half2*) (sh_B + double_buffer  + thread2)) = __float22half2_rn(*(B_start));
				*((half2*) (sh_B + double_buffer  + thread2 + 128)) = __float22half2_rn(*(B_start + 4*K));				
			}
       }
   }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
    //int C_offset = ind%(M) + (threadIdx.x/4)*(M) + blockIdx.y*16*(M);
	int C_offset = ind%(M) + (threadIdx.x/4)*(M) + block_base_x*16*(M);
	block_base_x

   if (blockIdx.x < M/16) && (blockIdx.y < N/16)
   {
		// Store the output
		if(threadIdx.x < 32)
		{
			for (int i=0; i < acc_frag.num_elements; i++)
				acc_frag.x[i] = acc_frag.x[i] > 0.f ? acc_frag.x[i]:0.f;
			wmma::store_matrix_sync(C + block_base_y + (block_base_x * M), acc_frag, M, wmma::mem_col_major);
		}
   }
   else
   {
		// Store the output
		if(threadIdx.x < 32)
			wmma::store_matrix_sync(sh_C, acc_frag, 16, wmma::mem_col_major);
		
		__syncthreads();
	   
        if (threadIdx.x%4 == 0)
	    {
		   if (N !=24 || blockIdx.y ！= 1) {
			   reg_C = *((float4*)(sh_C + threadIdx.x*4));
			   C[C_offset] = reg_C.x > 0.f ? reg_C.x:0.f;
			   C[C_offset+1] = reg_C.y > 0.f ? reg_C.y:0.f;
			   C[C_offset+2] = reg_C.z > 0.f ? reg_C.z:0.f;
			   C[C_offset+3] = reg_C.w > 0.f ? reg_C.w:0.f;
		   }
	    }
   }
}

__device__ void fp16gemm_16x16_tensor0(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float* sh) {
	
	half* sh_A = (half*)sh;
    half* sh_B = (half*)(sh + 256);	
	
	int im4 = threadIdx.x & 3;
	int id4 = threadIdx.x >> 2;
	
	int thread4 = threadIdx.x << 2;

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y << 4;
    int block_base_y = blockIdx.x << 4;

    //load A from global memory to shared memory
    float2 *A_start = (float2*) (A + block_base_y + (im4 << 2) + (id4)*M);
    *((half2*)(sh_A + thread4)) = __float22half2_rn(*(A_start));
	*((half2*)(sh_A + thread4 + 2)) = __float22half2_rn(*(A_start + 1));

    //load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (im4 << 2) + (id4)*K);
    *((half2*) (sh_B + thread4)) = __float22half2_rn(*(B_start));
	*((half2*) (sh_B + thread4 + 2)) = __float22half2_rn(*(B_start + 1));

    int double_buffer = 0;
	
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   
#pragma unroll
    for(int k = 0; k < K; k += 16)
	{
        __syncthreads();
		
		if (thread.Idx.x < 32)
		{	
			// Load the inputs
			wmma::load_matrix_sync(a_frag, sh_A + double_buffer, 16);
			wmma::load_matrix_sync(b_frag, sh_B + double_buffer, 16);
	 
			// Perform the matrix multiplication
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);	
		}

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
	if (thread.Idx.x < 32)
	{
		for (int i=0; i < acc_frag.num_elements; i++)
			acc_frag.x[i] = acc_frag.x[i] > 0.f ? acc_frag.x[i]:0.f;		
		wmma::store_matrix_sync(C + block_base_y + (block_base_x * M), acc_frag, M, wmma::mem_col_major);
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

    if (blockIdx.x*16 < (M + (M%16!=0)*16) && blockIdx.y*16 < (N + (N%16!=0)*16))
	{
		
   		if (M%16 == 0 && P%2 == 0)
		{
   			//(N*P*Q)%16==0 && (P*Q)%4==0
   			gemm_64_16x16_1(M, N, K, P, Q, A, B, C, sh);
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
		// 
   		if (M == 784)
		{
			fp16gemm_16x16_tensor(M, N, K, P, Q, A, B, C, sh);  // tensor
   		}
   		else if (M == 196)
		{
			fp16gemm_16x16_tensor0(M, N, K, P, Q, A, B, C, sh);  // tensor
   		}
   		else
		{
			gemm_64_16x16_3_tensor(M, N, K, P, Q, A, B, C, sh); // tensor
    	}
    }
}



#endif /* GEMM_KERNEL_H_ */
