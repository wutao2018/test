template<int kThreads>
__global__ void gemm(int M[], int N[], int K[], float *A[], float *B[], float *C[]);


template<>
__global__ void gemm<128>(int M[], int N[], int K[], float *A[], float *B[], float *C[]){
	
	int ii = blockIdx.z;
	extern __shared__ float sh[]; // 用作什么 kernel第三个参数分配的吗
	// int t = T_strategy[i];

	float *sh_A = sh;
	float *sh_B = sh + 2*32*8;

	float4 reg_C[2];
	float4 reg_A;
	float  reg_B[2];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*32;
	int block_base_y = blockIdx.x*32;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C[ii] + block_base_x*M[ii] + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M[ii]);

	reg_C[0] = *C_start;
	reg_C[1] = *(C_start + 4*M[ii]);

	//load A from global memory to shared memory
	float2 *A_start = (float2*) (A[ii] + block_base_y + (threadIdx.x%16)*2 + (threadIdx.x/16)*M[ii]);
	*((float2*)(sh_A + 2*threadIdx.x)) = *(A_start);

	//load B from global memory to shared memory
	float2 *B_start = (float2*) (B[ii] + K[ii]*block_base_x + (threadIdx.x/32)*2 + (threadIdx.x%32)*K[ii]);
	*((float2*)(sh_B + 2*threadIdx.x)) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K[ii]; k+=8)
	{
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*4;
		int B_offset = double_buffer + (threadIdx.x/8)*2;
			
#pragma unroll
		for (int i=0; i<8; i++)	
		{
			reg_A.x = sh_A[A_offset];
			reg_A.y = sh_A[A_offset+1];
			reg_A.z = sh_A[A_offset+2];
			reg_A.w = sh_A[A_offset+3];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+32];

			reg_C[0].x = fma(reg_A.x, reg_B[0], reg_C[0].x);
			reg_C[0].y = fma(reg_A.y, reg_B[0], reg_C[0].y);
			reg_C[0].z = fma(reg_A.z, reg_B[0], reg_C[0].z);
			reg_C[0].w = fma(reg_A.w, reg_B[0], reg_C[0].w);
			reg_C[1].x = fma(reg_A.x, reg_B[1], reg_C[1].x);
			reg_C[1].y = fma(reg_A.y, reg_B[1], reg_C[1].y);
			reg_C[1].z = fma(reg_A.z, reg_B[1], reg_C[1].z);
			reg_C[1].w = fma(reg_A.w, reg_B[1], reg_C[1].w);

			A_offset += 32;
			B_offset += ((i%2)*62 + 1);
		}

		double_buffer ^= 256;

		if (k+8 < K[ii])
		{
			A_start += 4*M[ii]; 
			*((float2*)(sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
			B_start += 4; 
			*((float2*)(sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
		}
	}
	
    *C_start = reg_C[0];
    *(C_start + 4*M[ii]) = reg_C[1];

	return;
}

