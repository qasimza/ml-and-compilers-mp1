#include "../include/utils.h"
#include <cuda_runtime.h>

#define NUM_RUNS 10

#define CUDA_CHECK(func)                                                     	   \
	do {                                                                           \
		cudaError_t status = (func);                                               \
		if (status != cudaSuccess) {                                               \
			printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,   \
				cudaGetErrorString(status), status);                               \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

#define CHECK(name) \
	float *d_Aref_ ## name, *d_Bref_ ## name, *d_Cref_ ## name; \
	std::cerr << "checking " << #name << std::endl; \
	CUDA_CHECK(cudaMalloc(&d_Aref_ ## name, Ref::M * Ref::K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Bref_ ## name, Ref::K * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Cref_ ## name, Ref::M * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_Aref_ ## name, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_Bref_ ## name, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	float* d_Cref_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < Ref::M; i++) { \
		for (int j = 0; j < Ref::N; j++) { \
			d_Cref_INI_ ## name[i * Ref::N + j] = 0; \
		} \
	} \
	CUDA_CHECK(cudaMemcpy(d_Cref_ ## name, d_Cref_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	name(d_Aref_ ## name, d_Bref_ ## name, d_Cref_ ## name, Ref::M, Ref::N, Ref::K); \
	cudaError_t err_c_ ## name = cudaGetLastError(); \
	if (err_c_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_ ## name) << std::endl; \
	} \
	CUDA_CHECK(cudaMemcpy(refC, d_Cref_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost)); \
	if (!ref.checkRef(refC)){ \
		std::cerr << "check ref failed!" << std::endl; \
	};

#define TIME(name) \
	float *d_A_ ## name, *d_B_ ## name, *d_C_ ## name; \
	CUDA_CHECK(cudaMalloc(&d_A_ ## name, M * K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_B_ ## name, K * N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_C_ ## name, M * N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_A_ ## name, A, M * K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_B_ ## name, B, K * N * sizeof(float), cudaMemcpyHostToDevice)); \
	cudaEvent_t start_ ## name, end_ ## name; \
	cudaEventCreate(&start_ ## name); \
	cudaEventCreate(&end_ ## name); \
	float* d_C_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < M; i++) { \
		for (int j = 0; j < N; j++) { \
			d_C_INI_ ## name[i * N + j] = 0; \
		} \
	} \
	for (int i = 0; i < 2; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
	} \
	cudaError_t err_t_ ## name = cudaGetLastError(); \
	if (err_t_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_t_ ## name) << std::endl; \
	} \
	float milliseconds_ ## name = 0; \
	for (int i = 0; i < NUM_RUNS; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		cudaDeviceSynchronize(); \
		cudaEventRecord(start_ ## name); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
		cudaEventRecord(end_ ## name); \
		cudaEventSynchronize(end_ ## name); \
		float milliseconds_ ## i = 0; \
		cudaEventElapsedTime(&milliseconds_ ## i, start_ ## name, end_ ## name); \
		milliseconds_ ## name += milliseconds_ ## i; \
	} \
	cudaMemcpy(C, d_C_ ## name, M * N * sizeof(float), cudaMemcpyDeviceToHost); \
	std::cout << "Time taken for GEMM (GPU, " << #name <<"): " << milliseconds_ ## name / (float)NUM_RUNS << "ms" << std::endl; \
	cudaFree(d_A_ ## name); \
	cudaFree(d_B_ ## name); \
	cudaFree(d_C_ ## name);

__global__ void gemm_gpu_o0_kernel(float* A, float* B, float *C, int M, int N, int K) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < K; k++) {
					C[i * N + j]  += A[i * K + k]  * B[k * N + j];
				}
			}
		}
    }
}

void gemm_gpu_o0(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(1);
	dim3 gridSize(1);
	gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// Parallelize the kernel across multiple Streaming Multiprocessors (SM) and thread blocks.
// Find a set of suitable kernel launch parameters. 
// Note that the starter code does all computations in one SM.

__global__ void gemm_gpu_o1_kernel(float* A, float* B, float *C, int M, int N, int K) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= M || col >= N) {
		return;
	} else {
		float sum = 0.0f;
		for (int k = 0; k < K; k++) {
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] += sum;
	}
}

void gemm_gpu_o1(float* A, float* B, float* C, int M, int N, int K)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
	              (M + blockSize.y - 1) / blockSize.y);
	gemm_gpu_o1_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}


// Tile your implementation to maximize data reuse. Please use GPU shared memory to load and store the tiles
// used for computation. Also, think about memory coalescing when you develop the tiled code.

#define O2_TILE_SIZE 16

__global__ void gemm_gpu_o2_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float As[O2_TILE_SIZE][O2_TILE_SIZE];
	__shared__ float Bs[O2_TILE_SIZE][O2_TILE_SIZE];

	int col = blockIdx.x * O2_TILE_SIZE + threadIdx.x;
	int row = blockIdx.y * O2_TILE_SIZE + threadIdx.y;
	float sum = 0.0f;
	int num_tiles = (K + O2_TILE_SIZE - 1) / O2_TILE_SIZE;

	for (int t = 0; t < num_tiles; t++) {
		int a_col = t * O2_TILE_SIZE + threadIdx.x;
		int b_row = t * O2_TILE_SIZE + threadIdx.y;

		if (row < M && a_col < K) {
			As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
		} else {
			As[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (b_row < K && col < N) {
			Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
		} else {
			Bs[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int k = 0; k < O2_TILE_SIZE; k++) {
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] += sum;
	}
}

void gemm_gpu_o2(float* A, float* B, float* C, int M, int N, int K)
{
	dim3 blockSize(O2_TILE_SIZE, O2_TILE_SIZE);
	dim3 gridSize((N + O2_TILE_SIZE - 1) / O2_TILE_SIZE,
	              (M + O2_TILE_SIZE - 1) / O2_TILE_SIZE);
	gemm_gpu_o2_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// Empirically try out multiple kernel launch parameters and find out a performant set of parameters that utilizes
// the GPU parallelism better. Note that you are not required to find the best.
// o2 is left untouched (fixed 16x16). Launch-parameter experiments (8 / 16 / 32 / 64) are o3 only.
template<int TILE>
__global__ void gemm_gpu_o3_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float As[TILE][TILE];
	__shared__ float Bs[TILE][TILE];

	int col = blockIdx.x * TILE + threadIdx.x;
	int row = blockIdx.y * TILE + threadIdx.y;
	float sum = 0.0f;
	int num_tiles = (K + TILE - 1) / TILE;

	for (int t = 0; t < num_tiles; t++) {
		int a_col = t * TILE + threadIdx.x;
		int b_row = t * TILE + threadIdx.y;

		if (row < M && a_col < K) {
			As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
		} else {
			As[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (b_row < K && col < N) {
			Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
		} else {
			Bs[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int k = 0; k < TILE; k++) {
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] += sum;
	}
}

template<int TILE>
void gemm_gpu_o3_launch(float* A, float* B, float* C, int M, int N, int K)
{
	dim3 blockSize(TILE, TILE);
	dim3 gridSize((N + TILE - 1) / TILE,
	              (M + TILE - 1) / TILE);
	gemm_gpu_o3_kernel<TILE><<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

void gemm_gpu_o3_8(float* A, float* B, float* C, int M, int N, int K) {
	gemm_gpu_o3_launch<8>(A, B, C, M, N, K);
}

void gemm_gpu_o3_16(float* A, float* B, float* C, int M, int N, int K) {
	gemm_gpu_o3_launch<16>(A, B, C, M, N, K);
}

void gemm_gpu_o3_32(float* A, float* B, float* C, int M, int N, int K) {
	gemm_gpu_o3_launch<32>(A, B, C, M, N, K);
}

void gemm_gpu_o3_64(float* A, float* B, float* C, int M, int N, int K) {
	gemm_gpu_o3_launch<64>(A, B, C, M, N, K);
}

// Default o3 = 32x32 after trying 8 / 16 / 32. Change this if another size wins.
void gemm_gpu_o3(float* A, float* B, float* C, int M, int N, int K)
{
	gemm_gpu_o3_launch<32>(A, B, C, M, N, K);
}



int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
		return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	// int runs = atoi(argv[3]);
	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	/// GPU Implementation
        // Check if implementation is correct
	auto ref = Ref();
	float* refC = new float[Ref::M * Ref::N]();
 	//CHECK(gemm_gpu_o0)
	CHECK(gemm_gpu_o1)
	CHECK(gemm_gpu_o2)
	CHECK(gemm_gpu_o3_8)
	CHECK(gemm_gpu_o3_16)
	CHECK(gemm_gpu_o3_32)
	CHECK(gemm_gpu_o3_64)

	// Actual run
 	//TIME(gemm_gpu_o0)
	TIME(gemm_gpu_o1)
	TIME(gemm_gpu_o2)
	TIME(gemm_gpu_o3_8)
	TIME(gemm_gpu_o3_16)
	TIME(gemm_gpu_o3_32)
	TIME(gemm_gpu_o3_64)

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}