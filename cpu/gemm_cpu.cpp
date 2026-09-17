#include <chrono>
#include "../include/utils.h"

#define NUM_RUNS 2

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  initialize(refC, Ref::M * Ref::N);				\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								
  
#define TIME(name) \
  for (int i = 0; i < 1; i++)						\
    {									\
      name(A, B, C, M, N, K);						\
    }									\
  std::chrono::duration<double, std::milli> time_##name(0);		\
  for (int i = 0; i < NUM_RUNS; i++)					\
    {									\
      initialize(C, M * N);						\
      auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K);						\
      auto end_time_ ## name = std::chrono::high_resolution_clock::now(); \
      time_ ## name += end_time_ ## name - start_time_ ## name;		\
    }									\
std::chrono::duration<double, std::milli> duration_ ## name = time_ ## name/float(NUM_RUNS); \
  std::cout << "Time taken for GEMM (CPU," << #name <<"): " << duration_ ## name.count() << "ms" << std::endl; 


// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
__attribute__((target("no-fma")))
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
	      C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler flags. So, you can use o3's code for that part

// Optimal loop order for this kernel based on data locality
// when loops are iterating through values of input and output matrices
__attribute__((target("no-fma")))
void gemm_cpu_o1(float* A, float* B, float *C, int M, int N, int K) {
  // i-k-j: B and C are walked along rows (contiguous in row-major)
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Tiled version of the kernel, where the inner two loops are transformed
// Tiling factor fits accessed data into the L1 cache of the computer
__attribute__((target("no-fma")))
void gemm_cpu_o2(float* A, float* B, float *C, int M, int N, int K) {

  const int T = 32;
  for (int ii = 0; ii < M; ii += T) {
    
    int iend = (ii + T < M) ? ii + T : M;

    for (int kk = 0; kk < K; kk += T) {

      int kend = (kk + T < K) ? kk + T : K;

      for (int jj = 0; jj < N; jj += T) {

        int jend = (jj + T < N) ? jj + T : N;

        for (int i = ii; i < iend; i++) {
          for (int k = kk; k < kend; k++) {
            for (int j = jj; j < jend; j++) {
              C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
          }
        }
      }
    }
  }
}


// Parallelizing the outer loop(s) using OpenMP 
// Vectorize the inner loop using suitable compiler flags
__attribute__((optimize("O3"), target("no-fma")))
void gemm_cpu_o3(float* A, float* B, float *C, int M, int N, int K) {

    const int T = 32;

    // parallelizing the outer loop(s) using OpenMP 
    #pragma omp parallel for
    for (int ii = 0; ii < M; ii += T) {
    
      int iend = (ii + T < M) ? ii + T : M;
  
      for (int kk = 0; kk < K; kk += T) {
  
        int kend = (kk + T < K) ? kk + T : K;
  
        for (int jj = 0; jj < N; jj += T) {
  
          int jend = (jj + T < N) ? jj + T : N;
  
          for (int i = ii; i < iend; i++) {
            for (int k = kk; k < kend; k++) {
              for (int j = jj; j < jend; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
              }
            }
          }
        }
      }
    }
  }

// o4 = o3 + FMA (o3 already has -O3; o0–o2 do not)
__attribute__((optimize("O3"), target("fma")))
void gemm_cpu_o4(float* A, float* B, float *C, int M, int N, int K) {

  const int T = 32;

  // parallelizing the outer loop(s) using OpenMP 
  #pragma omp parallel for
  for (int ii = 0; ii < M; ii += T) {
    
    int iend = (ii + T < M) ? ii + T : M;

    for (int kk = 0; kk < K; kk += T) {

      int kend = (kk + T < K) ? kk + T : K;

      for (int jj = 0; jj < N; jj += T) {

        int jend = (jj + T < N) ? jj + T : N;

        for (int i = ii; i < iend; i++) {
          for (int k = kk; k < kend; k++) {
            for (int j = jj; j < jend; j++) {
              C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
          }
        }
      }
    }
  }
}


int main(int argc, char* argv[]) {
	if (argc < 3) {
	  std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
	  return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	// Check if the kernel results are correct
	// note that even if the correctness check fails all optimized kernels will run.
	// We are not exiting the program at failure at this point.
	// It is a good idea to add more correctness checks to your code.
	// We may (at discretion) verify that your code is correct.
	float* refC = new float[Ref::M * Ref::N]();
	auto ref = Ref();
	//CHECK(gemm_cpu_o0)
	CHECK(gemm_cpu_o1)
	CHECK(gemm_cpu_o2)
	CHECK(gemm_cpu_o3)
  CHECK(gemm_cpu_o4)
	delete[] refC;
	
	//TIME(gemm_cpu_o0)
	TIME(gemm_cpu_o1)
	TIME(gemm_cpu_o2)
	TIME(gemm_cpu_o3)
	TIME(gemm_cpu_o4)

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}