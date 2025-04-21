#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

void printCudaMemoryInfo() {
  size_t free_mem = 0;
  size_t total_mem = 0;
  CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
  std::cout << "Total memory: " << total_mem / (1024 * 1024) << ", Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
}

void testCudaMallocLimits(bool verbose) {
  size_t size = 1024 * 1024 * 1024; // Start with 1 GB
  float* d_ptr = nullptr;

  while (true) {
    // Print CUDA memory info before allocation
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err == cudaSuccess) {
      if (verbose) {
        std::cout << "CUDA memory info after allocation:" << std::endl;
        printCudaMemoryInfo();
      }
      std::cout << "cudaMalloc succeeded for size: " << size / (1024 * 1024) << " MB" << std::endl;
      if (verbose) {
        std::cout << "cudaFree" << std::endl;
      }
      CHECK_CUDA(cudaFree(d_ptr));
      size += 1024 * 1024 * 1024; // Increase by 1 GB
      // Print CUDA memory info after allocation
      if (verbose) {
        std::cout << "CUDA memory info after deallocation:" << std::endl;
        printCudaMemoryInfo();
      }
    } else {
      std::cerr << "cudaMalloc failed for size: " << size / (1024 * 1024) << " MB" << std::endl;
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
      break;
    }
  }
}

int main() {
  std::cout << "CUDA memory info before any allocation:" << std::endl;
  printCudaMemoryInfo();

  // Test CUDA malloc limits
  testCudaMallocLimits(false);

  return 0;
}
