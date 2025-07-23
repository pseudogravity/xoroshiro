// nvcc -std=c++17 -O3 -arch=sm_86 -use_fast_math -o decorationreverse decorationreverse.cu

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cstring>           // strcmp()
#include <vector>            // std::vector
#include <cuda_runtime.h>
#include <chrono>
#include <inttypes.h>
#include <iostream>

// Constants
constexpr unsigned long long THREAD_SIZE         = 512;
constexpr unsigned long long BLOCK_SIZE          = 1ULL << 23;
constexpr unsigned long long BATCH_SIZE          = BLOCK_SIZE * THREAD_SIZE;
constexpr int               RESULTS_BUFFER_SIZE  = 8;

constexpr uint64_t XL = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t XH = 0x6A09E667F3BCC909ULL;

// Structs
struct Result {
    uint64_t seed;
    int64_t  a, b;
};

// Utils
static void gpuAssert(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d — %s\n",
                     file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

// Xoroshiro impl
__device__ __forceinline__ uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    const uint64_t M1 = 0xBF58476D1CE4E5B9ULL;
    const uint64_t M2 = 0x94D049BB133111EBULL;
    z = (z ^ (z >> 30)) * M1;
    z = (z ^ (z >> 27)) * M2;
    return z ^ (z >> 31);
}

struct PRNG128 {
    uint64_t lo, hi;

    __device__ explicit PRNG128(uint64_t s) {
        lo = mix64(s);
        hi = mix64(s + XL);
    }

    __device__ uint64_t next64() {
        uint64_t res = rotl64(lo + hi, 17) + lo;
        uint64_t t   = hi ^ lo;
        lo = rotl64(lo, 49) ^ t ^ (t << 21);
        hi = rotl64(t, 28);
        return res;
    }

    __device__ uint32_t nextLongLower32() {
        uint64_t t = hi ^ lo;
        lo = rotl64(lo, 49) ^ t ^ (t << 21);
        hi = rotl64(t, 28);
        t  = hi ^ lo;
        return static_cast<uint32_t>((rotl64(lo + hi, 17) + lo) >> 32);
    }

    __device__ void advance() {
        uint64_t t = hi ^ lo;
        lo = rotl64(lo, 49) ^ t ^ (t << 21);
        hi = rotl64(t, 28);
    }

    __device__ int64_t nextLong() {
        int32_t high = static_cast<int32_t>(next64() >> 32);
        int32_t low  = static_cast<int32_t>(next64() >> 32);
        return (static_cast<int64_t>(high) << 32) + static_cast<int64_t>(low);
    }
};

__device__ __forceinline__ void compute_ab(uint64_t xseed, int64_t &a, int64_t &b) {
    PRNG128 rng(xseed);
    a = rng.nextLong() | 1LL;
    b = rng.nextLong() | 1LL;
}

// Computation
__device__ static inline bool goodLower32(PRNG128 &rng,
                                          uint32_t decorationxworld_lower32,
                                          uint32_t block_x_lower32,
                                          uint32_t block_z_lower32)
{
    uint32_t al = rng.nextLongLower32() | 1U;
    rng.advance();
    uint32_t bl = rng.nextLongLower32() | 1U;

    uint32_t test_decorationxworld =
        block_x_lower32 * al + block_z_lower32 * bl;

    return test_decorationxworld == decorationxworld_lower32;
}

__device__ static inline void processFullPrngState(
    uint64_t world_seed,
    Result *results,
    volatile int *result_idx,
    uint64_t decoration_seed,
    uint64_t block_x,
    uint64_t block_z)
{
    // lower 32 bits match, vast majority of cases eliminated already
    // now we can just do a full check

    int64_t a, b;
    compute_ab(world_seed ^ XH, a, b);

    uint64_t test_decoration = (block_x * a + block_z * b) ^ world_seed;
    if (test_decoration != decoration_seed) {
        return;
    }

    int this_result_idx = atomicAdd((int*)result_idx, 1);
    results[this_result_idx] = { world_seed, a, b };
}

__global__ void searchKernel(uint64_t          start_seed_prefix,
                             Result           *results,
                             volatile int     *result_idx,
                             volatile int     *checksum,
                             uint64_t          decoration_seed,
                             uint64_t          block_x,
                             uint64_t          block_z)
{
    uint64_t gid       = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t test_seed = ((start_seed_prefix + gid) << 4) +
                         (decoration_seed & 0xFULL);

    PRNG128 prng{ test_seed ^ XH };
    if (goodLower32(prng,
                     static_cast<uint32_t>(decoration_seed ^ test_seed),
                     static_cast<uint32_t>(block_x),
                     static_cast<uint32_t>(block_z))) {
        processFullPrngState(test_seed,
                             results,
                             result_idx,
                             decoration_seed,
                             block_x,
                             block_z);
    }
}

int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    uint64_t start_seed      = 0;
    uint64_t end_seed        = 20000000000000ULL;
    uint64_t device_id       = 0;

    // answer world seed = 1000
    uint64_t decoration_seed = 5504131388542975368ULL;
    uint64_t chunk_x         = 5;
    uint64_t chunk_z         = 7;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            start_seed      = strtoull(argv[++i], nullptr, 0);
        } else if (!strcmp(argv[i], "-e") && i + 1 < argc) {
            end_seed        = strtoull(argv[++i], nullptr, 0);
        } else if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            device_id       = strtoull(argv[++i], nullptr, 0);
        } else if (!strcmp(argv[i], "-dec") && i + 1 < argc) {
            decoration_seed = strtoull(argv[++i], nullptr, 0);
        } else if (!strcmp(argv[i], "-x") && i + 1 < argc) {
            chunk_x         = strtoull(argv[++i], nullptr, 0);
        } else if (!strcmp(argv[i], "-z") && i + 1 < argc) {
            chunk_z         = strtoull(argv[++i], nullptr, 0);
        } else {
            std::printf("Usage: %s [-s start_seed] [-e end_seed] [-d device_id]"
                        " [-dec decoration_seed] [-x chunk_x] [-z chunk_z]\n",
                        argv[0]);
            return 0;
        }
    }

    start_seed >>= 4;
    end_seed   >>= 4;

    CUDA_CHECK(cudaSetDevice(device_id));

    Result *d_results;
    CUDA_CHECK(cudaMalloc(&d_results, RESULTS_BUFFER_SIZE * sizeof(Result)));

    Result h_results[RESULTS_BUFFER_SIZE];

    int *results_count;
    cudaMallocManaged(&results_count, sizeof(*results_count));

    int *checksum;
    cudaMallocManaged(&checksum, sizeof(*checksum));
    *checksum = 0;

    cudaDeviceSynchronize();

    for (uint64_t curr_seed_prefix = start_seed;
         curr_seed_prefix <= end_seed;
         curr_seed_prefix += BATCH_SIZE)
    {
        *results_count = 0;

        searchKernel<<<BLOCK_SIZE, THREAD_SIZE>>>(
            curr_seed_prefix,
            d_results,
            results_count,
            checksum,
            decoration_seed,
            16 * chunk_x,
            16 * chunk_z
        );

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        if (*results_count > 0) {
            CUDA_CHECK(cudaMemcpy(
                &h_results,
                d_results,
                *results_count * sizeof(Result),
                cudaMemcpyDeviceToHost
            ));

            for (uint64_t i = 0; i < *results_count; ++i) {
                Result result = h_results[i];
                std::cout << "seed: "  << result.seed
                          << " a: "     << result.a
                          << " b: "     << result.b
                          << std::endl;
            }
        }
    }

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Checksum: "      << *checksum                << std::endl;
    std::cout << "Seeds checked: " << (end_seed - start_seed)  << std::endl;
    std::cout << "Time taken: "    << duration.count() << "ms"  << std::endl;

    double sps = (end_seed - start_seed) / duration.count() * 1e3 / 1e9;
    std::cout << "GSPS: "          << sps                      << std::endl;

    return 0;
}

