#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

// ──────────────────────────────────────────────────────────────────────
//  Compile-time configuration
// ──────────────────────────────────────────────────────────────────────
constexpr unsigned long long THREAD_SIZE  = 256;
constexpr unsigned long long BLOCK_SIZE   = 1ULL << 23;          //  8 388 608
constexpr unsigned long long BATCH_SIZE   = BLOCK_SIZE * THREAD_SIZE;
constexpr int                RESULTS_BUFFER_SIZE = 8;            //  per batch

// ──────────────────────────────────────────────────────────────────────
//  Simple CUDA error helper
// ──────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            std::cerr << "CUDA error " << cudaGetErrorString(_e)            \
                      << " at " << __FILE__ << ':' << __LINE__ << std::endl;\
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ──────────────────────────────────────────────────────────────────────
//  Result container (device + host identical)
// ──────────────────────────────────────────────────────────────────────
struct Result {
    uint64_t guess_bits;   // lower 42 bits hold the actual guess
    uint64_t result_lo;
    uint64_t result_hi;
};

// ──────────────────────────────────────────────────────────────────────
//  Small device helpers
// ──────────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint8_t extractBit(uint64_t v, int idx)
{
    return static_cast<uint8_t>((v >> idx) & 1ULL);
}

__device__ void splitToBits(uint64_t v, uint8_t *bits /* size ≥ 64 */)
{
    #pragma unroll
    for (int i = 0; i < 64; ++i)
        bits[i] = extractBit(v, i);
}

// ──────────────────────────────────────────────────────────────────────
//  0.  Rotate‐left helper
// ──────────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

// ──────────────────────────────────────────────────────────────────────
//  5.  Validation
//      Takes the two 64‐bit results and the original seeds,
//      returns true if they round‐trip to nextlong1/nextlong2.
// ──────────────────────────────────────────────────────────────────────
__device__ bool isValid(uint64_t lo, uint64_t hi,
                        uint64_t nextlong1, uint64_t nextlong2)
{
    // compute testOut1 = rotl64(lo+hi,17) + lo
    uint64_t sum0      = lo + hi;
    uint64_t testOut1 = rotl64(sum0, 17) + lo;

    // t = hi ^ lo
    uint64_t t = hi ^ lo;

    // lo' = rotl64(lo,49) ^ t ^ (t << 21)
    uint64_t lo2 = rotl64(lo, 49) ^ t ^ (t << 21);

    // hi' = rotl64(t,28)
    uint64_t hi2 = rotl64(t, 28);

    // testOut2 = rotl64(lo'+hi',17) + lo'
    uint64_t sum1      = lo2 + hi2;
    uint64_t testOut2 = rotl64(sum1, 17) + lo2;

    // only valid if both match
    return (testOut1 == nextlong1 && testOut2 == nextlong2);
}

// ──────────────────────────────────────────────────────────────────────
//  The brute-force kernel
// ──────────────────────────────────────────────────────────────────────
__global__ void bruteKernel(uint64_t nextlong1, uint64_t nextlong2,
                            uint64_t guess_base,
                            Result  *d_results,
                            unsigned int *d_resIndex)
{
    // Global thread index in current batch
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    // The 42-bit value this thread will test
    uint64_t guess = guess_base + tid;          // assumes contiguous space

    // ----------------------------------------------------------------------------
    //  1.   Expand the two 64-bit seeds into bit arrays
    // ----------------------------------------------------------------------------
    
    uint8_t S1_lo[64];
    uint8_t S1_hi[64];
    uint8_t S1_hxl[64];
    uint8_t S1_ca[64];
    uint8_t S1_is[64];
    uint8_t S1_cb[64];
    uint8_t S1_out[64];
    
    uint8_t S2_lo[64];
    uint8_t S2_hi[64];
    uint8_t S2_hxl[64];
    uint8_t S2_ca[64];
    uint8_t S2_is[64];
    uint8_t S2_cb[64];
    uint8_t S2_out[64];
    splitToBits(nextlong1, S1_out);
    splitToBits(nextlong2, S2_out);

    // ----------------------------------------------------------------------------
    //  2.   Expand the 42-bit guess into individual bits
    // ----------------------------------------------------------------------------
    uint8_t guess_bits[64];
    splitToBits(guess, guess_bits);

    // ----------------------------------------------------------------------------
    //  3.   INSERT GENERATED LOGIC HERE
    //       –––––––––––––––––––––––––––
    //  Hundreds of lines like:
    //
    //        uint8_t S1_ca_46 = S1_hxl_47 ^ S1_is_47;
    //        uint8_t S1_ca_47;
    //        uint8_t tmpSum   = carry(S1_ca_46, S1_lo_47, S1_hi_47, &S1_ca_47);
    //
    //  Please paste the auto-generated section replacing the whole placeholder.
    // ----------------------------------------------------------------------------
    //  BEGIN ▼▼▼▼▼ (placeholder block)
    S1_lo[51] = guess_bits[0];
    S1_lo[50] = guess_bits[1];
    S1_lo[49] = guess_bits[2];
    S1_lo[48] = guess_bits[3];
    S1_lo[47] = guess_bits[4];
    S1_lo[46] = guess_bits[5];
    S1_lo[45] = guess_bits[6];
    S1_lo[44] = guess_bits[7];
    S1_lo[43] = guess_bits[8];
    S1_lo[35] = guess_bits[9];
    S1_lo[34] = guess_bits[10];
    S1_lo[33] = guess_bits[11];
    S1_lo[32] = guess_bits[12];
    S1_lo[31] = guess_bits[13];
    S1_lo[30] = guess_bits[14];
    S1_lo[29] = guess_bits[15];
    S1_lo[28] = guess_bits[16];
    S1_lo[27] = guess_bits[17];
    S1_lo[26] = guess_bits[18];
    S1_lo[21] = guess_bits[19];
    S1_lo[17] = guess_bits[20];
    S1_lo[15] = guess_bits[21];
    S1_lo[14] = guess_bits[22];
    S1_lo[13] = guess_bits[23];
    S1_lo[12] = guess_bits[24];
    S1_lo[11] = guess_bits[25];
    S1_lo[9] = guess_bits[26];
    S1_lo[6] = guess_bits[27];
    S1_lo[2] = guess_bits[28];
    S1_lo[1] = guess_bits[29];
    S1_lo[0] = guess_bits[30];
    S1_hxl[47] = guess_bits[31];
    S1_hxl[26] = guess_bits[32];
    S1_hxl[11] = guess_bits[33];
    S1_hxl[7] = guess_bits[34];
    S1_hxl[4] = guess_bits[35];
    S1_is[26] = guess_bits[36];
    S1_is[9] = guess_bits[37];
    S1_is[4] = guess_bits[38];
    S2_is[53] = guess_bits[39];
    S2_is[32] = guess_bits[40];
    S2_is[11] = guess_bits[41];


    S1_hi[47] = S1_lo[47] ^ S1_hxl[47];
    S1_hi[26] = S1_lo[26] ^ S1_hxl[26];
    S1_hi[11] = S1_lo[11] ^ S1_hxl[11];
    S2_lo[11] = S1_lo[26] ^ S1_hxl[11];
    S2_hi[54] = S1_hxl[26];
    S2_hi[39] = S1_hxl[11];
    S2_hi[35] = S1_hxl[7];
    S2_hi[32] = S1_hxl[4];
    S2_hi[11] = S1_hxl[47];
    S2_hxl[11] = S1_lo[26] ^ S1_hxl[47] ^ S1_hxl[11];
    S1_ca[3] = S1_hxl[4] ^ S1_is[4];
    S1_ca[25] = S1_hxl[26] ^ S1_is[26];
    S1_ca[26] = ((S1_ca[25] & S1_lo[26]) | (S1_lo[26] & S1_hi[26]) | (S1_ca[25] & S1_hi[26]));
    S1_is[47] = S1_lo[0] ^ S1_out[0];
    S1_ca[46] = S1_hxl[47] ^ S1_is[47];
    S1_ca[47] = ((S1_ca[46] & S1_lo[47]) | (S1_lo[47] & S1_hi[47]) | (S1_ca[46] & S1_hi[47]));
    S1_cb[0] = S1_is[47] & S1_lo[0];
    S1_is[48] = S1_lo[1] ^ S1_out[1] ^ S1_cb[0];
    S1_hxl[48] = S1_is[48] ^ S1_ca[47];
    S1_hi[48] = S1_lo[48] ^ S1_hxl[48];
    S2_hi[12] = S1_hxl[48];
    S1_ca[48] = ((S1_ca[47] & S1_lo[48]) | (S1_lo[48] & S1_hi[48]) | (S1_ca[47] & S1_hi[48]));
    S1_cb[1] = ((S1_cb[0] & S1_is[48]) | (S1_is[48] & S1_lo[1]) | (S1_cb[0] & S1_lo[1]));
    S1_is[49] = S1_lo[2] ^ S1_out[2] ^ S1_cb[1];
    S1_hxl[49] = S1_is[49] ^ S1_ca[48];
    S1_hi[49] = S1_lo[49] ^ S1_hxl[49];
    S2_hi[13] = S1_hxl[49];
    S1_ca[49] = ((S1_ca[48] & S1_lo[49]) | (S1_lo[49] & S1_hi[49]) | (S1_ca[48] & S1_hi[49]));
    S1_cb[2] = ((S1_cb[1] & S1_is[49]) | (S1_is[49] & S1_lo[2]) | (S1_cb[1] & S1_lo[2]));
    S1_cb[20] = S1_is[4] ^ S1_lo[21] ^ S1_out[21];
    S1_cb[21] = ((S1_cb[20] & S1_is[4]) | (S1_is[4] & S1_lo[21]) | (S1_cb[20] & S1_lo[21]));
    S1_cb[25] = S1_is[9] ^ S1_lo[26] ^ S1_out[26];
    S1_cb[26] = ((S1_cb[25] & S1_is[9]) | (S1_is[9] & S1_lo[26]) | (S1_cb[25] & S1_lo[26]));
    S1_is[10] = S1_lo[27] ^ S1_out[27] ^ S1_cb[26];
    S1_cb[27] = ((S1_cb[26] & S1_is[10]) | (S1_is[10] & S1_lo[27]) | (S1_cb[26] & S1_lo[27]));
    S1_is[11] = S1_lo[28] ^ S1_out[28] ^ S1_cb[27];
    S1_ca[10] = S1_hxl[11] ^ S1_is[11];
    S1_ca[11] = ((S1_ca[10] & S1_lo[11]) | (S1_lo[11] & S1_hi[11]) | (S1_ca[10] & S1_hi[11]));
    S1_cb[28] = ((S1_cb[27] & S1_is[11]) | (S1_is[11] & S1_lo[28]) | (S1_cb[27] & S1_lo[28]));
    S1_is[12] = S1_lo[29] ^ S1_out[29] ^ S1_cb[28];
    S1_hxl[12] = S1_is[12] ^ S1_ca[11];
    S1_hi[12] = S1_lo[12] ^ S1_hxl[12];
    S2_lo[12] = S1_lo[27] ^ S1_hxl[12];
    S2_hi[40] = S1_hxl[12];
    S2_hxl[12] = S1_lo[27] ^ S1_hxl[48] ^ S1_hxl[12];
    S1_ca[12] = ((S1_ca[11] & S1_lo[12]) | (S1_lo[12] & S1_hi[12]) | (S1_ca[11] & S1_hi[12]));
    S1_cb[29] = ((S1_cb[28] & S1_is[12]) | (S1_is[12] & S1_lo[29]) | (S1_cb[28] & S1_lo[29]));
    S1_is[13] = S1_lo[30] ^ S1_out[30] ^ S1_cb[29];
    S1_hxl[13] = S1_is[13] ^ S1_ca[12];
    S1_hi[13] = S1_lo[13] ^ S1_hxl[13];
    S2_lo[13] = S1_lo[28] ^ S1_hxl[13];
    S2_hi[41] = S1_hxl[13];
    S2_hxl[13] = S1_lo[28] ^ S1_hxl[49] ^ S1_hxl[13];
    S1_ca[13] = ((S1_ca[12] & S1_lo[13]) | (S1_lo[13] & S1_hi[13]) | (S1_ca[12] & S1_hi[13]));
    S1_cb[30] = ((S1_cb[29] & S1_is[13]) | (S1_is[13] & S1_lo[30]) | (S1_cb[29] & S1_lo[30]));
    S1_is[14] = S1_lo[31] ^ S1_out[31] ^ S1_cb[30];
    S1_hxl[14] = S1_is[14] ^ S1_ca[13];
    S1_hi[14] = S1_lo[14] ^ S1_hxl[14];
    S2_lo[14] = S1_lo[29] ^ S1_hxl[14];
    S2_hi[42] = S1_hxl[14];
    S1_ca[14] = ((S1_ca[13] & S1_lo[14]) | (S1_lo[14] & S1_hi[14]) | (S1_ca[13] & S1_hi[14]));
    S1_cb[31] = ((S1_cb[30] & S1_is[14]) | (S1_is[14] & S1_lo[31]) | (S1_cb[30] & S1_lo[31]));
    S1_is[15] = S1_lo[32] ^ S1_out[32] ^ S1_cb[31];
    S1_hxl[15] = S1_is[15] ^ S1_ca[14];
    S1_hi[15] = S1_lo[15] ^ S1_hxl[15];
    S2_lo[15] = S1_lo[30] ^ S1_hxl[15];
    S2_hi[43] = S1_hxl[15];
    S1_ca[15] = ((S1_ca[14] & S1_lo[15]) | (S1_lo[15] & S1_hi[15]) | (S1_ca[14] & S1_hi[15]));
    S1_cb[32] = ((S1_cb[31] & S1_is[15]) | (S1_is[15] & S1_lo[32]) | (S1_cb[31] & S1_lo[32]));
    S1_is[16] = S1_lo[33] ^ S1_out[33] ^ S1_cb[32];
    S1_hxl[16] = S1_is[16] ^ S1_ca[15];
    S2_lo[16] = S1_lo[31] ^ S1_hxl[16];
    S2_hi[44] = S1_hxl[16];
    S1_cb[33] = ((S1_cb[32] & S1_is[16]) | (S1_is[16] & S1_lo[33]) | (S1_cb[32] & S1_lo[33]));
    S1_is[17] = S1_lo[34] ^ S1_out[34] ^ S1_cb[33];
    S1_cb[34] = ((S1_cb[33] & S1_is[17]) | (S1_is[17] & S1_lo[34]) | (S1_cb[33] & S1_lo[34]));
    S1_is[18] = S1_lo[35] ^ S1_out[35] ^ S1_cb[34];
    S1_cb[35] = ((S1_cb[34] & S1_is[18]) | (S1_is[18] & S1_lo[35]) | (S1_cb[34] & S1_lo[35]));
    S1_cb[42] = S1_is[26] ^ S1_lo[43] ^ S1_out[43];
    S1_cb[43] = ((S1_cb[42] & S1_is[26]) | (S1_is[26] & S1_lo[43]) | (S1_cb[42] & S1_lo[43]));
    S1_is[27] = S1_lo[44] ^ S1_out[44] ^ S1_cb[43];
    S1_hxl[27] = S1_is[27] ^ S1_ca[26];
    S1_hi[27] = S1_lo[27] ^ S1_hxl[27];
    S2_hi[55] = S1_hxl[27];
    S1_ca[27] = ((S1_ca[26] & S1_lo[27]) | (S1_lo[27] & S1_hi[27]) | (S1_ca[26] & S1_hi[27]));
    S1_cb[44] = ((S1_cb[43] & S1_is[27]) | (S1_is[27] & S1_lo[44]) | (S1_cb[43] & S1_lo[44]));
    S1_is[28] = S1_lo[45] ^ S1_out[45] ^ S1_cb[44];
    S1_hxl[28] = S1_is[28] ^ S1_ca[27];
    S1_hi[28] = S1_lo[28] ^ S1_hxl[28];
    S2_lo[49] = S1_lo[0] ^ S1_hxl[49] ^ S1_hxl[28];
    S2_lo[28] = S1_lo[43] ^ S1_hxl[28] ^ S1_hxl[7];
    S2_hi[56] = S1_hxl[28];
    S1_ca[28] = ((S1_ca[27] & S1_lo[28]) | (S1_lo[28] & S1_hi[28]) | (S1_ca[27] & S1_hi[28]));
    S1_cb[45] = ((S1_cb[44] & S1_is[28]) | (S1_is[28] & S1_lo[45]) | (S1_cb[44] & S1_lo[45]));
    S1_is[29] = S1_lo[46] ^ S1_out[46] ^ S1_cb[45];
    S1_hxl[29] = S1_is[29] ^ S1_ca[28];
    S1_hi[29] = S1_lo[29] ^ S1_hxl[29];
    S2_hi[57] = S1_hxl[29];
    S1_ca[29] = ((S1_ca[28] & S1_lo[29]) | (S1_lo[29] & S1_hi[29]) | (S1_ca[28] & S1_hi[29]));
    S1_cb[46] = ((S1_cb[45] & S1_is[29]) | (S1_is[29] & S1_lo[46]) | (S1_cb[45] & S1_lo[46]));
    S1_is[30] = S1_lo[47] ^ S1_out[47] ^ S1_cb[46];
    S1_hxl[30] = S1_is[30] ^ S1_ca[29];
    S1_hi[30] = S1_lo[30] ^ S1_hxl[30];
    S2_hi[58] = S1_hxl[30];
    S1_ca[30] = ((S1_ca[29] & S1_lo[30]) | (S1_lo[30] & S1_hi[30]) | (S1_ca[29] & S1_hi[30]));
    S1_cb[47] = ((S1_cb[46] & S1_is[30]) | (S1_is[30] & S1_lo[47]) | (S1_cb[46] & S1_lo[47]));
    S1_is[31] = S1_lo[48] ^ S1_out[48] ^ S1_cb[47];
    S1_hxl[31] = S1_is[31] ^ S1_ca[30];
    S1_hi[31] = S1_lo[31] ^ S1_hxl[31];
    S2_hi[59] = S1_hxl[31];
    S1_ca[31] = ((S1_ca[30] & S1_lo[31]) | (S1_lo[31] & S1_hi[31]) | (S1_ca[30] & S1_hi[31]));
    S1_cb[48] = ((S1_cb[47] & S1_is[31]) | (S1_is[31] & S1_lo[48]) | (S1_cb[47] & S1_lo[48]));
    S1_is[32] = S1_lo[49] ^ S1_out[49] ^ S1_cb[48];
    S1_hxl[32] = S1_is[32] ^ S1_ca[31];
    S1_hi[32] = S1_lo[32] ^ S1_hxl[32];
    S2_lo[32] = S1_lo[47] ^ S1_lo[11] ^ S1_hi[11] ^ S1_hxl[32];
    S2_hi[60] = S1_hxl[32];
    S2_hxl[32] = S1_lo[47] ^ S1_lo[11] ^ S1_hi[11] ^ S1_hxl[32] ^ S1_hxl[4];
    S1_ca[32] = ((S1_ca[31] & S1_lo[32]) | (S1_lo[32] & S1_hi[32]) | (S1_ca[31] & S1_hi[32]));
    S1_cb[49] = ((S1_cb[48] & S1_is[32]) | (S1_is[32] & S1_lo[49]) | (S1_cb[48] & S1_lo[49]));
    S1_is[33] = S1_lo[50] ^ S1_out[50] ^ S1_cb[49];
    S1_hxl[33] = S1_is[33] ^ S1_ca[32];
    S1_hi[33] = S1_lo[33] ^ S1_hxl[33];
    S2_lo[33] = S1_lo[48] ^ S1_lo[12] ^ S1_hi[12] ^ S1_hxl[33];
    S2_hi[61] = S1_hxl[33];
    S1_ca[33] = ((S1_ca[32] & S1_lo[33]) | (S1_lo[33] & S1_hi[33]) | (S1_ca[32] & S1_hi[33]));
    S1_cb[50] = ((S1_cb[49] & S1_is[33]) | (S1_is[33] & S1_lo[50]) | (S1_cb[49] & S1_lo[50]));
    S1_is[34] = S1_lo[51] ^ S1_out[51] ^ S1_cb[50];
    S1_hxl[34] = S1_is[34] ^ S1_ca[33];
    S1_hi[34] = S1_lo[34] ^ S1_hxl[34];
    S2_lo[34] = S1_lo[49] ^ S1_lo[13] ^ S1_hi[13] ^ S1_hxl[34];
    S2_hi[62] = S1_hxl[34];
    S1_ca[34] = ((S1_ca[33] & S1_lo[34]) | (S1_lo[34] & S1_hi[34]) | (S1_ca[33] & S1_hi[34]));
    S1_cb[51] = ((S1_cb[50] & S1_is[34]) | (S1_is[34] & S1_lo[51]) | (S1_cb[50] & S1_lo[51]));
    S2_ca[10] = S2_hxl[11] ^ S2_is[11];
    S2_ca[11] = ((S2_ca[10] & S2_lo[11]) | (S2_lo[11] & S2_hi[11]) | (S2_ca[10] & S2_hi[11]));
    S2_is[12] = S2_ca[11] ^ S2_hxl[12];
    S2_ca[12] = ((S2_ca[11] & S2_lo[12]) | (S2_lo[12] & S2_hi[12]) | (S2_ca[11] & S2_hi[12]));
    S2_is[13] = S2_ca[12] ^ S2_hxl[13];
    S2_ca[13] = ((S2_ca[12] & S2_lo[13]) | (S2_lo[13] & S2_hi[13]) | (S2_ca[12] & S2_hi[13]));
    S2_ca[31] = S2_hxl[32] ^ S2_is[32];
    S2_ca[32] = ((S2_ca[31] & S2_lo[32]) | (S2_lo[32] & S2_hi[32]) | (S2_ca[31] & S2_hi[32]));
    S2_cb[27] = S2_is[11] ^ S2_lo[28] ^ S2_out[28];
    S2_cb[28] = ((S2_cb[27] & S2_is[11]) | (S2_is[11] & S2_lo[28]) | (S2_cb[27] & S2_lo[28]));
    S2_lo[29] = S2_is[12] ^ S2_out[29] ^ S2_cb[28];
    S1_hxl[8] = S1_lo[44] ^ S1_lo[29] ^ S1_hi[29] ^ S2_lo[29];
    S2_hi[36] = S1_lo[44] ^ S1_lo[29] ^ S1_hi[29] ^ S2_lo[29];
    S2_cb[29] = ((S2_cb[28] & S2_is[12]) | (S2_is[12] & S2_lo[29]) | (S2_cb[28] & S2_lo[29]));
    S2_lo[30] = S2_is[13] ^ S2_out[30] ^ S2_cb[29];
    S1_hi[9] = S1_lo[45] ^ S1_lo[30] ^ S1_lo[9] ^ S1_hi[30] ^ S2_lo[30];
    S1_hxl[9] = S1_lo[45] ^ S1_lo[30] ^ S1_hi[30] ^ S2_lo[30];
    S2_hi[37] = S1_lo[45] ^ S1_lo[30] ^ S1_hi[30] ^ S2_lo[30];
    S1_ca[8] = S1_hxl[9] ^ S1_is[9];
    S1_ca[9] = ((S1_ca[8] & S1_lo[9]) | (S1_lo[9] & S1_hi[9]) | (S1_ca[8] & S1_hi[9]));
    S1_hxl[10] = S1_is[10] ^ S1_ca[9];
    S2_lo[31] = S1_lo[46] ^ S1_lo[31] ^ S1_hi[31] ^ S1_hxl[10];
    S2_hi[38] = S1_hxl[10];
    S2_cb[30] = ((S2_cb[29] & S2_is[13]) | (S2_is[13] & S2_lo[30]) | (S2_cb[29] & S2_lo[30]));
    S2_is[14] = S2_lo[31] ^ S2_out[31] ^ S2_cb[30];
    S2_hxl[14] = S2_is[14] ^ S2_ca[13];
    S1_hi[50] = S1_lo[50] ^ S1_lo[29] ^ S1_lo[14] ^ S1_hi[14] ^ S2_hxl[14];
    S1_hxl[50] = S1_lo[29] ^ S1_lo[14] ^ S1_hi[14] ^ S2_hxl[14];
    S2_lo[50] = S1_lo[14] ^ S1_lo[1] ^ S1_hi[29] ^ S1_hi[14] ^ S2_hxl[14];
    S2_hi[14] = S1_lo[29] ^ S1_lo[14] ^ S1_hi[14] ^ S2_hxl[14];
    S1_is[50] = S1_ca[49] ^ S1_hxl[50];
    S1_ca[50] = ((S1_ca[49] & S1_lo[50]) | (S1_lo[50] & S1_hi[50]) | (S1_ca[49] & S1_hi[50]));
    S1_lo[3] = S1_is[50] ^ S1_out[3] ^ S1_cb[2];
    S1_cb[3] = ((S1_cb[2] & S1_is[50]) | (S1_is[50] & S1_lo[3]) | (S1_cb[2] & S1_lo[3]));
    S2_ca[14] = ((S2_ca[13] & S2_lo[14]) | (S2_lo[14] & S2_hi[14]) | (S2_ca[13] & S2_hi[14]));
    S2_cb[31] = ((S2_cb[30] & S2_is[14]) | (S2_is[14] & S2_lo[31]) | (S2_cb[30] & S2_lo[31]));
    S2_is[15] = S2_lo[32] ^ S2_out[32] ^ S2_cb[31];
    S2_hxl[15] = S2_is[15] ^ S2_ca[14];
    S1_hi[51] = S1_lo[51] ^ S1_lo[30] ^ S1_lo[15] ^ S1_hi[15] ^ S2_hxl[15];
    S1_hxl[51] = S1_lo[30] ^ S1_lo[15] ^ S1_hi[15] ^ S2_hxl[15];
    S2_lo[51] = S1_lo[15] ^ S1_lo[2] ^ S1_hi[30] ^ S1_hi[15] ^ S2_hxl[15];
    S2_hi[15] = S1_lo[30] ^ S1_lo[15] ^ S1_hi[15] ^ S2_hxl[15];
    S1_is[51] = S1_ca[50] ^ S1_hxl[51];
    S1_ca[51] = ((S1_ca[50] & S1_lo[51]) | (S1_lo[51] & S1_hi[51]) | (S1_ca[50] & S1_hi[51]));
    S1_lo[4] = S1_is[51] ^ S1_out[4] ^ S1_cb[3];
    S1_hi[4] = S1_lo[4] ^ S1_hxl[4];
    S1_ca[4] = ((S1_ca[3] & S1_lo[4]) | (S1_lo[4] & S1_hi[4]) | (S1_ca[3] & S1_hi[4]));
    S1_cb[4] = ((S1_cb[3] & S1_is[51]) | (S1_is[51] & S1_lo[4]) | (S1_cb[3] & S1_lo[4]));
    S2_ca[15] = ((S2_ca[14] & S2_lo[15]) | (S2_lo[15] & S2_hi[15]) | (S2_ca[14] & S2_hi[15]));
    S2_cb[32] = ((S2_cb[31] & S2_is[15]) | (S2_is[15] & S2_lo[32]) | (S2_cb[31] & S2_lo[32]));
    S2_is[16] = S2_lo[33] ^ S2_out[33] ^ S2_cb[32];
    S2_hxl[16] = S2_is[16] ^ S2_ca[15];
    S1_hxl[52] = S1_lo[31] ^ S1_hxl[16] ^ S2_hxl[16];
    S2_lo[52] = S1_lo[3] ^ S1_hi[31] ^ S1_hxl[16] ^ S2_hxl[16];
    S2_hi[16] = S1_lo[31] ^ S1_hxl[16] ^ S2_hxl[16];
    S1_is[52] = S1_ca[51] ^ S1_hxl[52];
    S1_lo[5] = S1_is[52] ^ S1_out[5] ^ S1_cb[4];
    S1_cb[5] = ((S1_cb[4] & S1_is[52]) | (S1_is[52] & S1_lo[5]) | (S1_cb[4] & S1_lo[5]));
    S1_is[53] = S1_lo[6] ^ S1_out[6] ^ S1_cb[5];
    S1_cb[6] = ((S1_cb[5] & S1_is[53]) | (S1_is[53] & S1_lo[6]) | (S1_cb[5] & S1_lo[6]));
    S2_ca[16] = ((S2_ca[15] & S2_lo[16]) | (S2_lo[16] & S2_hi[16]) | (S2_ca[15] & S2_hi[16]));
    S2_cb[33] = ((S2_cb[32] & S2_is[16]) | (S2_is[16] & S2_lo[33]) | (S2_cb[32] & S2_lo[33]));
    S2_is[17] = S2_lo[34] ^ S2_out[34] ^ S2_cb[33];
    S2_hxl[17] = S2_is[17] ^ S2_ca[16];
    S2_cb[34] = ((S2_cb[33] & S2_is[17]) | (S2_is[17] & S2_lo[34]) | (S2_cb[33] & S2_lo[34]));
    S2_cb[48] = S2_is[32] ^ S2_lo[49] ^ S2_out[49];
    S2_cb[49] = ((S2_cb[48] & S2_is[32]) | (S2_is[32] & S2_lo[49]) | (S2_cb[48] & S2_lo[49]));
    S2_is[33] = S2_lo[50] ^ S2_out[50] ^ S2_cb[49];
    S2_hxl[33] = S2_is[33] ^ S2_ca[32];
    S1_hi[5] = S1_lo[48] ^ S1_lo[33] ^ S1_lo[12] ^ S1_lo[5] ^ S1_hi[33] ^ S1_hi[12] ^ S2_hxl[33];
    S1_hxl[5] = S1_lo[48] ^ S1_lo[33] ^ S1_lo[12] ^ S1_hi[33] ^ S1_hi[12] ^ S2_hxl[33];
    S2_hi[33] = S1_lo[48] ^ S1_lo[33] ^ S1_lo[12] ^ S1_hi[33] ^ S1_hi[12] ^ S2_hxl[33];
    S1_is[5] = S1_ca[4] ^ S1_hxl[5];
    S1_ca[5] = ((S1_ca[4] & S1_lo[5]) | (S1_lo[5] & S1_hi[5]) | (S1_ca[4] & S1_hi[5]));
    S1_lo[22] = S1_is[5] ^ S1_out[22] ^ S1_cb[21];
    S2_lo[7] = S1_lo[22] ^ S1_hxl[7];
    S1_cb[22] = ((S1_cb[21] & S1_is[5]) | (S1_is[5] & S1_lo[22]) | (S1_cb[21] & S1_lo[22]));
    S2_ca[33] = ((S2_ca[32] & S2_lo[33]) | (S2_lo[33] & S2_hi[33]) | (S2_ca[32] & S2_hi[33]));
    S2_cb[50] = ((S2_cb[49] & S2_is[33]) | (S2_is[33] & S2_lo[50]) | (S2_cb[49] & S2_lo[50]));
    S2_is[34] = S2_lo[51] ^ S2_out[51] ^ S2_cb[50];
    S2_hxl[34] = S2_is[34] ^ S2_ca[33];
    S1_hi[6] = S1_lo[49] ^ S1_lo[34] ^ S1_lo[13] ^ S1_lo[6] ^ S1_hi[34] ^ S1_hi[13] ^ S2_hxl[34];
    S1_hxl[6] = S1_lo[49] ^ S1_lo[34] ^ S1_lo[13] ^ S1_hi[34] ^ S1_hi[13] ^ S2_hxl[34];
    S2_lo[6] = S1_lo[49] ^ S1_lo[34] ^ S1_lo[21] ^ S1_lo[13] ^ S1_hi[34] ^ S1_hi[13] ^ S2_hxl[34];
    S2_hi[34] = S1_lo[49] ^ S1_lo[34] ^ S1_lo[13] ^ S1_hi[34] ^ S1_hi[13] ^ S2_hxl[34];
    S1_is[6] = S1_ca[5] ^ S1_hxl[6];
    S1_ca[6] = ((S1_ca[5] & S1_lo[6]) | (S1_lo[6] & S1_hi[6]) | (S1_ca[5] & S1_hi[6]));
    S1_is[7] = S1_ca[6] ^ S1_hxl[7];
    S1_lo[23] = S1_is[6] ^ S1_out[23] ^ S1_cb[22];
    S2_lo[8] = S1_lo[23] ^ S1_hxl[8];
    S1_cb[23] = ((S1_cb[22] & S1_is[6]) | (S1_is[6] & S1_lo[23]) | (S1_cb[22] & S1_lo[23]));
    S1_lo[24] = S1_is[7] ^ S1_out[24] ^ S1_cb[23];
    S2_lo[9] = S1_lo[24] ^ S1_lo[9] ^ S1_hi[9];
    S1_cb[24] = ((S1_cb[23] & S1_is[7]) | (S1_is[7] & S1_lo[24]) | (S1_cb[23] & S1_lo[24]));
    S2_ca[34] = ((S2_ca[33] & S2_lo[34]) | (S2_lo[34] & S2_hi[34]) | (S2_ca[33] & S2_hi[34]));
    S2_cb[5] = S2_is[53] ^ S2_lo[6] ^ S2_out[6];
    S2_cb[6] = ((S2_cb[5] & S2_is[53]) | (S2_is[53] & S2_lo[6]) | (S2_cb[5] & S2_lo[6]));
    S2_is[54] = S2_lo[7] ^ S2_out[7] ^ S2_cb[6];
    S2_cb[7] = ((S2_cb[6] & S2_is[54]) | (S2_is[54] & S2_lo[7]) | (S2_cb[6] & S2_lo[7]));
    S2_is[55] = S2_lo[8] ^ S2_out[8] ^ S2_cb[7];
    S2_cb[8] = ((S2_cb[7] & S2_is[55]) | (S2_is[55] & S2_lo[8]) | (S2_cb[7] & S2_lo[8]));
    S2_is[56] = S2_lo[9] ^ S2_out[9] ^ S2_cb[8];
    S2_cb[9] = ((S2_cb[8] & S2_is[56]) | (S2_is[56] & S2_lo[9]) | (S2_cb[8] & S2_lo[9]));
    S2_cb[51] = ((S2_cb[50] & S2_is[34]) | (S2_is[34] & S2_lo[51]) | (S2_cb[50] & S2_lo[51]));
    S2_is[35] = S2_lo[52] ^ S2_out[52] ^ S2_cb[51];
    S2_hxl[35] = S2_is[35] ^ S2_ca[34];
    S1_hi[35] = S1_lo[50] ^ S1_lo[35] ^ S1_lo[14] ^ S1_hi[14] ^ S1_hxl[7] ^ S2_hxl[35];
    S1_hxl[35] = S1_lo[50] ^ S1_lo[14] ^ S1_hi[14] ^ S1_hxl[7] ^ S2_hxl[35];
    S2_lo[35] = S1_hxl[7] ^ S2_hxl[35];
    S2_hi[63] = S1_lo[50] ^ S1_lo[14] ^ S1_hi[14] ^ S1_hxl[7] ^ S2_hxl[35];
    S1_is[35] = S1_ca[34] ^ S1_hxl[35];
    S1_ca[35] = ((S1_ca[34] & S1_lo[35]) | (S1_lo[35] & S1_hi[35]) | (S1_ca[34] & S1_hi[35]));
    S1_lo[52] = S1_is[35] ^ S1_out[52] ^ S1_cb[51];
    S1_hi[52] = S1_lo[52] ^ S1_hxl[52];
    S1_ca[52] = ((S1_ca[51] & S1_lo[52]) | (S1_lo[52] & S1_hi[52]) | (S1_ca[51] & S1_hi[52]));
    S1_hxl[53] = S1_is[53] ^ S1_ca[52];
    S1_hi[17] = S1_lo[32] ^ S1_lo[17] ^ S1_hxl[53] ^ S2_hxl[17];
    S1_hxl[17] = S1_lo[32] ^ S1_hxl[53] ^ S2_hxl[17];
    S2_lo[53] = S1_lo[32] ^ S1_lo[4] ^ S1_hi[32] ^ S1_hxl[53];
    S2_lo[17] = S1_hxl[53] ^ S2_hxl[17];
    S2_hi[45] = S1_lo[32] ^ S1_hxl[53] ^ S2_hxl[17];
    S2_hi[17] = S1_hxl[53];
    S1_ca[16] = S1_hxl[17] ^ S1_is[17];
    S1_ca[17] = ((S1_ca[16] & S1_lo[17]) | (S1_lo[17] & S1_hi[17]) | (S1_ca[16] & S1_hi[17]));
    S1_hxl[18] = S1_is[18] ^ S1_ca[17];
    S2_lo[18] = S1_lo[33] ^ S1_hxl[18];
    S2_hi[46] = S1_hxl[18];
    S1_cb[52] = ((S1_cb[51] & S1_is[35]) | (S1_is[35] & S1_lo[52]) | (S1_cb[51] & S1_lo[52]));
    S2_ca[17] = ((S2_ca[16] & S2_lo[17]) | (S2_lo[17] & S2_hi[17]) | (S2_ca[16] & S2_hi[17]));
    S2_ca[35] = ((S2_ca[34] & S2_lo[35]) | (S2_lo[35] & S2_hi[35]) | (S2_ca[34] & S2_hi[35]));
    S2_is[18] = S2_lo[35] ^ S2_out[35] ^ S2_cb[34];
    S2_hxl[18] = S2_is[18] ^ S2_ca[17];
    S1_hxl[54] = S1_lo[33] ^ S1_hxl[18] ^ S2_hxl[18];
    S2_lo[54] = S1_lo[5] ^ S1_hi[33] ^ S1_hxl[18] ^ S2_hxl[18];
    S2_hi[18] = S1_lo[33] ^ S1_hxl[18] ^ S2_hxl[18];
    S2_hxl[54] = S1_lo[26] ^ S1_lo[5] ^ S1_hi[33] ^ S1_hi[26] ^ S1_hxl[18] ^ S2_hxl[18];
    S2_ca[18] = ((S2_ca[17] & S2_lo[18]) | (S2_lo[18] & S2_hi[18]) | (S2_ca[17] & S2_hi[18]));
    S2_ca[53] = S2_hxl[54] ^ S2_is[54];
    S2_ca[54] = ((S2_ca[53] & S2_lo[54]) | (S2_lo[54] & S2_hi[54]) | (S2_ca[53] & S2_hi[54]));
    S2_hxl[55] = S2_is[55] ^ S2_ca[54];
    S1_hxl[55] = S1_lo[27] ^ S1_lo[6] ^ S1_hi[27] ^ S1_hxl[34] ^ S2_hxl[55];
    S2_lo[55] = S1_lo[27] ^ S1_hi[27] ^ S2_hxl[55];
    S2_hi[19] = S1_lo[27] ^ S1_lo[6] ^ S1_hi[27] ^ S1_hxl[34] ^ S2_hxl[55];
    S2_ca[55] = ((S2_ca[54] & S2_lo[55]) | (S2_lo[55] & S2_hi[55]) | (S2_ca[54] & S2_hi[55]));
    S2_hxl[56] = S2_is[56] ^ S2_ca[55];
    S2_lo[56] = S1_lo[28] ^ S1_hi[28] ^ S2_hxl[56];
    S2_ca[56] = ((S2_ca[55] & S2_lo[56]) | (S2_lo[56] & S2_hi[56]) | (S2_ca[55] & S2_hi[56]));
    S2_cb[35] = ((S2_cb[34] & S2_is[18]) | (S2_is[18] & S2_lo[35]) | (S2_cb[34] & S2_lo[35]));
    S2_cb[52] = ((S2_cb[51] & S2_is[35]) | (S2_is[35] & S2_lo[52]) | (S2_cb[51] & S2_lo[52]));
    S2_is[36] = S2_lo[53] ^ S2_out[53] ^ S2_cb[52];
    S2_hxl[36] = S2_is[36] ^ S2_ca[35];
    S1_hxl[36] = S1_lo[51] ^ S1_lo[15] ^ S1_hi[15] ^ S1_hxl[8] ^ S2_hxl[36];
    S2_lo[36] = S1_hxl[8] ^ S2_hxl[36];
    S2_hi[0] = S1_lo[51] ^ S1_lo[15] ^ S1_hi[15] ^ S1_hxl[8] ^ S2_hxl[36];
    S1_is[36] = S1_ca[35] ^ S1_hxl[36];
    S1_lo[53] = S1_is[36] ^ S1_out[53] ^ S1_cb[52];
    S1_hi[53] = S1_lo[53] ^ S1_hxl[53];
    S1_ca[53] = ((S1_ca[52] & S1_lo[53]) | (S1_lo[53] & S1_hi[53]) | (S1_ca[52] & S1_hi[53]));
    S1_is[54] = S1_ca[53] ^ S1_hxl[54];
    S1_lo[7] = S1_is[54] ^ S1_out[7] ^ S1_cb[6];
    S1_hi[7] = S1_lo[7] ^ S1_hxl[7];
    S1_hxl[56] = S1_lo[7] ^ S1_hxl[35] ^ S2_lo[56];
    S2_hi[20] = S1_lo[7] ^ S1_hxl[35] ^ S2_lo[56];
    S1_ca[7] = ((S1_ca[6] & S1_lo[7]) | (S1_lo[7] & S1_hi[7]) | (S1_ca[6] & S1_hi[7]));
    S1_is[8] = S1_ca[7] ^ S1_hxl[8];
    S1_cb[7] = ((S1_cb[6] & S1_is[54]) | (S1_is[54] & S1_lo[7]) | (S1_cb[6] & S1_lo[7]));
    S1_lo[25] = S1_is[8] ^ S1_out[25] ^ S1_cb[24];
    S2_lo[10] = S1_lo[25] ^ S1_hxl[10];
    S1_cb[53] = ((S1_cb[52] & S1_is[36]) | (S1_is[36] & S1_lo[53]) | (S1_cb[52] & S1_lo[53]));
    S2_ca[36] = ((S2_ca[35] & S2_lo[36]) | (S2_lo[36] & S2_hi[36]) | (S2_ca[35] & S2_hi[36]));
    S2_is[57] = S2_lo[10] ^ S2_out[10] ^ S2_cb[9];
    S2_hxl[57] = S2_is[57] ^ S2_ca[56];
    S2_lo[57] = S1_lo[29] ^ S1_hi[29] ^ S2_hxl[57];
    S2_ca[57] = ((S2_ca[56] & S2_lo[57]) | (S2_lo[57] & S2_hi[57]) | (S2_ca[56] & S2_hi[57]));
    S2_cb[10] = ((S2_cb[9] & S2_is[57]) | (S2_is[57] & S2_lo[10]) | (S2_cb[9] & S2_lo[10]));
    S2_is[58] = S2_lo[11] ^ S2_out[11] ^ S2_cb[10];
    S2_hxl[58] = S2_is[58] ^ S2_ca[57];
    S2_lo[58] = S1_lo[30] ^ S1_hi[30] ^ S2_hxl[58];
    S2_ca[58] = ((S2_ca[57] & S2_lo[58]) | (S2_lo[58] & S2_hi[58]) | (S2_ca[57] & S2_hi[58]));
    S2_cb[11] = ((S2_cb[10] & S2_is[58]) | (S2_is[58] & S2_lo[11]) | (S2_cb[10] & S2_lo[11]));
    S2_is[59] = S2_lo[12] ^ S2_out[12] ^ S2_cb[11];
    S2_hxl[59] = S2_is[59] ^ S2_ca[58];
    S2_lo[59] = S1_hxl[31] ^ S2_hxl[59];
    S2_ca[59] = ((S2_ca[58] & S2_lo[59]) | (S2_lo[59] & S2_hi[59]) | (S2_ca[58] & S2_hi[59]));
    S2_cb[12] = ((S2_cb[11] & S2_is[59]) | (S2_is[59] & S2_lo[12]) | (S2_cb[11] & S2_lo[12]));
    S2_is[60] = S2_lo[13] ^ S2_out[13] ^ S2_cb[12];
    S2_hxl[60] = S2_is[60] ^ S2_ca[59];
    S2_lo[60] = S1_lo[32] ^ S1_hi[32] ^ S2_hxl[60];
    S2_ca[60] = ((S2_ca[59] & S2_lo[60]) | (S2_lo[60] & S2_hi[60]) | (S2_ca[59] & S2_hi[60]));
    S2_cb[13] = ((S2_cb[12] & S2_is[60]) | (S2_is[60] & S2_lo[13]) | (S2_cb[12] & S2_lo[13]));
    S2_is[61] = S2_lo[14] ^ S2_out[14] ^ S2_cb[13];
    S2_hxl[61] = S2_is[61] ^ S2_ca[60];
    S2_lo[61] = S1_hxl[33] ^ S2_hxl[61];
    S2_ca[61] = ((S2_ca[60] & S2_lo[61]) | (S2_lo[61] & S2_hi[61]) | (S2_ca[60] & S2_hi[61]));
    S2_cb[14] = ((S2_cb[13] & S2_is[61]) | (S2_is[61] & S2_lo[14]) | (S2_cb[13] & S2_lo[14]));
    S2_is[62] = S2_lo[15] ^ S2_out[15] ^ S2_cb[14];
    S2_hxl[62] = S2_is[62] ^ S2_ca[61];
    S2_lo[62] = S1_lo[34] ^ S1_hi[34] ^ S2_hxl[62];
    S2_ca[62] = ((S2_ca[61] & S2_lo[62]) | (S2_lo[62] & S2_hi[62]) | (S2_ca[61] & S2_hi[62]));
    S2_cb[15] = ((S2_cb[14] & S2_is[62]) | (S2_is[62] & S2_lo[15]) | (S2_cb[14] & S2_lo[15]));
    S2_is[63] = S2_lo[16] ^ S2_out[16] ^ S2_cb[15];
    S2_hxl[63] = S2_is[63] ^ S2_ca[62];
    S2_lo[63] = S1_lo[35] ^ S1_hi[35] ^ S2_hxl[63];
    S2_ca[63] = ((S2_ca[62] & S2_lo[63]) | (S2_lo[63] & S2_hi[63]) | (S2_ca[62] & S2_hi[63]));
    S2_cb[16] = ((S2_cb[15] & S2_is[63]) | (S2_is[63] & S2_lo[16]) | (S2_cb[15] & S2_lo[16]));
    S2_is[0] = S2_lo[17] ^ S2_out[17] ^ S2_cb[16];
    S2_hxl[0] = S2_is[0];
    S1_hi[0] = S1_lo[15] ^ S1_lo[0] ^ S1_hxl[36] ^ S2_hxl[0];
    S1_hxl[0] = S1_lo[15] ^ S1_hxl[36] ^ S2_hxl[0];
    S2_lo[0] = S1_hxl[36] ^ S2_hxl[0];
    S2_hi[28] = S1_lo[15] ^ S1_hxl[36] ^ S2_hxl[0];
    S2_hxl[28] = S1_lo[43] ^ S1_lo[28] ^ S1_lo[15] ^ S1_lo[7] ^ S1_hi[28] ^ S1_hi[7] ^ S1_hxl[36] ^ S2_hxl[0];
    S1_is[0] = S1_hxl[0];
    S1_ca[0] = S1_lo[0] & S1_hi[0];
    S1_cb[16] = S1_is[0] ^ S1_lo[17] ^ S1_out[17];
    S1_cb[17] = ((S1_cb[16] & S1_is[0]) | (S1_is[0] & S1_lo[17]) | (S1_cb[16] & S1_lo[17]));
    S2_ca[0] = S2_lo[0] & S2_hi[0];
    S2_is[47] = S2_lo[0] ^ S2_out[0];
    S2_cb[0] = S2_is[47] & S2_lo[0];
    S2_cb[17] = ((S2_cb[16] & S2_is[0]) | (S2_is[0] & S2_lo[17]) | (S2_cb[16] & S2_lo[17]));
    S2_is[1] = S2_lo[18] ^ S2_out[18] ^ S2_cb[17];
    S2_hxl[1] = S2_is[1] ^ S2_ca[0];
    S2_cb[18] = ((S2_cb[17] & S2_is[1]) | (S2_is[1] & S2_lo[18]) | (S2_cb[17] & S2_lo[18]));
    S2_is[19] = S2_lo[36] ^ S2_out[36] ^ S2_cb[35];
    S2_hxl[19] = S2_is[19] ^ S2_ca[18];
    S1_hxl[19] = S1_lo[34] ^ S1_hxl[55] ^ S2_hxl[19];
    S2_lo[19] = S1_hxl[55] ^ S2_hxl[19];
    S2_hi[47] = S1_lo[34] ^ S1_hxl[55] ^ S2_hxl[19];
    S2_ca[19] = ((S2_ca[18] & S2_lo[19]) | (S2_lo[19] & S2_hi[19]) | (S2_ca[18] & S2_hi[19]));
    S2_is[2] = S2_lo[19] ^ S2_out[19] ^ S2_cb[18];
    S2_cb[19] = ((S2_cb[18] & S2_is[2]) | (S2_is[2] & S2_lo[19]) | (S2_cb[18] & S2_lo[19]));
    S2_cb[36] = ((S2_cb[35] & S2_is[19]) | (S2_is[19] & S2_lo[36]) | (S2_cb[35] & S2_lo[36]));
    S2_cb[53] = ((S2_cb[52] & S2_is[36]) | (S2_is[36] & S2_lo[53]) | (S2_cb[52] & S2_lo[53]));
    S2_is[37] = S2_lo[54] ^ S2_out[54] ^ S2_cb[53];
    S2_hxl[37] = S2_is[37] ^ S2_ca[36];
    S1_hxl[58] = S1_hi[52] ^ S1_hi[9] ^ S1_hxl[52] ^ S1_hxl[16] ^ S2_lo[58] ^ S2_hxl[37];
    S1_hxl[37] = S1_lo[9] ^ S1_hi[52] ^ S1_hi[9] ^ S1_hxl[52] ^ S1_hxl[16] ^ S2_hxl[37];
    S2_lo[37] = S1_lo[9] ^ S1_hi[9] ^ S2_hxl[37];
    S2_lo[1] = S1_lo[9] ^ S1_hi[52] ^ S1_hi[9] ^ S1_hxl[52] ^ S1_hxl[16] ^ S2_hxl[37] ^ S2_hxl[1];
    S2_hi[22] = S1_hi[52] ^ S1_hi[9] ^ S1_hxl[52] ^ S1_hxl[16] ^ S2_lo[58] ^ S2_hxl[37];
    S2_hi[1] = S1_lo[9] ^ S1_hi[52] ^ S1_hi[9] ^ S1_hxl[52] ^ S1_hxl[16] ^ S2_hxl[37];
    S2_ca[1] = ((S2_ca[0] & S2_lo[1]) | (S2_lo[1] & S2_hi[1]) | (S2_ca[0] & S2_hi[1]));
    S2_hxl[2] = S2_is[2] ^ S2_ca[1];
    S2_ca[37] = ((S2_ca[36] & S2_lo[37]) | (S2_lo[37] & S2_hi[37]) | (S2_ca[36] & S2_hi[37]));
    S2_is[48] = S2_lo[1] ^ S2_out[1] ^ S2_cb[0];
    S2_cb[1] = ((S2_cb[0] & S2_is[48]) | (S2_is[48] & S2_lo[1]) | (S2_cb[0] & S2_lo[1]));
    S2_is[20] = S2_lo[37] ^ S2_out[37] ^ S2_cb[36];
    S2_hxl[20] = S2_is[20] ^ S2_ca[19];
    S1_hxl[20] = S1_lo[35] ^ S1_hxl[56] ^ S2_hxl[20];
    S2_lo[20] = S1_hxl[56] ^ S2_hxl[20];
    S2_hi[48] = S1_lo[35] ^ S1_hxl[56] ^ S2_hxl[20];
    S2_ca[20] = ((S2_ca[19] & S2_lo[20]) | (S2_lo[20] & S2_hi[20]) | (S2_ca[19] & S2_hi[20]));
    S2_is[3] = S2_lo[20] ^ S2_out[20] ^ S2_cb[19];
    S2_cb[20] = ((S2_cb[19] & S2_is[3]) | (S2_is[3] & S2_lo[20]) | (S2_cb[19] & S2_lo[20]));
    S2_cb[37] = ((S2_cb[36] & S2_is[20]) | (S2_is[20] & S2_lo[37]) | (S2_cb[36] & S2_lo[37]));
    S2_cb[54] = ((S2_cb[53] & S2_is[37]) | (S2_is[37] & S2_lo[54]) | (S2_cb[53] & S2_lo[54]));
    S2_is[38] = S2_lo[55] ^ S2_out[55] ^ S2_cb[54];
    S2_hxl[38] = S2_is[38] ^ S2_ca[37];
    S1_hi[2] = S1_lo[2] ^ S1_hi[53] ^ S1_hi[17] ^ S1_hxl[53] ^ S1_hxl[10] ^ S2_hxl[38] ^ S2_hxl[2];
    S1_hxl[38] = S1_lo[17] ^ S1_hi[53] ^ S1_hi[17] ^ S1_hxl[53] ^ S1_hxl[10] ^ S2_hxl[38];
    S1_hxl[2] = S1_hi[53] ^ S1_hi[17] ^ S1_hxl[53] ^ S1_hxl[10] ^ S2_hxl[38] ^ S2_hxl[2];
    S2_lo[38] = S1_hxl[10] ^ S2_hxl[38];
    S2_lo[2] = S1_lo[17] ^ S1_hi[53] ^ S1_hi[17] ^ S1_hxl[53] ^ S1_hxl[10] ^ S2_hxl[38] ^ S2_hxl[2];
    S2_hi[30] = S1_hi[53] ^ S1_hi[17] ^ S1_hxl[53] ^ S1_hxl[10] ^ S2_hxl[38] ^ S2_hxl[2];
    S2_hi[2] = S1_lo[17] ^ S1_hi[53] ^ S1_hi[17] ^ S1_hxl[53] ^ S1_hxl[10] ^ S2_hxl[38];
    S2_hxl[30] = S1_lo[45] ^ S1_lo[9] ^ S1_hi[53] ^ S1_hi[17] ^ S1_hi[9] ^ S1_hxl[53] ^ S1_hxl[30] ^ S1_hxl[10] ^ S2_hxl[38] ^ S2_hxl[2];
    S2_ca[2] = ((S2_ca[1] & S2_lo[2]) | (S2_lo[2] & S2_hi[2]) | (S2_ca[1] & S2_hi[2]));
    S2_hxl[3] = S2_is[3] ^ S2_ca[2];
    S2_ca[38] = ((S2_ca[37] & S2_lo[38]) | (S2_lo[38] & S2_hi[38]) | (S2_ca[37] & S2_hi[38]));
    S2_is[49] = S2_lo[2] ^ S2_out[2] ^ S2_cb[1];
    S2_cb[2] = ((S2_cb[1] & S2_is[49]) | (S2_is[49] & S2_lo[2]) | (S2_cb[1] & S2_lo[2]));
    S2_is[21] = S2_lo[38] ^ S2_out[38] ^ S2_cb[37];
    S2_hxl[21] = S2_is[21] ^ S2_ca[20];
    S2_cb[38] = ((S2_cb[37] & S2_is[21]) | (S2_is[21] & S2_lo[38]) | (S2_cb[37] & S2_lo[38]));
    S2_cb[55] = ((S2_cb[54] & S2_is[38]) | (S2_is[38] & S2_lo[55]) | (S2_cb[54] & S2_lo[55]));
    S2_is[39] = S2_lo[56] ^ S2_out[56] ^ S2_cb[55];
    S2_hxl[39] = S2_is[39] ^ S2_ca[38];
    S2_lo[39] = S1_lo[11] ^ S1_hi[11] ^ S2_hxl[39];
    S2_ca[39] = ((S2_ca[38] & S2_lo[39]) | (S2_lo[39] & S2_hi[39]) | (S2_ca[38] & S2_hi[39]));
    S2_is[22] = S2_lo[39] ^ S2_out[39] ^ S2_cb[38];
    S2_cb[39] = ((S2_cb[38] & S2_is[22]) | (S2_is[22] & S2_lo[39]) | (S2_cb[38] & S2_lo[39]));
    S2_cb[56] = ((S2_cb[55] & S2_is[39]) | (S2_is[39] & S2_lo[56]) | (S2_cb[55] & S2_lo[56]));
    S2_is[40] = S2_lo[57] ^ S2_out[57] ^ S2_cb[56];
    S2_hxl[40] = S2_is[40] ^ S2_ca[39];
    S2_lo[40] = S1_hxl[12] ^ S2_hxl[40];
    S2_ca[40] = ((S2_ca[39] & S2_lo[40]) | (S2_lo[40] & S2_hi[40]) | (S2_ca[39] & S2_hi[40]));
    S2_is[23] = S2_lo[40] ^ S2_out[40] ^ S2_cb[39];
    S2_cb[40] = ((S2_cb[39] & S2_is[23]) | (S2_is[23] & S2_lo[40]) | (S2_cb[39] & S2_lo[40]));
    S2_cb[57] = ((S2_cb[56] & S2_is[40]) | (S2_is[40] & S2_lo[57]) | (S2_cb[56] & S2_lo[57]));
    S2_is[41] = S2_lo[58] ^ S2_out[58] ^ S2_cb[57];
    S2_hxl[41] = S2_is[41] ^ S2_ca[40];
    S2_lo[41] = S1_hxl[13] ^ S2_hxl[41];
    S2_ca[41] = ((S2_ca[40] & S2_lo[41]) | (S2_lo[41] & S2_hi[41]) | (S2_ca[40] & S2_hi[41]));
    S2_is[24] = S2_lo[41] ^ S2_out[41] ^ S2_cb[40];
    S2_cb[41] = ((S2_cb[40] & S2_is[24]) | (S2_is[24] & S2_lo[41]) | (S2_cb[40] & S2_lo[41]));
    S2_cb[58] = ((S2_cb[57] & S2_is[41]) | (S2_is[41] & S2_lo[58]) | (S2_cb[57] & S2_lo[58]));
    S2_is[42] = S2_lo[59] ^ S2_out[59] ^ S2_cb[58];
    S2_hxl[42] = S2_is[42] ^ S2_ca[41];
    S2_lo[42] = S1_lo[14] ^ S1_hi[14] ^ S2_hxl[42];
    S2_ca[42] = ((S2_ca[41] & S2_lo[42]) | (S2_lo[42] & S2_hi[42]) | (S2_ca[41] & S2_hi[42]));
    S2_is[25] = S2_lo[42] ^ S2_out[42] ^ S2_cb[41];
    S2_cb[42] = ((S2_cb[41] & S2_is[25]) | (S2_is[25] & S2_lo[42]) | (S2_cb[41] & S2_lo[42]));
    S2_cb[59] = ((S2_cb[58] & S2_is[42]) | (S2_is[42] & S2_lo[59]) | (S2_cb[58] & S2_lo[59]));
    S2_is[43] = S2_lo[60] ^ S2_out[60] ^ S2_cb[59];
    S2_hxl[43] = S2_is[43] ^ S2_ca[42];
    S2_lo[43] = S1_lo[15] ^ S1_hi[15] ^ S2_hxl[43];
    S2_ca[43] = ((S2_ca[42] & S2_lo[43]) | (S2_lo[43] & S2_hi[43]) | (S2_ca[42] & S2_hi[43]));
    S2_is[26] = S2_lo[43] ^ S2_out[43] ^ S2_cb[42];
    S2_cb[43] = ((S2_cb[42] & S2_is[26]) | (S2_is[26] & S2_lo[43]) | (S2_cb[42] & S2_lo[43]));
    S2_cb[60] = ((S2_cb[59] & S2_is[43]) | (S2_is[43] & S2_lo[60]) | (S2_cb[59] & S2_lo[60]));
    S2_is[44] = S2_lo[61] ^ S2_out[61] ^ S2_cb[60];
    S2_hxl[44] = S2_is[44] ^ S2_ca[43];
    S2_lo[44] = S1_hxl[16] ^ S2_hxl[44];
    S2_ca[44] = ((S2_ca[43] & S2_lo[44]) | (S2_lo[44] & S2_hi[44]) | (S2_ca[43] & S2_hi[44]));
    S2_is[27] = S2_lo[44] ^ S2_out[44] ^ S2_cb[43];
    S2_cb[44] = ((S2_cb[43] & S2_is[27]) | (S2_is[27] & S2_lo[44]) | (S2_cb[43] & S2_lo[44]));
    S2_cb[61] = ((S2_cb[60] & S2_is[44]) | (S2_is[44] & S2_lo[61]) | (S2_cb[60] & S2_lo[61]));
    S2_is[45] = S2_lo[62] ^ S2_out[62] ^ S2_cb[61];
    S2_hxl[45] = S2_is[45] ^ S2_ca[44];
    S2_lo[45] = S1_lo[17] ^ S1_hi[17] ^ S2_hxl[45];
    S2_ca[45] = ((S2_ca[44] & S2_lo[45]) | (S2_lo[45] & S2_hi[45]) | (S2_ca[44] & S2_hi[45]));
    S2_is[28] = S2_lo[45] ^ S2_out[45] ^ S2_cb[44];
    S2_ca[27] = S2_hxl[28] ^ S2_is[28];
    S2_ca[28] = ((S2_ca[27] & S2_lo[28]) | (S2_lo[28] & S2_hi[28]) | (S2_ca[27] & S2_hi[28]));
    S2_cb[45] = ((S2_cb[44] & S2_is[28]) | (S2_is[28] & S2_lo[45]) | (S2_cb[44] & S2_lo[45]));
    S2_cb[62] = ((S2_cb[61] & S2_is[45]) | (S2_is[45] & S2_lo[62]) | (S2_cb[61] & S2_lo[62]));
    S2_is[46] = S2_lo[63] ^ S2_out[63] ^ S2_cb[62];
    S2_hxl[46] = S2_is[46] ^ S2_ca[45];
    S2_lo[46] = S1_hxl[18] ^ S2_hxl[46];
    S2_ca[46] = ((S2_ca[45] & S2_lo[46]) | (S2_lo[46] & S2_hi[46]) | (S2_ca[45] & S2_hi[46]));
    S2_hxl[47] = S2_is[47] ^ S2_ca[46];
    S1_lo[62] = S1_hxl[47] ^ S1_hxl[26] ^ S1_hxl[19] ^ S2_hxl[47];
    S2_lo[47] = S1_hxl[19] ^ S2_hxl[47];
    S2_ca[47] = ((S2_ca[46] & S2_lo[47]) | (S2_lo[47] & S2_hi[47]) | (S2_ca[46] & S2_hi[47]));
    S2_hxl[48] = S2_is[48] ^ S2_ca[47];
    S1_lo[63] = S1_hxl[48] ^ S1_hxl[27] ^ S1_hxl[20] ^ S2_hxl[48];
    S2_lo[48] = S1_hxl[20] ^ S2_hxl[48];
    S2_ca[48] = ((S2_ca[47] & S2_lo[48]) | (S2_lo[48] & S2_hi[48]) | (S2_ca[47] & S2_hi[48]));
    S2_hxl[49] = S2_is[49] ^ S2_ca[48];
    S1_hi[21] = S1_lo[21] ^ S1_lo[0] ^ S1_hxl[49] ^ S1_hxl[28] ^ S2_hxl[49];
    S1_hxl[21] = S1_lo[0] ^ S1_hxl[49] ^ S1_hxl[28] ^ S2_hxl[49];
    S2_hi[49] = S1_lo[0] ^ S1_hxl[49] ^ S1_hxl[28] ^ S2_hxl[49];
    S2_ca[49] = ((S2_ca[48] & S2_lo[49]) | (S2_lo[49] & S2_hi[49]) | (S2_ca[48] & S2_hi[49]));
    S2_is[29] = S2_lo[46] ^ S2_out[46] ^ S2_cb[45];
    S2_hxl[29] = S2_is[29] ^ S2_ca[28];
    S1_lo[16] = S1_lo[44] ^ S1_lo[29] ^ S1_hi[29] ^ S1_hxl[8] ^ S2_lo[1] ^ S2_hxl[29];
    S1_hi[16] = S1_lo[44] ^ S1_lo[29] ^ S1_hi[29] ^ S1_hxl[16] ^ S1_hxl[8] ^ S2_lo[1] ^ S2_hxl[29];
    S1_hi[1] = S1_lo[44] ^ S1_lo[29] ^ S1_lo[1] ^ S1_hi[29] ^ S1_hxl[8] ^ S2_hxl[29];
    S1_hxl[1] = S1_lo[44] ^ S1_lo[29] ^ S1_hi[29] ^ S1_hxl[8] ^ S2_hxl[29];
    S2_hi[29] = S1_lo[44] ^ S1_lo[29] ^ S1_hi[29] ^ S1_hxl[8] ^ S2_hxl[29];
    S1_is[1] = S1_ca[0] ^ S1_hxl[1];
    S1_ca[1] = ((S1_ca[0] & S1_lo[1]) | (S1_lo[1] & S1_hi[1]) | (S1_ca[0] & S1_hi[1]));
    S1_is[2] = S1_ca[1] ^ S1_hxl[2];
    S1_ca[2] = ((S1_ca[1] & S1_lo[2]) | (S1_lo[2] & S1_hi[2]) | (S1_ca[1] & S1_hi[2]));
    S1_lo[18] = S1_is[1] ^ S1_out[18] ^ S1_cb[17];
    S1_hi[18] = S1_lo[18] ^ S1_hxl[18];
    S1_ca[18] = ((S1_ca[17] & S1_lo[18]) | (S1_lo[18] & S1_hi[18]) | (S1_ca[17] & S1_hi[18]));
    S1_is[19] = S1_ca[18] ^ S1_hxl[19];
    S1_cb[18] = ((S1_cb[17] & S1_is[1]) | (S1_is[1] & S1_lo[18]) | (S1_cb[17] & S1_lo[18]));
    S1_lo[19] = S1_is[2] ^ S1_out[19] ^ S1_cb[18];
    S1_hi[19] = S1_lo[19] ^ S1_hxl[19];
    S2_lo[4] = S1_lo[19] ^ S1_lo[4] ^ S1_hi[4];
    S1_ca[19] = ((S1_ca[18] & S1_lo[19]) | (S1_lo[19] & S1_hi[19]) | (S1_ca[18] & S1_hi[19]));
    S1_is[20] = S1_ca[19] ^ S1_hxl[20];
    S1_cb[19] = ((S1_cb[18] & S1_is[2]) | (S1_is[2] & S1_lo[19]) | (S1_cb[18] & S1_lo[19]));
    S1_lo[36] = S1_is[19] ^ S1_out[36] ^ S1_cb[35];
    S1_lo[8] = S1_lo[36] ^ S1_lo[0] ^ S1_hi[0] ^ S1_hxl[36] ^ S1_hxl[21] ^ S2_lo[57] ^ S2_hxl[21];
    S1_hi[36] = S1_lo[36] ^ S1_hxl[36];
    S1_hi[8] = S1_lo[36] ^ S1_lo[0] ^ S1_hi[0] ^ S1_hxl[36] ^ S1_hxl[21] ^ S1_hxl[8] ^ S2_lo[57] ^ S2_hxl[21];
    S1_hxl[57] = S1_lo[36] ^ S1_lo[0] ^ S1_hi[0] ^ S1_hxl[21] ^ S2_hxl[21];
    S2_lo[21] = S1_lo[36] ^ S1_lo[0] ^ S1_hi[0] ^ S1_hxl[21];
    S2_hi[21] = S1_lo[36] ^ S1_lo[0] ^ S1_hi[0] ^ S1_hxl[21] ^ S2_hxl[21];
    S1_ca[36] = ((S1_ca[35] & S1_lo[36]) | (S1_lo[36] & S1_hi[36]) | (S1_ca[35] & S1_hi[36]));
    S1_is[37] = S1_ca[36] ^ S1_hxl[37];
    S1_is[55] = S1_lo[8] ^ S1_out[8] ^ S1_cb[7];
    S1_ca[54] = S1_hxl[55] ^ S1_is[55];
    S1_cb[8] = ((S1_cb[7] & S1_is[55]) | (S1_is[55] & S1_lo[8]) | (S1_cb[7] & S1_lo[8]));
    S1_is[56] = S1_lo[9] ^ S1_out[9] ^ S1_cb[8];
    S1_ca[55] = S1_hxl[56] ^ S1_is[56];
    S1_cb[9] = ((S1_cb[8] & S1_is[56]) | (S1_is[56] & S1_lo[9]) | (S1_cb[8] & S1_lo[9]));
    S1_cb[36] = ((S1_cb[35] & S1_is[19]) | (S1_is[19] & S1_lo[36]) | (S1_cb[35] & S1_lo[36]));
    S1_lo[37] = S1_is[20] ^ S1_out[37] ^ S1_cb[36];
    S1_hi[37] = S1_lo[37] ^ S1_hxl[37];
    S1_ca[37] = ((S1_ca[36] & S1_lo[37]) | (S1_lo[37] & S1_hi[37]) | (S1_ca[36] & S1_hi[37]));
    S1_is[38] = S1_ca[37] ^ S1_hxl[38];
    S1_cb[37] = ((S1_cb[36] & S1_is[20]) | (S1_is[20] & S1_lo[37]) | (S1_cb[36] & S1_lo[37]));
    S1_lo[54] = S1_is[37] ^ S1_out[54] ^ S1_cb[53];
    S1_hi[54] = S1_lo[54] ^ S1_hxl[54];
    S1_hi[3] = S1_lo[54] ^ S1_lo[3] ^ S1_hi[18] ^ S2_lo[39] ^ S2_hxl[3];
    S1_hxl[60] = S1_lo[54] ^ S1_lo[18] ^ S1_lo[11] ^ S1_hi[18] ^ S2_lo[60] ^ S2_lo[39];
    S1_hxl[39] = S1_lo[54] ^ S1_lo[18] ^ S1_hi[18] ^ S2_lo[39];
    S1_hxl[3] = S1_lo[54] ^ S1_hi[18] ^ S2_lo[39] ^ S2_hxl[3];
    S2_lo[3] = S1_lo[54] ^ S1_lo[18] ^ S1_hi[18] ^ S2_lo[39] ^ S2_hxl[3];
    S2_hi[31] = S1_lo[54] ^ S1_hi[18] ^ S2_lo[39] ^ S2_hxl[3];
    S2_hi[24] = S1_lo[54] ^ S1_lo[18] ^ S1_lo[11] ^ S1_hi[18] ^ S2_lo[60] ^ S2_lo[39];
    S2_hi[3] = S1_lo[54] ^ S1_lo[18] ^ S1_hi[18] ^ S2_lo[39];
    S2_hxl[31] = S1_lo[54] ^ S1_lo[46] ^ S1_hi[18] ^ S1_hxl[31] ^ S1_hxl[10] ^ S2_lo[39] ^ S2_hxl[3];
    S1_is[3] = S1_ca[2] ^ S1_hxl[3];
    S1_lo[20] = S1_is[3] ^ S1_out[20] ^ S1_cb[19];
    S1_hi[20] = S1_lo[20] ^ S1_hxl[20];
    S2_lo[5] = S1_lo[20] ^ S1_hxl[5];
    S1_ca[20] = ((S1_ca[19] & S1_lo[20]) | (S1_lo[20] & S1_hi[20]) | (S1_ca[19] & S1_hi[20]));
    S1_is[21] = S1_ca[20] ^ S1_hxl[21];
    S1_ca[21] = ((S1_ca[20] & S1_lo[21]) | (S1_lo[21] & S1_hi[21]) | (S1_ca[20] & S1_hi[21]));
    S1_lo[38] = S1_is[21] ^ S1_out[38] ^ S1_cb[37];
    S1_hi[38] = S1_lo[38] ^ S1_hxl[38];
    S1_ca[38] = ((S1_ca[37] & S1_lo[38]) | (S1_lo[38] & S1_hi[38]) | (S1_ca[37] & S1_hi[38]));
    S1_is[39] = S1_ca[38] ^ S1_hxl[39];
    S1_cb[38] = ((S1_cb[37] & S1_is[21]) | (S1_is[21] & S1_lo[38]) | (S1_cb[37] & S1_lo[38]));
    S1_cb[54] = ((S1_cb[53] & S1_is[37]) | (S1_is[37] & S1_lo[54]) | (S1_cb[53] & S1_lo[54]));
    S1_lo[55] = S1_is[38] ^ S1_out[55] ^ S1_cb[54];
    S1_hi[55] = S1_lo[55] ^ S1_hxl[55];
    S1_hxl[61] = S1_lo[55] ^ S1_lo[19] ^ S1_lo[12] ^ S1_hi[19] ^ S2_lo[61] ^ S2_lo[40];
    S1_hxl[40] = S1_lo[55] ^ S1_lo[19] ^ S1_hi[19] ^ S2_lo[40];
    S2_hi[25] = S1_lo[55] ^ S1_lo[19] ^ S1_lo[12] ^ S1_hi[19] ^ S2_lo[61] ^ S2_lo[40];
    S2_hi[4] = S1_lo[55] ^ S1_lo[19] ^ S1_hi[19] ^ S2_lo[40];
    S2_hxl[4] = S1_lo[55] ^ S1_lo[4] ^ S1_hi[19] ^ S1_hi[4] ^ S2_lo[40];
    S1_cb[55] = ((S1_cb[54] & S1_is[38]) | (S1_is[38] & S1_lo[55]) | (S1_cb[54] & S1_lo[55]));
    S1_lo[56] = S1_is[39] ^ S1_out[56] ^ S1_cb[55];
    S1_hi[62] = S1_lo[62] ^ S1_lo[56] ^ S1_lo[20] ^ S1_lo[13] ^ S1_hi[20] ^ S2_lo[62] ^ S2_lo[41];
    S1_hi[56] = S1_lo[56] ^ S1_hxl[56];
    S1_hxl[62] = S1_lo[56] ^ S1_lo[20] ^ S1_lo[13] ^ S1_hi[20] ^ S2_lo[62] ^ S2_lo[41];
    S1_hxl[41] = S1_lo[56] ^ S1_lo[20] ^ S1_hi[20] ^ S2_lo[41];
    S2_hi[26] = S1_lo[56] ^ S1_lo[20] ^ S1_lo[13] ^ S1_hi[20] ^ S2_lo[62] ^ S2_lo[41];
    S2_hi[5] = S1_lo[56] ^ S1_lo[20] ^ S1_hi[20] ^ S2_lo[41];
    S2_hxl[5] = S1_lo[56] ^ S1_lo[5] ^ S1_hi[20] ^ S1_hi[5] ^ S2_lo[41];
    S1_ca[56] = ((S1_ca[55] & S1_lo[56]) | (S1_lo[56] & S1_hi[56]) | (S1_ca[55] & S1_hi[56]));
    S1_is[57] = S1_ca[56] ^ S1_hxl[57];
    S1_lo[10] = S1_is[57] ^ S1_out[10] ^ S1_cb[9];
    S1_hi[10] = S1_lo[10] ^ S1_hxl[10];
    S1_hxl[59] = S1_lo[10] ^ S1_hxl[38] ^ S2_lo[59];
    S2_hi[23] = S1_lo[10] ^ S1_hxl[38] ^ S2_lo[59];
    S1_cb[10] = ((S1_cb[9] & S1_is[57]) | (S1_is[57] & S1_lo[10]) | (S1_cb[9] & S1_lo[10]));
    S1_is[58] = S1_lo[11] ^ S1_out[11] ^ S1_cb[10];
    S1_ca[57] = S1_hxl[58] ^ S1_is[58];
    S1_cb[11] = ((S1_cb[10] & S1_is[58]) | (S1_is[58] & S1_lo[11]) | (S1_cb[10] & S1_lo[11]));
    S1_is[59] = S1_lo[12] ^ S1_out[12] ^ S1_cb[11];
    S1_ca[58] = S1_hxl[59] ^ S1_is[59];
    S1_cb[12] = ((S1_cb[11] & S1_is[59]) | (S1_is[59] & S1_lo[12]) | (S1_cb[11] & S1_lo[12]));
    S1_is[60] = S1_lo[13] ^ S1_out[13] ^ S1_cb[12];
    S1_ca[59] = S1_hxl[60] ^ S1_is[60];
    S1_cb[13] = ((S1_cb[12] & S1_is[60]) | (S1_is[60] & S1_lo[13]) | (S1_cb[12] & S1_lo[13]));
    S1_is[61] = S1_lo[14] ^ S1_out[14] ^ S1_cb[13];
    S1_ca[60] = S1_hxl[61] ^ S1_is[61];
    S1_cb[14] = ((S1_cb[13] & S1_is[61]) | (S1_is[61] & S1_lo[14]) | (S1_cb[13] & S1_lo[14]));
    S1_is[62] = S1_lo[15] ^ S1_out[15] ^ S1_cb[14];
    S1_ca[61] = S1_hxl[62] ^ S1_is[62];
    S1_ca[62] = ((S1_ca[61] & S1_lo[62]) | (S1_lo[62] & S1_hi[62]) | (S1_ca[61] & S1_hi[62]));
    S1_cb[15] = ((S1_cb[14] & S1_is[62]) | (S1_is[62] & S1_lo[15]) | (S1_cb[14] & S1_lo[15]));
    S1_is[63] = S1_lo[16] ^ S1_out[16] ^ S1_cb[15];
    S1_hxl[63] = S1_is[63] ^ S1_ca[62];
    S1_lo[57] = S1_lo[21] ^ S1_lo[14] ^ S1_hi[21] ^ S1_hxl[63] ^ S2_lo[63] ^ S2_lo[42];
    S1_hi[63] = S1_lo[63] ^ S1_hxl[63];
    S1_hi[57] = S1_lo[21] ^ S1_lo[14] ^ S1_hi[21] ^ S1_hxl[63] ^ S1_hxl[57] ^ S2_lo[63] ^ S2_lo[42];
    S1_hxl[42] = S1_lo[14] ^ S1_hxl[63] ^ S2_lo[63];
    S2_hi[27] = S1_hxl[63];
    S2_hi[6] = S1_lo[14] ^ S1_hxl[63] ^ S2_lo[63];
    S2_hxl[6] = S1_lo[21] ^ S1_lo[14] ^ S1_lo[6] ^ S1_hi[6] ^ S1_hxl[63] ^ S2_lo[63];
    S1_ca[63] = ((S1_ca[62] & S1_lo[63]) | (S1_lo[63] & S1_hi[63]) | (S1_ca[62] & S1_hi[63]));
    S1_cb[56] = ((S1_cb[55] & S1_is[39]) | (S1_is[39] & S1_lo[56]) | (S1_cb[55] & S1_lo[56]));
    S1_is[40] = S1_lo[57] ^ S1_out[57] ^ S1_cb[56];
    S1_ca[39] = S1_hxl[40] ^ S1_is[40];
    S1_cb[57] = ((S1_cb[56] & S1_is[40]) | (S1_is[40] & S1_lo[57]) | (S1_cb[56] & S1_lo[57]));
    S2_ca[3] = ((S2_ca[2] & S2_lo[3]) | (S2_lo[3] & S2_hi[3]) | (S2_ca[2] & S2_hi[3]));
    S2_is[4] = S2_ca[3] ^ S2_hxl[4];
    S2_ca[4] = ((S2_ca[3] & S2_lo[4]) | (S2_lo[4] & S2_hi[4]) | (S2_ca[3] & S2_hi[4]));
    S2_is[5] = S2_ca[4] ^ S2_hxl[5];
    S2_ca[5] = ((S2_ca[4] & S2_lo[5]) | (S2_lo[5] & S2_hi[5]) | (S2_ca[4] & S2_hi[5]));
    S2_is[6] = S2_ca[5] ^ S2_hxl[6];
    S2_ca[6] = ((S2_ca[5] & S2_lo[6]) | (S2_lo[6] & S2_hi[6]) | (S2_ca[5] & S2_hi[6]));
    S2_ca[21] = ((S2_ca[20] & S2_lo[21]) | (S2_lo[21] & S2_hi[21]) | (S2_ca[20] & S2_hi[21]));
    S2_hxl[22] = S2_is[22] ^ S2_ca[21];
    S1_hi[22] = S1_lo[37] ^ S1_lo[22] ^ S1_lo[1] ^ S1_hi[1] ^ S1_hxl[58] ^ S2_hxl[22];
    S1_hxl[22] = S1_lo[37] ^ S1_lo[1] ^ S1_hi[1] ^ S1_hxl[58] ^ S2_hxl[22];
    S2_lo[22] = S1_hxl[58] ^ S2_hxl[22];
    S2_hi[50] = S1_lo[37] ^ S1_lo[1] ^ S1_hi[1] ^ S1_hxl[58] ^ S2_hxl[22];
    S2_hxl[50] = S1_lo[50] ^ S1_lo[37] ^ S1_lo[29] ^ S1_hi[50] ^ S1_hi[29] ^ S1_hi[1] ^ S1_hxl[58] ^ S2_hxl[22];
    S1_is[22] = S1_ca[21] ^ S1_hxl[22];
    S1_ca[22] = ((S1_ca[21] & S1_lo[22]) | (S1_lo[22] & S1_hi[22]) | (S1_ca[21] & S1_hi[22]));
    S1_lo[39] = S1_is[22] ^ S1_out[39] ^ S1_cb[38];
    S1_hi[39] = S1_lo[39] ^ S1_hxl[39];
    S1_cb[39] = ((S1_cb[38] & S1_is[22]) | (S1_is[22] & S1_lo[39]) | (S1_cb[38] & S1_lo[39]));
    S2_ca[22] = ((S2_ca[21] & S2_lo[22]) | (S2_lo[22] & S2_hi[22]) | (S2_ca[21] & S2_hi[22]));
    S2_hxl[23] = S2_is[23] ^ S2_ca[22];
    S1_hi[23] = S1_lo[38] ^ S1_lo[23] ^ S1_lo[2] ^ S1_hi[2] ^ S1_hxl[59] ^ S2_hxl[23];
    S1_hxl[23] = S1_lo[38] ^ S1_lo[2] ^ S1_hi[2] ^ S1_hxl[59] ^ S2_hxl[23];
    S2_lo[23] = S1_hxl[59] ^ S2_hxl[23];
    S2_hi[51] = S1_lo[38] ^ S1_lo[2] ^ S1_hi[2] ^ S1_hxl[59] ^ S2_hxl[23];
    S2_hxl[51] = S1_lo[51] ^ S1_lo[38] ^ S1_lo[30] ^ S1_hi[51] ^ S1_hi[30] ^ S1_hi[2] ^ S1_hxl[59] ^ S2_hxl[23];
    S1_is[23] = S1_ca[22] ^ S1_hxl[23];
    S1_ca[23] = ((S1_ca[22] & S1_lo[23]) | (S1_lo[23] & S1_hi[23]) | (S1_ca[22] & S1_hi[23]));
    S1_lo[40] = S1_is[23] ^ S1_out[40] ^ S1_cb[39];
    S1_hi[40] = S1_lo[40] ^ S1_hxl[40];
    S1_ca[40] = ((S1_ca[39] & S1_lo[40]) | (S1_lo[40] & S1_hi[40]) | (S1_ca[39] & S1_hi[40]));
    S1_is[41] = S1_ca[40] ^ S1_hxl[41];
    S1_cb[40] = ((S1_cb[39] & S1_is[23]) | (S1_is[23] & S1_lo[40]) | (S1_cb[39] & S1_lo[40]));
    S1_lo[58] = S1_is[41] ^ S1_out[58] ^ S1_cb[57];
    S1_hi[58] = S1_lo[58] ^ S1_hxl[58];
    S1_hi[43] = S1_lo[58] ^ S1_lo[43] ^ S1_lo[22] ^ S1_hi[22] ^ S2_lo[43];
    S1_hxl[43] = S1_lo[58] ^ S1_lo[22] ^ S1_hi[22] ^ S2_lo[43];
    S2_hi[7] = S1_lo[58] ^ S1_lo[22] ^ S1_hi[22] ^ S2_lo[43];
    S2_hxl[7] = S1_lo[58] ^ S1_lo[7] ^ S1_hi[22] ^ S1_hi[7] ^ S2_lo[43];
    S1_cb[58] = ((S1_cb[57] & S1_is[41]) | (S1_is[41] & S1_lo[58]) | (S1_cb[57] & S1_lo[58]));
    S2_is[7] = S2_ca[6] ^ S2_hxl[7];
    S2_ca[7] = ((S2_ca[6] & S2_lo[7]) | (S2_lo[7] & S2_hi[7]) | (S2_ca[6] & S2_hi[7]));
    S2_ca[23] = ((S2_ca[22] & S2_lo[23]) | (S2_lo[23] & S2_hi[23]) | (S2_ca[22] & S2_hi[23]));
    S2_hxl[24] = S2_is[24] ^ S2_ca[23];
    S1_hi[24] = S1_lo[39] ^ S1_lo[24] ^ S1_lo[3] ^ S1_hi[3] ^ S1_hxl[60] ^ S2_hxl[24];
    S1_hxl[24] = S1_lo[39] ^ S1_lo[3] ^ S1_hi[3] ^ S1_hxl[60] ^ S2_hxl[24];
    S2_lo[24] = S1_hxl[60] ^ S2_hxl[24];
    S2_hi[52] = S1_lo[39] ^ S1_lo[3] ^ S1_hi[3] ^ S1_hxl[60] ^ S2_hxl[24];
    S2_hxl[52] = S1_lo[39] ^ S1_lo[31] ^ S1_hi[31] ^ S1_hi[3] ^ S1_hxl[60] ^ S1_hxl[52] ^ S2_hxl[24];
    S1_is[24] = S1_ca[23] ^ S1_hxl[24];
    S1_ca[24] = ((S1_ca[23] & S1_lo[24]) | (S1_lo[24] & S1_hi[24]) | (S1_ca[23] & S1_hi[24]));
    S1_lo[41] = S1_is[24] ^ S1_out[41] ^ S1_cb[40];
    S1_hi[41] = S1_lo[41] ^ S1_hxl[41];
    S2_lo[26] = S1_lo[41] ^ S1_lo[26] ^ S1_lo[5] ^ S1_hi[26] ^ S1_hi[5];
    S2_hxl[26] = S1_lo[41] ^ S1_lo[26] ^ S1_lo[5] ^ S1_hi[26] ^ S1_hi[5] ^ S1_hxl[62];
    S1_ca[41] = ((S1_ca[40] & S1_lo[41]) | (S1_lo[41] & S1_hi[41]) | (S1_ca[40] & S1_hi[41]));
    S1_is[42] = S1_ca[41] ^ S1_hxl[42];
    S1_cb[41] = ((S1_cb[40] & S1_is[24]) | (S1_is[24] & S1_lo[41]) | (S1_cb[40] & S1_lo[41]));
    S1_lo[59] = S1_is[42] ^ S1_out[59] ^ S1_cb[58];
    S1_hi[59] = S1_lo[59] ^ S1_hxl[59];
    S1_hi[44] = S1_lo[59] ^ S1_lo[44] ^ S1_lo[23] ^ S1_hi[23] ^ S2_lo[44];
    S1_hxl[44] = S1_lo[59] ^ S1_lo[23] ^ S1_hi[23] ^ S2_lo[44];
    S2_hi[8] = S1_lo[59] ^ S1_lo[23] ^ S1_hi[23] ^ S2_lo[44];
    S2_hxl[8] = S1_lo[59] ^ S1_lo[8] ^ S1_hi[23] ^ S1_hi[8] ^ S2_lo[44];
    S1_cb[59] = ((S1_cb[58] & S1_is[42]) | (S1_is[42] & S1_lo[59]) | (S1_cb[58] & S1_lo[59]));
    S2_is[8] = S2_ca[7] ^ S2_hxl[8];
    S2_ca[8] = ((S2_ca[7] & S2_lo[8]) | (S2_lo[8] & S2_hi[8]) | (S2_ca[7] & S2_hi[8]));
    S2_ca[24] = ((S2_ca[23] & S2_lo[24]) | (S2_lo[24] & S2_hi[24]) | (S2_ca[23] & S2_hi[24]));
    S2_hxl[25] = S2_is[25] ^ S2_ca[24];
    S1_hi[25] = S1_lo[40] ^ S1_lo[25] ^ S1_lo[4] ^ S1_hi[4] ^ S1_hxl[61] ^ S2_hxl[25];
    S1_hxl[25] = S1_lo[40] ^ S1_lo[4] ^ S1_hi[4] ^ S1_hxl[61] ^ S2_hxl[25];
    S2_lo[25] = S1_hxl[61] ^ S2_hxl[25];
    S2_hi[53] = S1_lo[40] ^ S1_lo[4] ^ S1_hi[4] ^ S1_hxl[61] ^ S2_hxl[25];
    S2_hxl[53] = S1_lo[53] ^ S1_lo[40] ^ S1_lo[32] ^ S1_hi[53] ^ S1_hi[32] ^ S1_hi[4] ^ S1_hxl[61] ^ S2_hxl[25];
    S1_is[25] = S1_ca[24] ^ S1_hxl[25];
    S1_lo[42] = S1_is[25] ^ S1_out[42] ^ S1_cb[41];
    S1_hi[42] = S1_lo[42] ^ S1_hxl[42];
    S2_lo[27] = S1_lo[42] ^ S1_lo[27] ^ S1_lo[6] ^ S1_hi[27] ^ S1_hi[6];
    S2_hxl[27] = S1_lo[42] ^ S1_lo[27] ^ S1_lo[6] ^ S1_hi[27] ^ S1_hi[6] ^ S1_hxl[63];
    S1_ca[42] = ((S1_ca[41] & S1_lo[42]) | (S1_lo[42] & S1_hi[42]) | (S1_ca[41] & S1_hi[42]));
    S1_is[43] = S1_ca[42] ^ S1_hxl[43];
    S1_ca[43] = ((S1_ca[42] & S1_lo[43]) | (S1_lo[43] & S1_hi[43]) | (S1_ca[42] & S1_hi[43]));
    S1_is[44] = S1_ca[43] ^ S1_hxl[44];
    S1_ca[44] = ((S1_ca[43] & S1_lo[44]) | (S1_lo[44] & S1_hi[44]) | (S1_ca[43] & S1_hi[44]));
    S1_lo[60] = S1_is[43] ^ S1_out[60] ^ S1_cb[59];
    S1_hi[60] = S1_lo[60] ^ S1_hxl[60];
    S1_hi[45] = S1_lo[60] ^ S1_lo[45] ^ S1_lo[24] ^ S1_hi[24] ^ S2_lo[45];
    S1_hxl[45] = S1_lo[60] ^ S1_lo[24] ^ S1_hi[24] ^ S2_lo[45];
    S2_hi[9] = S1_lo[60] ^ S1_lo[24] ^ S1_hi[24] ^ S2_lo[45];
    S2_hxl[9] = S1_lo[60] ^ S1_lo[9] ^ S1_hi[24] ^ S1_hi[9] ^ S2_lo[45];
    S1_is[45] = S1_ca[44] ^ S1_hxl[45];
    S1_ca[45] = ((S1_ca[44] & S1_lo[45]) | (S1_lo[45] & S1_hi[45]) | (S1_ca[44] & S1_hi[45]));
    S1_cb[60] = ((S1_cb[59] & S1_is[43]) | (S1_is[43] & S1_lo[60]) | (S1_cb[59] & S1_lo[60]));
    S1_lo[61] = S1_is[44] ^ S1_out[61] ^ S1_cb[60];
    S1_hi[61] = S1_lo[61] ^ S1_hxl[61];
    S1_hi[46] = S1_lo[61] ^ S1_lo[46] ^ S1_lo[25] ^ S1_hi[25] ^ S2_lo[46];
    S1_hxl[46] = S1_lo[61] ^ S1_lo[25] ^ S1_hi[25] ^ S2_lo[46];
    S2_hi[10] = S1_lo[61] ^ S1_lo[25] ^ S1_hi[25] ^ S2_lo[46];
    S2_hxl[10] = S1_lo[61] ^ S1_lo[10] ^ S1_hi[25] ^ S1_hi[10] ^ S2_lo[46];
    S1_is[46] = S1_ca[45] ^ S1_hxl[46];
    S1_cb[61] = ((S1_cb[60] & S1_is[44]) | (S1_is[44] & S1_lo[61]) | (S1_cb[60] & S1_lo[61]));
    S1_cb[62] = ((S1_cb[61] & S1_is[45]) | (S1_is[45] & S1_lo[62]) | (S1_cb[61] & S1_lo[62]));
    S1_cb[63] = ((S1_cb[62] & S1_is[46]) | (S1_is[46] & S1_lo[63]) | (S1_cb[62] & S1_lo[63]));
    S2_is[9] = S2_ca[8] ^ S2_hxl[9];
    S2_ca[9] = ((S2_ca[8] & S2_lo[9]) | (S2_lo[9] & S2_hi[9]) | (S2_ca[8] & S2_hi[9]));
    S2_is[10] = S2_ca[9] ^ S2_hxl[10];
    S2_ca[25] = ((S2_ca[24] & S2_lo[25]) | (S2_lo[25] & S2_hi[25]) | (S2_ca[24] & S2_hi[25]));
    S2_ca[26] = ((S2_ca[25] & S2_lo[26]) | (S2_lo[26] & S2_hi[26]) | (S2_ca[25] & S2_hi[26]));
    S2_ca[29] = ((S2_ca[28] & S2_lo[29]) | (S2_lo[29] & S2_hi[29]) | (S2_ca[28] & S2_hi[29]));
    S2_is[30] = S2_ca[29] ^ S2_hxl[30];
    S2_ca[30] = ((S2_ca[29] & S2_lo[30]) | (S2_lo[30] & S2_hi[30]) | (S2_ca[29] & S2_hi[30]));
    S2_is[31] = S2_ca[30] ^ S2_hxl[31];
    S2_is[50] = S2_ca[49] ^ S2_hxl[50];
    S2_ca[50] = ((S2_ca[49] & S2_lo[50]) | (S2_lo[50] & S2_hi[50]) | (S2_ca[49] & S2_hi[50]));
    S2_is[51] = S2_ca[50] ^ S2_hxl[51];
    S2_ca[51] = ((S2_ca[50] & S2_lo[51]) | (S2_lo[51] & S2_hi[51]) | (S2_ca[50] & S2_hi[51]));
    S2_is[52] = S2_ca[51] ^ S2_hxl[52];
    S2_ca[52] = ((S2_ca[51] & S2_lo[52]) | (S2_lo[52] & S2_hi[52]) | (S2_ca[51] & S2_hi[52]));
    S2_cb[3] = ((S2_cb[2] & S2_is[50]) | (S2_is[50] & S2_lo[3]) | (S2_cb[2] & S2_lo[3]));
    S2_cb[4] = ((S2_cb[3] & S2_is[51]) | (S2_is[51] & S2_lo[4]) | (S2_cb[3] & S2_lo[4]));
    S2_cb[21] = ((S2_cb[20] & S2_is[4]) | (S2_is[4] & S2_lo[21]) | (S2_cb[20] & S2_lo[21]));
    S2_cb[22] = ((S2_cb[21] & S2_is[5]) | (S2_is[5] & S2_lo[22]) | (S2_cb[21] & S2_lo[22]));
    S2_cb[23] = ((S2_cb[22] & S2_is[6]) | (S2_is[6] & S2_lo[23]) | (S2_cb[22] & S2_lo[23]));
    S2_cb[24] = ((S2_cb[23] & S2_is[7]) | (S2_is[7] & S2_lo[24]) | (S2_cb[23] & S2_lo[24]));
    S2_cb[25] = ((S2_cb[24] & S2_is[8]) | (S2_is[8] & S2_lo[25]) | (S2_cb[24] & S2_lo[25]));
    S2_cb[26] = ((S2_cb[25] & S2_is[9]) | (S2_is[9] & S2_lo[26]) | (S2_cb[25] & S2_lo[26]));
    S2_cb[46] = ((S2_cb[45] & S2_is[29]) | (S2_is[29] & S2_lo[46]) | (S2_cb[45] & S2_lo[46]));
    S2_cb[47] = ((S2_cb[46] & S2_is[30]) | (S2_is[30] & S2_lo[47]) | (S2_cb[46] & S2_lo[47]));
    S2_cb[63] = ((S2_cb[62] & S2_is[46]) | (S2_is[46] & S2_lo[63]) | (S2_cb[62] & S2_lo[63]));
    //  END   ▲▲▲▲▲
    // ----------------------------------------------------------------------------

    // ----------------------------------------------------------------------------
    //  4.   Assemble the two 64-bit results from bit slices
    // ----------------------------------------------------------------------------
    uint64_t result_lo = 0ULL;
    uint64_t result_hi = 0ULL;
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        result_lo |= (static_cast<uint64_t>(S1_lo[i]) & 1ULL) << i;
        result_hi |= (static_cast<uint64_t>(S1_hi[i]) & 1ULL) << i;
    }

    // ----------------------------------------------------------------------------
    //  5.   Validate; if good, push to results buffer via atomic index
    // ----------------------------------------------------------------------------
    if (isValid(result_lo, result_hi, nextlong1, nextlong2))
    {
        unsigned int idx = atomicInc(d_resIndex, RESULTS_BUFFER_SIZE - 1);
        if (idx < RESULTS_BUFFER_SIZE)    // simple overflow protection
        {
            d_results[idx].guess_bits = guess & ((1ULL << 42) - 1);
            d_results[idx].result_lo  = result_lo;
            d_results[idx].result_hi  = result_hi;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
//  Host-side helpers
// ──────────────────────────────────────────────────────────────────────
static void usage(const char *prog)
{
    std::cerr
        << "Usage: " << prog
        << " -nl1 <nextlong1> -nl2 <nextlong2> [-d <cudaDevice>] "
        << "[-s <startGuess>] [-e <endGuess>]\n";
}

static unsigned long long atoull(const char *s)
{
    char *end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 0);
    if (end == s || *end != '\0') {
        std::cerr << "Invalid numeric argument: " << s << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return v;
}

// ──────────────────────────────────────────────────────────────────────
//  main()
// ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    if (argc < 5) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    int deviceId = 0;
    uint64_t nextlong1 = 0, nextlong2 = 0;
    unsigned long long startGuess = 0, endGuess = (1ULL << 42);   // full space

    // Very light-weight argument parse
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-nl1") && i + 1 < argc) {
            nextlong1 = atoull(argv[++i]);
        } else if (!std::strcmp(argv[i], "-nl2") && i + 1 < argc) {
            nextlong2 = atoull(argv[++i]);
        } else if (!std::strcmp(argv[i], "-d") && i + 1 < argc) {
            deviceId = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) {
            startGuess = atoull(argv[++i]);
        } else if (!std::strcmp(argv[i], "-e") && i + 1 < argc) {
            endGuess = atoull(argv[++i]);
        } else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // -----------------------------------------------------------------
    //  Select GPU and report
    // -----------------------------------------------------------------
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "Running on GPU " << deviceId << " ["
              << prop.name << "]\n";

    // -----------------------------------------------------------------
    //  Allocate device buffers
    // -----------------------------------------------------------------
    Result *d_results = nullptr;
    unsigned int *d_resIndex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, RESULTS_BUFFER_SIZE * sizeof(Result)));
    CUDA_CHECK(cudaMalloc(&d_resIndex, sizeof(unsigned int)));

    // Host-side result scratch
    std::vector<Result> h_results(RESULTS_BUFFER_SIZE);

    // -----------------------------------------------------------------
    //  Iterate batches
    // -----------------------------------------------------------------
    unsigned long long totalTested = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (unsigned long long batchBase = startGuess;
         batchBase < endGuess;
         batchBase += BATCH_SIZE)
    {
        // Reset result index to 0
        unsigned int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_resIndex, &zero, sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // Launch grid
        dim3 grid(BLOCK_SIZE);
        dim3 block(THREAD_SIZE);

        bruteKernel<<<grid, block>>>(nextlong1, nextlong2, batchBase,
                                     d_results, d_resIndex);
        CUDA_CHECK(cudaGetLastError());

        // Copy back results index
        unsigned int h_resCount = 0;
        CUDA_CHECK(cudaMemcpy(&h_resCount, d_resIndex, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        if (h_resCount > RESULTS_BUFFER_SIZE)
            h_resCount = RESULTS_BUFFER_SIZE;

        if (h_resCount) {
            CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
                                  h_resCount * sizeof(Result),
                                  cudaMemcpyDeviceToHost));

            std::cout << '\n';
            for (unsigned int i = 0; i < h_resCount; ++i) {
                const auto &r = h_results[i];
                std::cout << "FOUND - guess 0x" << std::hex << r.guess_bits
                          << "  result_lo:0x" << r.result_lo
                          << "  result_hi:0x" << r.result_hi << std::dec << '\n';
            }
        }

        totalTested += BATCH_SIZE;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsedSec = std::chrono::duration<double>(now - t_start).count();
        double MHs = (totalTested / 1e6) / elapsedSec;

        std::cout << "\rBatch complete. Tested "
                  << std::setw(10) << totalTested
                  << " inputs [" << std::fixed << std::setprecision(2)
                  << MHs << " MH/s]" << std::flush;
    }

    std::cout << "\nAll batches finished.\n";

    // -----------------------------------------------------------------
    //  Cleanup
    // -----------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_resIndex));
    return 0;
}
