#include <cstdint>
#include <cstdlib>
#include <iostream>

struct Xoroshiro128PlusPlus {
    using u64 = uint64_t;

    static constexpr u64 kSilver = 0x6a09e667f3bcc909ULL;
    static constexpr u64 kGolden = 0x9e3779b97f4a7c15ULL;
    static constexpr u64 kMix1   = 0xbf58476d1ce4e5b9ULL;
    static constexpr u64 kMix2   = 0x94d049bb133111ebULL;

    u64 lo = 0, hi = 0;

    static u64 rotl(u64 x, int r) {
        return (x << r) | (x >> (64 - r));
    }

    static u64 mix64(u64 x) {
        x = (x ^ (x >> 30)) * kMix1;
        x = (x ^ (x >> 27)) * kMix2;
        return x ^ (x >> 31);
    }

    void xSetSeed(u64 s) {
        s ^= kSilver;
        lo = mix64(s);
        hi = mix64(s + kGolden);
    }

    u64 next64() {
        u64 l = lo, h = hi;
        u64 result = rotl(l + h, 17) + l;
        h ^= l;
        lo = rotl(l, 49) ^ h ^ (h << 21);
        hi = rotl(h, 28);
        return result;
    }

    int64_t nextLongJ() {
        int32_t high = static_cast<int32_t>(next64() >> 32);
        int32_t low  = static_cast<int32_t>(next64() >> 32);
        return (static_cast<int64_t>(high) << 32) + static_cast<int64_t>(low);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <worldSeed> <chunkX> <chunkZ>\n";
        return EXIT_FAILURE;
    }

    uint64_t worldSeed = std::strtoull(argv[1], nullptr, 10);
    int64_t chunkX = std::strtoll(argv[2], nullptr, 10);
    int64_t chunkZ = std::strtoll(argv[3], nullptr, 10);

    Xoroshiro128PlusPlus prng;
    prng.xSetSeed(worldSeed);

    int64_t a = prng.nextLongJ() | 1LL;
    int64_t b = prng.nextLongJ() | 1LL;

    // blockX = 16 * chunkX, blockZ = 16 * chunkZ
    int64_t decorationSeed = (16 * chunkX * a + 16 * chunkZ * b) ^ static_cast<int64_t>(worldSeed);

    std::cout << "a = " << a
              << "\nb = " << b
              << "\ndecorationSeed = " << decorationSeed << "\n";

    return EXIT_SUCCESS;
}
