/*
 * optimizer.cpp – C++17 port of the Python randomized‑optimisation algorithm.
 *
 * Major external dependencies (all open‑source):
 *   • Eigen 3 (header‑only)      – dense vector/matrix math, SIMD‑friendly
 *   • cnpy                       – read/write NumPy .npz archives
 *   • zlib                       – (pulled in by cnpy)
 *
 * The program reproduces the Python behaviour as closely as practical:
 *   – Identical command‑line interface and file IO (checkpoint.npz, boinc_* files).
 *   – Same random search / inference logic (GF(2) linear span + custom rules).
 *   – Deterministic restarts via serialised std::mt19937 state.
 *   – Output formatting (progress bar, summary line, hex bitstring) preserved.
 *
 * Build example:
 *   g++ -std=c++17 -O3 -pipe -Icnpy -I/usr/include/eigen3 optimizer.cpp cnpy/cnpy.cpp -static-libstdc++ -static-libgcc -lz -o optimizer
 *   g++ -std=c++17 -O3 -flto -ffast-math -DNDEBUG -funroll-loops -fomit-frame-pointer -pipe -Icnpy -I/usr/include/eigen3 optimizer5.cpp cnpy/cnpy.cpp -static-libstdc++ -static-libgcc -lz -o optim
izer5
 *   g++ -std=c++17 -O3 -DNDEBUG -flto -funroll-loops -fomit-frame-pointer -ffast-math -pipe -Icnpy -I/usr/include/eigen3 optimizer5.cpp cnpy/cnpy.cpp -o optimizer5 -static-libstdc++ -static-libgcc -lz
 */

#include <cnpy.h>
 // Eigen may be installed as either
 //   /usr/include/Eigen/...
 // or /usr/include/eigen3/Eigen/...
 // The following conditional include makes the code portable.
#if __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#elif __has_include(<eigen3/Eigen/Dense>)
#include <eigen3/Eigen/Dense>
#else
#error "Eigen headers not found – install libeigen3-dev (Ubuntu) or add -I<eigen_path> to the compiler flags."
#endif

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ──────────────────────────────── Type aliases ──────────────────────────────
using Byte = uint8_t;
using VecXb = Eigen::Array<Byte, Eigen::Dynamic, 1>;            // 0/1 vector
using VecXi = Eigen::Array<int, Eigen::Dynamic, 1>;            // signed‑int vector
using MatXb = Eigen::Array<Byte, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using UIntVec = std::vector<std::size_t>;

constexpr Byte B0 = static_cast<Byte>(0);
constexpr Byte B1 = static_cast<Byte>(1);

// ─────────────────────────────── Utility helpers ────────────────────────────

// Count set bits in a VecXb (returns size_t)
inline std::size_t countOnes(const VecXb& v) { return v.cast<int>().sum(); }

// Left‑pad a binary string to 4‑bit boundary, convert to lower‑case hex
std::string binaryToHex(const VecXb& bits)
{
    const std::size_t n = static_cast<std::size_t>(bits.size());
    const std::size_t pad = (4 - (n % 4)) % 4;

    std::string bin(pad, '0');
    bin.reserve(pad + n);
    for (Byte b : bits) bin.push_back('0' + static_cast<char>(b));

    std::string hex;
    hex.reserve(bin.size() / 4);
    for (std::size_t i = 0; i < bin.size(); i += 4)
    {
        std::bitset<4> bs(bin.substr(i, 4));
        std::ostringstream h; h << std::hex << bs.to_ulong();
        hex += h.str();
    }
    return hex;
}

// Compute #1 bits per label (label values assumed ≥0, densely packed but we
// handle gaps).
std::vector<int> groupCountsFor(const VecXi& labels, const VecXb& bits)
{
    const int maxLbl = labels.maxCoeff();
    std::vector<int> counts(maxLbl + 1, 0);
    for (int i = 0; i < labels.size(); ++i)
        if (bits(i) == B1) ++counts[labels(i)];
    return counts;
}

// Write BOINC‑style progress + stdout info
void writeProgress(double frac, std::size_t score, const std::vector<int>& groupCounts)
{
    std::cout << std::fixed << std::setprecision(5) << frac << ' ' << score << " [";
    for (std::size_t i = 0; i < groupCounts.size(); ++i)
    {
        if (i) std::cout << ',';
        std::cout << groupCounts[i];
    }
    std::cout << "]\n";
    std::ofstream f("boinc_frac");
    f << std::fixed << std::setprecision(5) << frac;
}

/* ========================================================================
 *  ultra_fast_inference.h   (v2025-07-10)
 *
 *  – Fully static packing of `full`, `ante`, `cons` (built once – ever).
 *  – Flat, cache-friendly row layout (one big aligned buffer).
 *  – No per-iteration heap-allocation; all scratch is thread-local & reused.
 *  – Tight inner loops with SIMD-friendly #pragma omp simd unrolling.
 *  – Optional OpenMP parallelism (define OPENMP or compile with -fopenmp).
 *  – Rank / dependency test keeps a compact triangular “basis” in-place.
 * ====================================================================== */
#ifndef ULTRA_FAST_INFERENCE_H_
#define ULTRA_FAST_INFERENCE_H_

#include <vector>
#include <cstdint>
#include <cstring>
#include <array>
#include <algorithm>

#if defined(_OPENMP)
#  include <omp.h>
#endif

 /* ───────────────────────── low-level helpers ─────────────────────────── */

namespace fast_inf {

    /* trailing-zeros – defined for x==0 → 64 */
    inline int ctz64(uint64_t x) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        return x ? __builtin_ctzll(x) : 64;
#else
        /* portable fallback */
        static const int tbl[64] = {
            0,1,2,57,3,61,58,42,4,33,62,52,59,47,43,22,
            5,16,34,11,63,51,53,41,60,40,48,31,44,21,23,15,
            6,56,32,60,12,50,10,46,17,55,35,28,64,54,13,45,
            7,30,18,27,36,49,29,14,19,38,37,26,39,25,24,20 };
        return x ? tbl[((x & -x) * 0x03F79D71B4CB0A89ULL) >> 58] : 64;
#endif
    }

    /* small aligned flexible buffer ------------------------------------------------ */
    struct AlignedBuffer
    {
        uint64_t* ptr = nullptr;
        std::size_t sz = 0;                 /* elements */

        ~AlignedBuffer() { std::free(ptr); }

        void resize(std::size_t n)
        {
            if (n <= sz) return;
            std::free(ptr);
            ptr = static_cast<uint64_t*>(std::aligned_alloc(64, n * sizeof(uint64_t)));
            sz = n;
        }
    };

    /* Packed row-major GF(2) matrix – **flat & aligned** ------------------ */
    struct Packed
    {
        int rows = 0, cols = 0, words = 0;
        std::vector<uint64_t> data;         /* length = rows*words, 64-byte aligned */

        void build(const MatXb& M)
        {
            if (rows) return;               /* already built */
            rows = static_cast<int>(M.rows());
            cols = static_cast<int>(M.cols());
            words = (cols + 63) >> 6;
            data.resize(std::size_t(rows) * words);
            std::fill(data.begin(), data.end(), 0ULL);

            /* pack bit-matrix ---------------------------------------------------- */
#pragma omp parallel for schedule(static) if (rows > 64 && defined(_OPENMP))
            for (int r = 0; r < rows; ++r)
            {
                const Byte* src = M.row(r).data();
                uint64_t* dst = data.data() + std::size_t(r) * words;
                for (int c = 0; c < cols; ++c)
                    if (src[c]) dst[c >> 6] |= 1ULL << (c & 63);
            }
        }

        /* read-only row pointer */
        inline const uint64_t* row(int r) const noexcept
        {
            return data.data() + std::size_t(r) * words;
        }
    };

    /* Convert packed mask → VecXb ---------------------------------------- */
    inline VecXb to_vec(const uint64_t* mask, int cols, int words)
    {
        VecXb out(cols);  out.setZero();
        for (int w = 0; w < words; ++w)
        {
            uint64_t v = mask[w];
            int base = w << 6;
            while (v)
            {
                int bit = ctz64(v);
                out(base + bit) = B1;
                v &= v - 1;                 /* clear lowest bit */
            }
        }
        return out;
    }

    /* Static cache (constructed only once) -------------------------------- */
    struct Cache
    {
        Packed FULL, ANTE, CONS;
        int words = 0;

        Cache(const MatXb& f, const MatXb& a, const MatXb& c)
        {
            FULL.build(f);
            ANTE.build(a);
            CONS.build(c);
            words = FULL.words;
        }
    };

    /* global (function-local static) accessor – thread-safe in C++11 ------ */
    inline Cache& cache(const MatXb& f, const MatXb& a, const MatXb& c)
    {
        static Cache cinst(f, a, c);
        return cinst;
    }

    /* ───────────────────── dependency check (GF(2)) ─────────────────────── */

        /* returns {rank_of_selected, dependent_mask ( over FULL.rows )} */
    inline std::pair<int, VecXb>
        dependent_subset(const VecXb& sel, const Packed& P)
    {
        const int n = P.rows;
        const int words = P.words;

        /* thread-local scratch -------------------------------------------------- */
        thread_local AlignedBuffer scratch;
        scratch.resize(words);

        std::vector<uint64_t> basis;    basis.reserve(std::size_t(n) * words);
        std::vector<int>      pivots;   pivots.reserve(n);

        /* ---- build basis from selected rows ---------------------------------- */
        for (int r = 0; r < n; ++r)
            if (sel(r))
            {
                uint64_t* row = scratch.ptr;
                std::memcpy(row, P.row(r), std::size_t(words) * 8);

                /* eliminate against current basis ---------------------------- */
                for (std::size_t k = 0; k < pivots.size(); ++k)
                {
                    const int p = pivots[k];
                    if (row[p >> 6] & (1ULL << (p & 63)))
                    {
                        const uint64_t* src = basis.data() + k * words;
#pragma omp simd
                        for (int w = 0; w < words; ++w) row[w] ^= src[w];
                    }
                }
                /* find pivot -------------------------------------------------- */
                int pivot = -1;
                for (int w = 0; w < words && pivot == -1; ++w)
                    if (row[w]) pivot = (w << 6) + ctz64(row[w]);

                if (pivot != -1)
                {
                    /* store new basis row ------------------------------------ */
                    basis.resize(basis.size() + words);
                    std::memcpy(basis.data() + (basis.size() - words), row, std::size_t(words) * 8);
                    pivots.push_back(pivot);
                }
            }

        const int rankVal = static_cast<int>(pivots.size());
        VecXb     dep = VecXb::Zero(n);

        /* ---- classify the remaining rows ------------------------------------ */
#pragma omp parallel for schedule(static) if (n > 128 && defined(_OPENMP))
        for (int r = 0; r < n; ++r)
        {
            if (sel(r)) continue;

            uint64_t localBuf[512];                    /* up to 512*64 = 32768 cols */
            uint64_t* row = localBuf;
            std::memcpy(row, P.row(r), std::size_t(words) * 8);

            for (std::size_t k = 0; k < pivots.size(); ++k)
            {
                const int p = pivots[k];
                if (row[p >> 6] & (1ULL << (p & 63)))
                {
                    const uint64_t* src = basis.data() + k * words;
#pragma omp simd
                    for (int w = 0; w < words; ++w) row[w] ^= src[w];
                }
            }
            bool zero = true;
#pragma omp simd reduction(&:zero)
            for (int w = 0; w < words; ++w) zero &= !row[w];
            if (zero) dep(r) = B1;
        }

        return { rankVal, dep };
    }

    /* ───────────────────── apply inference rules (ante → cons) ──────────── */

    inline VecXb apply_rules(const VecXb& V, const Packed& A, const Packed& C)
    {
        const int R = A.rows;
        const int words = A.words;
        const int cols = A.cols;

        /* pack V into bit mask ------------------------------------------------- */
        thread_local AlignedBuffer vmaskBuf, newBitsBuf;
        vmaskBuf.resize(words);
        newBitsBuf.resize(words);
        uint64_t* vMask = vmaskBuf.ptr;
        uint64_t* newBits = newBitsBuf.ptr;
        std::fill(vMask, vMask + words, 0ULL);
        std::fill(newBits, newBits + words, 0ULL);

        for (int i = 0; i < cols; ++i)
            if (V(i)) vMask[i >> 6] |= 1ULL << (i & 63);

        /* scan every rule row -------------------------------------------------- */
#pragma omp parallel for schedule(static) if (R > 128 && defined(_OPENMP))
        for (int r = 0; r < R; ++r)
        {
            const uint64_t* ante = A.row(r);
            bool satisfied = true;

#pragma omp simd reduction(&:satisfied)
            for (int w = 0; w < words; ++w)
                satisfied &= !(ante[w] & ~vMask[w]);

            if (!satisfied) continue;

            const uint64_t* cons = C.row(r);
#pragma omp simd
            for (int w = 0; w < words; ++w)
            {
#pragma omp atomic
                newBits[w] |= cons[w];
            }
        }

        /* keep only new symbols ---------------------------------------------- */
#pragma omp simd
        for (int w = 0; w < words; ++w) newBits[w] &= ~vMask[w];

        return to_vec(newBits, cols, words);
    }

} /* namespace fast_inf */


/* ─────────────────────── fixed-point closure ─────────────────────────── */

inline std::pair<VecXb, std::size_t> fill_inferable_symbols(
    const VecXb& start,
    const MatXb& full,
    const MatXb& ante,
    const MatXb& cons)
{
    using namespace fast_inf;

    Cache& C = cache(full, ante, cons);          /* built once, thereafter free */
    VecXb  cur = start;
    std::size_t evals = 0;

    while (true)
    {
        auto [rank, dep] = dependent_subset(cur, C.FULL);
        ++evals;
        VecXb nxt = cur.max(dep);

        VecXb inf = apply_rules(nxt, C.ANTE, C.CONS);
        ++evals;
        nxt = nxt.max(inf);

        if ((nxt == cur).all()) break;
        cur.swap(nxt);
    }
    return { cur, evals };
}

#endif /* ULTRA_FAST_INFERENCE_H_ */

// ───────────────────── Random initial vector subject to counts ──────────────
VecXb generate_random_binary_vector(const VecXi& labels,
    const std::vector<int>& counts,
    std::mt19937& rng)
{
    if (labels.size() == 0) return VecXb::Zero(0);

    const int maxLbl = labels.maxCoeff();
    if (maxLbl >= static_cast<int>(counts.size()))
        throw std::runtime_error(
            std::string("label exceeds counts array: maxLbl=") + std::to_string(maxLbl)
            + ", counts.size()=" + std::to_string(counts.size()));

    // Build index buckets per label
    std::unordered_map<int, std::vector<int>> buckets;
    for (int i = 0; i < labels.size(); ++i) buckets[labels(i)].push_back(i);

    VecXb bits = VecXb::Zero(labels.size());
    for (int l = 0; l < static_cast<int>(counts.size()); ++l)
    {
        int cnt = counts[l]; if (cnt == 0) continue;
        auto& idxs = buckets[l];
        if (static_cast<int>(idxs.size()) < cnt)
            throw std::runtime_error(
                std::string("not enough indices for label count: label=") + std::to_string(l)
                + ", available=" + std::to_string(idxs.size())
                + ", required=" + std::to_string(cnt));
        std::shuffle(idxs.begin(), idxs.end(), rng);
        for (int k = 0; k < cnt; ++k) bits(idxs[k]) = B1;
    }
    return bits;
}

// ─────────────────────────── Swap iterator (random) ─────────────────────────
class SwapIterator
{
public:
    SwapIterator(const VecXb& bits,
        const VecXi& labels,
        std::mt19937& rng)
        : rng_(rng)
    {
        if (bits.size() != labels.size())
            throw std::runtime_error(
                "bits/labels size mismatch: bits.size()=" + std::to_string(bits.size())
                + ", labels.size()=" + std::to_string(labels.size()));

        // Build full groups of all positive labels
        for (int idx = 0; idx < bits.size(); ++idx) {
            int lbl = labels(idx);
            if (lbl <= 0) continue;
            auto& grp = groups_[lbl];
            if (bits(idx) == B1)
                grp.first.push_back(idx);
            else
                grp.second.push_back(idx);
        }

        // Gather all labels (we'll pick hi and lo from here)
        labels_.reserve(groups_.size());
        for (auto& kv : groups_)
            labels_.push_back(kv.first);
        std::shuffle(labels_.begin(), labels_.end(), rng_);

        // prime the iteration
        advance_hi_label();
    }

    /// Return true and set (i,j) if there is a next swap.  Otherwise return false.
    bool next(int& i, int& j)
    {
        if (done_) return false;

        i = hi_ones_[i_idx_];
        j = lo_zeros_[j_idx_];

        // advance inner-most
        if (++j_idx_ >= lo_zeros_.size()) {
            j_idx_ = 0;
            // advance ones
            if (++i_idx_ >= hi_ones_.size()) {
                i_idx_ = 0;
                // advance to next low-label block
                ++lo_idx_;
                advance_lo_label();
            }
        }

        return !done_;
    }

private:
    // Move to the next hi-label that has at least one '1'
    void advance_hi_label()
    {
        while (hi_idx_ < labels_.size()) {
            int hi_lbl = labels_[hi_idx_++];
            auto& ones = groups_[hi_lbl].first;
            if (ones.empty())
                continue;

            // shuffle ones for this hi-block
            hi_ones_ = ones;
            std::shuffle(hi_ones_.begin(), hi_ones_.end(), rng_);
            i_idx_ = 0;

            // collect all labels ≤ hi_lbl that have zeros
            lo_labels_.clear();
            for (int lbl : labels_) {
                auto& zeros = groups_[lbl].second;
                if (lbl <= hi_lbl && !zeros.empty())
                    lo_labels_.push_back(lbl);
            }

            if (lo_labels_.empty())
                continue;

            std::shuffle(lo_labels_.begin(), lo_labels_.end(), rng_);
            lo_idx_ = 0;
            advance_lo_label();
            return;
        }

        // nothing left
        done_ = true;
    }

    // Move to the next low-label (within current hi-label) that has zeros
    void advance_lo_label()
    {
        while (lo_idx_ < lo_labels_.size()) {
            int lo_lbl = lo_labels_[lo_idx_];
            auto& zeros = groups_[lo_lbl].second;
            if (!zeros.empty()) {
                lo_zeros_ = zeros;
                std::shuffle(lo_zeros_.begin(), lo_zeros_.end(), rng_);
                j_idx_ = 0;
                return;
            }
            ++lo_idx_;
        }
        // exhausted zeros for this hi → advance hi
        advance_hi_label();
    }

    // --- state ---
    std::unordered_map<int, std::pair<UIntVec, UIntVec>> groups_;
    std::vector<int>     labels_;

    // high-label state
    size_t               hi_idx_ = 0;
    UIntVec              hi_ones_;
    size_t               i_idx_ = 0;

    // low-label state
    std::vector<int>     lo_labels_;
    size_t               lo_idx_ = 0;
    UIntVec              lo_zeros_;
    size_t               j_idx_ = 0;

    bool                 done_ = false;
    std::mt19937& rng_;
};

// ───────────────────────────── RNG serialisation ────────────────────────────
void save_rng_state(const std::mt19937& rng, const std::string& name)
{
    std::ostringstream oss; oss << rng; std::string s = oss.str();
    cnpy::npz_save("checkpoint.npz", name, reinterpret_cast<const unsigned char*>(s.data()), { s.size() }, "a");
}
std::mt19937 load_rng_state(const cnpy::npz_t& npz, const std::string& name)
{
    std::mt19937 rng; if (!npz.count(name)) return rng;
    const auto& arr = npz.at(name);
    std::string s(reinterpret_cast<const char*>(arr.data<unsigned char>()), arr.shape[0]);
    std::istringstream iss(s); iss >> rng; return rng;
}

// ───────────────────────────── Checkpoint helpers ───────────────────────────
void save_checkpoint(const VecXb& best, std::size_t evals, const std::mt19937& rng)
{
    cnpy::npz_save("checkpoint.npz", "best_vect", best.data(), { static_cast<unsigned long>(best.size()) }, "w");
    cnpy::npz_save("checkpoint.npz", "running_count", &evals, { 1 }, "a");
    save_rng_state(rng, "prng_state");
}

bool load_checkpoint(VecXb& best, std::size_t& evals, std::mt19937& rng)
{
    try
    {
        cnpy::npz_t npz = cnpy::npz_load("checkpoint.npz");
        const auto& bv = npz.at("best_vect");
        best = Eigen::Map<const VecXb>(bv.data<Byte>(), bv.shape[0]);
        evals = *npz.at("running_count").data<std::size_t>();
        rng = load_rng_state(npz, "prng_state");
        return true;
    }
    catch (...) { return false; }
}

// ───────────────────────────────────── main ─────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " CONFIG.npz MAX_EVALS SEED [LOCAL_CONFIG.npz]\n";
        return 1;
    }

    const std::string configFile = argv[1];
    const std::size_t maxEvals = std::stoull(argv[2]);
    const unsigned long seed = std::stoul(argv[3]);
    const std::string localConfig = (argc >= 5) ? argv[4] : "";

    const cnpy::npz_t cfg = cnpy::npz_load(localConfig.empty() ? configFile : localConfig);

    // Load matrices (stored as NumPy bool_ / uint8)
    MatXb fullsys = Eigen::Map<const MatXb>(cfg.at("fullsystem").data<Byte>(), cfg.at("fullsystem").shape[0], cfg.at("fullsystem").shape[1]);
    MatXb ante = Eigen::Map<const MatXb>(cfg.at("antecedents").data<Byte>(), cfg.at("antecedents").shape[0], cfg.at("antecedents").shape[1]);
    MatXb cons = Eigen::Map<const MatXb>(cfg.at("consequents").data<Byte>(), cfg.at("consequents").shape[0], cfg.at("consequents").shape[1]);

    VecXi initLabels = Eigen::Map<const VecXi>(cfg.at("init_labels").data<int>(), cfg.at("init_labels").shape[0]);
    std::vector<int> initCounts(cfg.at("init_counts").data<int>(), cfg.at("init_counts").data<int>() + cfg.at("init_counts").shape[0]);
    VecXi swapLabels = Eigen::Map<const VecXi>(cfg.at("swap_labels").data<int>(), cfg.at("swap_labels").shape[0]);
    VecXi targetMask = Eigen::Map<const VecXi>(cfg.at("target_mask").data<int>(), cfg.at("target_mask").shape[0]);

    const int maxPossibleScore = targetMask.cwiseMax(0).sum();

    std::mt19937 rng(seed);
    std::size_t runningCount = 0;

    VecXb bestVect = generate_random_binary_vector(initLabels, initCounts, rng);

    if (load_checkpoint(bestVect, runningCount, rng))
        std::cout << "Loaded checkpoint @ " << runningCount << " evaluations\n";
    else
        save_checkpoint(bestVect, runningCount, rng);

    auto [filled0, eval0] = fill_inferable_symbols(bestVect, fullsys, ante, cons);
    runningCount += eval0;
    int bestScore = (filled0.cast<int>() * targetMask).sum();

    // ───────────── main optimisation loop (swaps within label groups) ────────────
    bool improved = true;
    while (improved && runningCount < maxEvals)
    {
        improved = false;
        SwapIterator it(bestVect, swapLabels, rng);
        int i, j;
        while (it.next(i, j) && runningCount < maxEvals)
        {
            std::swap(bestVect(i), bestVect(j));
            auto [filled, ev] = fill_inferable_symbols(bestVect, fullsys, ante, cons);
            runningCount += ev;
            int score = (filled.cast<int>() * targetMask).sum();
            if (score >= bestScore)
            {
                bestScore = score; improved = true; break; // keep swap
            }
            std::swap(bestVect(i), bestVect(j)); // revert
        }

        // Greedy prune when optimal score reached
        if (improved && bestScore == maxPossibleScore)
        {
            bool pruned = true;
            while (pruned && runningCount < maxEvals)
            {
                pruned = false;
                for (int idx = 0; idx < bestVect.size(); ++idx)
                {
                    if (bestVect(idx) == B0 || swapLabels(idx) == 0) continue;
                    bestVect(idx) = B0;
                    auto [f, ev] = fill_inferable_symbols(bestVect, fullsys, ante, cons);
                    runningCount += ev;
                    if ((f.cast<int>() * targetMask).sum() == maxPossibleScore)
                    {
                        pruned = true; break; // keep deletion
                    }
                    bestVect(idx) = B1; // undo
                }
            }
        }

        // Periodic persistence & progress
        save_checkpoint(bestVect, runningCount, rng);
        writeProgress(0.99 * std::min<double>(double(runningCount) / maxEvals, 1.0),
            bestScore, groupCountsFor(swapLabels, bestVect));
    }

    // ───────────────────────── Final output files ────────────────────────────
    save_checkpoint(bestVect, runningCount, rng);

    const int pad = std::to_string(fullsys.rows()).size();
    std::ofstream out("output.txt");
    out << configFile << ' ' << maxEvals << ' ' << seed << "\n";
    out << std::setw(pad) << bestScore << "; ";

    // swap label counts
    {
        auto cnts = groupCountsFor(swapLabels, bestVect);
        for (int c : cnts) out << std::setw(pad) << c << ' ';
        out << "; ";
    }
    // init label counts
    {
        auto cnts = groupCountsFor(initLabels, bestVect);
        for (int c : cnts) out << std::setw(pad) << c << ' ';
        out << "; ";
    }
    out << binaryToHex(bestVect) << "\n";

    std::ofstream("boinc_finish_called") << "finished\n";
    writeProgress(1.0, bestScore, groupCountsFor(swapLabels, bestVect));
    return 0;
}
