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
 *
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

// ────────────────────── GF(2) rank + dependency detection ───────────────────
/*
 * Given a boolean selection vector `sel` and matrix M (rows×cols) over GF(2),
 * returns rank(selRows) and a vector marking candidate rows that are linear
 * combinations of the selected rows (in the span).
 */
std::pair<int, VecXb> gf2_dependent_subset(const VecXb& sel, const MatXb& M)
{
    const std::size_t n = static_cast<std::size_t>(M.rows());
    UIntVec selIdx, candIdx; selIdx.reserve(n); candIdx.reserve(n);
    for (std::size_t i = 0; i < n; ++i) (sel(i) ? selIdx : candIdx).push_back(i);

    if (selIdx.empty()) return { 0, VecXb::Zero(n) };

    const std::size_t top = selIdx.size();
    const std::size_t bottom = candIdx.size();
    const std::size_t cols = static_cast<std::size_t>(M.cols());

    MatXb B(top + bottom, cols);
    for (std::size_t r = 0; r < top; ++r) B.row(r) = M.row(selIdx[r]);
    for (std::size_t r = 0; r < bottom; ++r) B.row(top + r) = M.row(candIdx[r]);

    int pivotRow = 0;
    for (std::size_t c = 0; c < cols && pivotRow < static_cast<int>(top); ++c)
    {
        // Find a pivot (1) at or below pivotRow within the *selected* block
        int pivot = -1;
        for (int r = pivotRow; r < static_cast<int>(top); ++r)
            if (B(r, c) == B1) { pivot = r; break; }
        if (pivot == -1) continue;              // no pivot in this column

        if (pivot != pivotRow) B.row(pivot).swap(B.row(pivotRow));

        // XOR‑eliminate 1s below current pivot across full matrix
        for (int r = pivotRow + 1; r < B.rows(); ++r)
            if (B(r, c) == B1)
                // XOR (GF(2)): a ← a ⊕ pivot
                B.row(r) = (B.row(r) != B.row(pivotRow)).template cast<Byte>();

        ++pivotRow;
    }
    const int rankVal = pivotRow;

    VecXb dep = VecXb::Zero(n);
    if (bottom && rankVal)
    {
        for (std::size_t r = 0; r < bottom; ++r)
        {
            // Row is dependent if all zeros after elimination
            if (!(B.row(top + r) == B1).any()) dep(candIdx[r]) = B1;
        }
    }
    return { rankVal, dep };
}

// ─────────────────────── Non‑linear inference rules ─────────────────────────
/*
 * Implements the apply_inference_rules(V, X, Y) operation from the Python
 * version.
 */
VecXb apply_inference_rules(const VecXb& V, const MatXb& X, const MatXb& Y)
{
    const int R = static_cast<int>(X.rows());
    const int W = static_cast<int>(X.cols());

    // Prepare int versions for bitwise logic ease
    const VecXi Vint = V.cast<int>();
    VecXb newBits = VecXb::Zero(W);

    for (int r = 0; r < R; ++r)
    {
        // Rule r is satisfied iff (X[r] & ~V) has no 1s
        bool violated = false;
        for (int c = 0; c < W; ++c)
            if (X(r, c) == B1 && V(c) == B0) { violated = true; break; }
        if (violated) continue;

        // OR all consequents Y[r] into accumulator
        for (int c = 0; c < W; ++c)
            if (Y(r, c) == B1) newBits(c) = B1;
    }

    // Remove already‑known vars (bitwise‑AND with ¬V)
    VecXb mask = VecXb::Constant(W, B1) - V;   // 1 where V==0
    newBits = newBits * mask;                  // 0‑1 arithmetic ⇒ logical AND
    return newBits;
}

// ────────────────── Expand vector via linear + non‑linear rules ─────────────
std::pair<VecXb, std::size_t> fill_inferable_symbols(const VecXb& start,
    const MatXb& full,
    const MatXb& ante,
    const MatXb& cons)
{
    VecXb cur = start;
    std::size_t evalCount = 0;
    std::size_t prevLinear = 0, prevNonLinear = 0;
    std::size_t curLinear = countOnes(cur);
    std::size_t curNonLin = curLinear;

    while (prevLinear != curLinear || prevNonLinear != curNonLin)
    {
        if (prevLinear != curLinear)
        {
            auto [rank, dep] = gf2_dependent_subset(cur, full);
            ++evalCount;
            cur = cur.max(dep);  // logical OR (values are 0/1)
            prevLinear = curLinear;
            curLinear = countOnes(cur);
            curNonLin = curLinear;
        }
        if (prevNonLinear != curNonLin)
        {
            VecXb infer = apply_inference_rules(cur, ante, cons);
            ++evalCount;
            cur = cur.max(infer);
            prevNonLinear = curNonLin;
            curLinear = countOnes(cur);
            curNonLin = curLinear;
        }
    }
    return { cur, evalCount };
}

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
    SwapIterator(const VecXb& bits, const VecXi& labels, std::mt19937& rng)
        : rng_(rng)
    {
        if (bits.size() != labels.size())
            throw std::runtime_error(
                std::string("bits/labels size mismatch: bits.size()=") + std::to_string(bits.size())
                + ", labels.size()=" + std::to_string(labels.size()));

        for (int i = 0; i < bits.size(); ++i)
        {
            int lbl = labels(i);
            if (lbl <= 0) continue;
            auto& pair = groups_[lbl];
            (bits(i) == B1 ? pair.first : pair.second).push_back(i);
        }

        for (auto it = groups_.begin(); it != groups_.end(); )
            if (it->second.first.empty() || it->second.second.empty())
                it = groups_.erase(it);
            else ++it;

        for (const auto& kv : groups_) labelOrder_.push_back(kv.first);
        std::shuffle(labelOrder_.begin(), labelOrder_.end(), rng_);
        labelIter_ = labelOrder_.begin();
        advanceLabel();
    }

    bool next(int& i, int& j)
    {
        if (done_) return false;
        i = ones_[iIdx_]; j = zeros_[jIdx_];

        if (++jIdx_ >= zeros_.size()) { jIdx_ = 0; if (++iIdx_ >= ones_.size()) advanceLabel(); }
        return true;
    }

private:
    void advanceLabel()
    {
        if (labelIter_ == labelOrder_.end()) { done_ = true; return; }
        int lbl = *labelIter_++;
        auto& p = groups_[lbl];
        ones_ = p.first; zeros_ = p.second;
        std::shuffle(ones_.begin(), ones_.end(), rng_);
        std::shuffle(zeros_.begin(), zeros_.end(), rng_);
        iIdx_ = jIdx_ = 0;
    }

    std::unordered_map<int, std::pair<UIntVec, UIntVec>> groups_;
    std::vector<int> labelOrder_; std::vector<int>::iterator labelIter_;
    UIntVec ones_, zeros_; std::size_t iIdx_ = 0, jIdx_ = 0; bool done_ = false;
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
void usage(){
    printf("Usage:\n");
    printf("--config [-c]: Config logical name for results reporting\n");
    printf("--file [-f]: Config file actual path\n");
    printf("--evals [-e]: Max evaluation count (integer)\n");
    printf("--seed [-s]: RNG Seed\n");
    exit(1);
}
// ───────────────────────────────────── main ─────────────────────────────────
int main(int argc, char* argv[])
{
    std::string configFile = "";
    std::size_t maxEvals = 0;
    unsigned long seed = 0;
    std::string localConfig = "";
    for(int i = 1; i < argc; i+=2){
if(strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "-s") == 0){
            seed = std::stoul(argv[i+1]);
        } else if(strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0){ 
            configFile = argv[i+1];
        } else if(strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-f") == 0){   
            localConfig = argv[i+1];
        } else if(strcmp(argv[i], "--evals") == 0 || strcmp(argv[i], "-e") == 0){ 
            maxEvals = std::stoull(argv[i+1]);
        } else{
            printf("%s unrecognized.\n", argv[i]);
            usage();
        }
    }

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
