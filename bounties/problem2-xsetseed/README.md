(draft statement)

## Bounty Problem 2: xSetSeed reversal from 64 linear constraints

Minecraft\@Home is proud to announce our second programming bounty competition, complete with a total prize pool of **US $5000**. As a competitor, you are tasked with designing an efficient algorithm to solve a very difficult problem related to Minecraft seedfinding. The faster your code, the better. Prizes are awarded on an ongoing basis: whenever a new speed record is set, the record holder receives a prize proportional to how much they improved the record. This document outlines the competition rules.

---

### Overview

Minecraft uses the **Xoroshiro128++** pseudorandom number generator, which has a 128-bit internal state. This bounty problem focuses on how that PRNG is seeded. Minecraft seeds such as world seeds, decoration seeds, and similar values are only 64 bits long, and those 64 bits are converted into the 128-bit Xoroshiro128++ state using the function `xSetSeed()`.

If the full 128-bit output state of `xSetSeed()` is known, then, due to the mathematical form of this function, recovering the original 64-bit seed is easy. However, because there are only `2**64` possible inputs, there are also only `2**64` valid outputs, not `2**128`. This suggests that one should not need the full 128 bits of output to solve for the seed. In principle, 64 bits of information should be enough.

Your challenge is to devise an efficient algorithm which takes a set of 64 linear constraints defined on the 128 output bits of `xSetSeed()` and returns all matching 64-bit inputs which satisfy those constraints. An efficient solution would have dramatic implications for the efficiency of all future xoroshiro reversal algorithms.

---

### Background and Forward Implementation

In the attached file `xsetseed.cpp`, we provide a self-contained reference implementation of the forward algorithm, which computes the 128-bit Xoroshiro128++ state corresponding to a given 64-bit seed.

```
#include <cstdint>

static inline uint64_t splitMix64(uint64_t x)
{
    constexpr uint64_t kMix1 = 0xBF58476D1CE4E5B9ULL;
    constexpr uint64_t kMix2 = 0x94D049BB133111EBULL;

    x = (x ^ (x >> 30)) * kMix1;
    x = (x ^ (x >> 27)) * kMix2;
    x = x ^ (x >> 31);
    return x;
}

static inline void xSetSeed(uint64_t s, uint64_t* lo, uint64_t* hi)
{
    constexpr uint64_t kSilver = 0x6A09E667F3BCC909ULL;
    constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

    s ^= kSilver;
    *lo = splitMix64(s);
    *hi = splitMix64(s + kGolden);
}
```

Given a 64-bit seed `s`, the program returns two 64-bit values, `lo` and `hi`, which together form the 128-bit internal state.

For this competition, the goal is not to recover `s` from the full values of `lo` and `hi`, but rather from a set of 64 linear constraints imposed on those 128 output bits.

Imagine that `lo` and `hi` together form a binary vector of length 128. One may then define a `128 x 64` binary constraint matrix and a length-64 binary target vector. The constraints are of the form

```
target = matrix @ {lo, hi} mod 2
```

where `@` denotes matrix multiplication over GF(2).

In the supplied input format, the target is represented by a 64-bit integer, and the matrix is represented by a list of 128 64-bit integers. To evaluate the matrix multiplication, select from the 128 rows all rows corresponding to positions of 1-bits in `lo` and `hi`. The first row corresponds to the most-significant bit of `lo`, and the 65th row corresponds to the most-significant bit of `hi`. XOR all selected rows together. If the result equals the target, then the seed is a match.

Located in this repo is a text file containing an example constraint matrix and target, as well as a seed `s` which fulfills the constraints. Also included is `constraint-generator.py`, which was used to generate the example matrix.

---

### Core Challenge and Baseline Solution

Right now, there is no known efficient method for solving this reverse problem beyond performing a full 64-bit brute force. For this bounty competition, we are challenging you to develop a fast algorithm to solve it.

Given the values of **target** and **matrix**, your code should output **all matching 64-bit seeds** `s` such that

```
target = matrix @ xSetSeed(s) mod 2
```

On average, we expect one solution for a random instance, but some instances yield none or multiple. Your program must handle all cases.

The current best known strategy is brute-forcing over all possible 64-bit seeds. We have implemented the fastest known version of this approach as a CUDA kernel in `xsetseedreverse.cu`, which serves as the baseline solution for this competition. On an RTX 5090, this code achieves roughly 580 billion seeds per second, implying a runtime of around one year for a full search. It uses a trick which searches the seed space in a non-consecutive order so that the `hi` value of one seed becomes the `lo` value of the next, reducing the number of `splitMix64()` calls required.

Submitted code may be **parallelizable** or **non-parallelizable**:

* *Non-parallelizable* algorithms run on a single machine.
* A *parallelizable* algorithm allows the search space to be split into disjoint subsets processed independently. If the pre- or post-processing steps required by your algorithm are more complex than simply dividing ranges and merging results, include the necessary code and explanations. The baseline solution `xsetseedreverse.cu` is an example of a parallelizable algorithm.

---

### Command-line Interface and Constraints

**Input:**

* a 64-bit target value
* a list of 128 64-bit integers representing the constraint matrix
* (for parallelizable algorithms) search bounds necessary to divide work into discrete tasks

**Output:**

* a list of all matching 64-bit seeds, or an indication that none exist
* (optional) throughput metrics (e.g., seeds per second)

While we expect an average of one solution for any given `(target, matrix)`, some combinations yield none or multiple. Your program must handle all cases.

**Resource limits (per computing node):**

* Built code + data files: <= 5 GB
* VRAM, RAM, and disk: <= 24 GB each

Example test cases can be generated with the forward algorithm in `xsetseed.cpp`. An example constraint file is also included in this repo.

---

### Evaluation

Submissions are judged on **expected runtime** relative to the baseline on a notional BOINC-style grid of 16 desktop PCs, each with an RTX 4090 GPU and an i9-13900K CPU. Because exhaustive `2**64` searches are infeasible to benchmark directly, Minecraft\@Home members will run partial benchmarks and extrapolate to this hypothetical environment.

Each submission receives a **logarithmic score**:

```
Score = 64 + log2(projected runtime of your code / projected runtime of baseline code xsetseedreverse.cu)
```

Lower scores are better: a one-point drop corresponds to cutting the brute-force complexity by one bit. Our baseline code, which performs a 64-bit brute force, is assigned a score of 64. If, for example, your submission yields an 8x speedup, it would be assigned a score of 61. We will aim to be fair and consistent, but organizers have final say. We will attempt to tune parameters (e.g., grid and block dimensions for GPU code) to maximize performance. Minor, inconsistent, or hard-to-measure improvements receive a tying score. Due to the difficulty of assigning scores, submissions that are not in contention for a prize may not receive a precise score. Submissions may be disqualified for rule violations, legal concerns, or impractical deployment.

---

### Incremental Awards and Prize Pool

Prize money is proportional to performance improvement over the current leader. Each "bit" of complexity reduction translates to a certain amount of money won.

```
Award = (previous top score - your score) / previous top score * remaining prize pool
```

Example: if the leader’s score is 62.5 and your score is 61, you improve by 1.5 points and receive `1.5 / 62.5` of the remaining pool. This scheme rewards both incremental optimizations and major breakthroughs.

The initial prize pool is approximately US $5,000 and may grow with additional contributions. This corresponds to slightly over $78 per point of improvement. As incremental prizes are awarded, the remaining funds will decrease accordingly, but the per-point rate will remain approximately the same.

---

### Competition Duration

 The competition will run for **at least one year**, though organizers may extend, pause, or close it as circumstances require. All prize payments are in **Monero (XMR)**, and winners will be asked for a destination wallet address.

---

### Submission Process

Because Minecraft\@Home encourages public engagement, all submissions are public.

1. Upload your code and any required files to a platform such as GitHub.
2. Post a link in our Discord server, explicitly declaring it a submission.
3. The timestamp of that Discord message is your official submission time.
4. All contact will occur through the Discord account that made the submission.

**Submission package must include**

* Source code and any non-source data files (with origins or generation scripts).
* If parallelization involves non-trivial pre/post-processing, all associated code.
* A written document containing:

  * Detailed build and running instructions.
  * A thorough explanation of your algorithm and optimizations.
  * Your own estimate of performance gain versus the baseline or prior entries.

Contestants who wish to make multiple submissions must wait at least one week between submissions. Each entry must be self-contained and declared separately.

---

### Licensing Requirements

To be considered, your submission must be **open-source** under the **MIT License** or a compatible permissive license.

---

### Legal and Tax Considerations

This competition is void where prohibited. Prize payments are subject to all applicable regulations in both the organizers’ and contestants’ jurisdictions. Tax withholding may apply. Winners may be asked for additional documentation before disbursement. Neither Minecraft\@Home nor this competition is affiliated with or endorsed by Microsoft or Mojang Studios.

---

### Privacy Notice

Certain information (submissions, scores, prize amounts, and publicly visible profile details) will be public. Personally identifiable information collected for legal or tax reasons will be handled privately via Discord direct messages and private channels.
