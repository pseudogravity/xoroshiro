(unofficial work-in-progress text)

## Bounty Problem 1: Decoration seed to world seed with known coordinates

Minecraft@Home is proud to announce our first programming bounty competition, complete with a total prize pool of **US $5000**. As a competitor, you are tasked with designing an efficient algorithm to solve a very difficult problem related to Minecraft seedfinding. The faster your code, the better. Prizes are awarded on a ongoing basis: whenever a new speed record is set, the recordholder recieves a prize proportional to how much they improved the record. This document outlines the competition rules.

---

### Overview

In Minecraft, each chunk has a decoration seed, a value derived from the world seed and the chunk's coordinates. This seed determines features like trees, ores, and portions of structures. While computing the decoration seed from a known world seed is straightforward, the reverse process is computationally difficult. Your challenge is to find a fast and efficient method for finding the world seeds which produce a target decoration seed at a specified location.

---

### Background and Forward Implementation

In the attached file `worldtodecoration.cpp`, we provide a self‑contained reference implementation of the forward algorithm, which calculates the decoration seed for any combination of world seed and coordinate pair.

```
Xoroshiro128PlusPlus prng
prng.xSetSeed(worldSeed)

a = prng.nextLongJ() | 1
b = prng.nextLongJ() | 1

decorationSeed = (a * blockX + b * blockZ) ^ worldSeed
```

* `xSetSeed()` seeds a fresh instance of **Xoroshiro128++** with the 64‑bit world seed.
* `nextLongJ()` uses the top 32 bits from two consecutive PRNG outputs, so four PRNG calls are made in total to obtain `a` and `b`.
* `blockX` and `blockZ` refer to the block coordinates at the chunk corner and are always multiples of 16:

```
blockX = 16 * chunkX
blockZ = 16 * chunkZ
```

Given `worldSeed`, `chunkX`, and `chunkZ`, the program returns `a`, `b`, and the resulting decoration seed.

---

### Core Challenge and Baseline Solution

Right now, there is no efficient method to determine the world seed from the decoration seed. For this bounty competition, we are challenging you to develop a fast algorithm to solve this reverse problem. Given the values of **decoration seed**, **chunk x**, and **chunk z**, your code should output **all matching 64‑bit world seeds**.

The current best known strategy is brute‑forcing over possible world seeds, leveraging the fact that the least‑significant four bits of the world seed must equal those of the decoration seed (because block coordinates are multiples of 16). This requires up to 2**60 checks. We have implemented this algorithm as a CUDA kernel in `decorationreverse.cu`, which serves as the baseline solution for this competition.

Submitted code may be **parallelizable** or **non‑parallelizable**:

* *Non‑parallelizable* algorithms run on a single machine.
* A *parallelizable* algorithm allows the search space to be split into disjoint subsets processed independently. If the pre‑ or post‑processing steps required by your algorithm are more complex than simply dividing ranges and merging results, include the necessary code and explanations. The baseline solution `decorationreverse.cu` is an example of a parallelizable algorithm.

---

### Command‑line Interface and Constraints

**Input:**

* a 64‑bit decoration seed
* `chunkX` and `chunkZ`, each a signed integer with |value| <= 1 875 000
* (for parallelizable algorithms) search bounds necessary to divide work into discrete tasks

**Output:**

* a list of all matching world seeds, or an indication that none exist
* (optional) throughput metrics (e.g., seeds per second)

While we expect an average of one solution for any given `(decoration seed, chunkX, chunkZ)`, some combinations yield none or multiple. Your program must handle all cases.

**Resource limits (per computing node):**

* Built code + data files: <= 5 GB
* VRAM, RAM, and disk: <= 24 GB each

Example test cases can be generated with the forward algorithm in `worldtodecoration.cpp`.

---

### Evaluation

Submissions are judged on **expected runtime** relative to the baseline on a notional BOINC‑style grid of 256 desktop PCs, each with an RTX 4090 GPU and an i9‑13900K CPU. Because exhaustive 2**60 searches are infeasible to benchmark directly, Minecraft\@Home members will run partial benchmarks and extrapolate to this hypothetical environment.

Each submission receives a **logarithmic score**:

```
Score = 60 + log2(projected runtime of your code / projected runtime of baseline code decorationreverse.cu)
```

Lower scores are better: a one‑point drop corresponds to cutting the brute‑force complexity by one bit. Our baseline code, which performs a 60-bit brute force, is assigned a score of 60. If, for example, your submission yields an 8x speedup, it would be assigned a score of 57. We will aim to be fair and consistent, but organizers have final say. We will attempt to tune parameters (e.g., grid and block dimensions for GPU code) to maximize performance. Minor, inconsistent, or hard‑to‑measure improvements receive a tying score. Due to the difficulty of assigning scores, submissions that are not in contention for a prize may not receive a precise score. Submissions may be disqualified for rule violations, legal concerns, or impractical deployment.

---

### Incremental Awards and Prize Pool

Prize money is proportional to performance improvement over the current leader. Each "bit" of complexity reduction translates to a certain amount of money won.

```
Award = (previous top score - your score) / previous top score * remaining prize pool
```

Example: if the leader’s score is 57.5 and your score is 56, you improve by 1.5 points and receive `1.5 / 57.5` of the remaining pool. This scheme rewards both incremental optimizations and major breakthroughs.

The initial prize pool is approximately US \$5 000 and may grow with additional contributions. This corresponds to approximately $83 per point of improvement. As incremental prizes are awarded, the remaining funds will decrease accordingly, but the $83/point rate will remain the same.

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

* Source code and any non‑source data files (with origins or generation scripts).
* If parallelization involves non‑trivial pre/post‑processing, all associated code.
* A written document containing:

  * Detailed build and running instructions.
  * A thorough explanation of your algorithm and optimizations.
  * Your own estimate of performance gain versus the baseline or prior entries.

Contestants who wish to make multiple submissions must wait at least one week between submissions. Each entry must be self‑contained and declared separately.

---

### Licensing Requirements

To be considered, your submission must be **open‑source** under the **MIT License** or a compatible permissive license.

---

### Legal and Tax Considerations

This competition is void where prohibited. Prize payments are subject to all applicable regulations in both the organizers’ and contestants’ jurisdictions. Tax withholding may apply. Winners may be asked for additional documentation before disbursement. Neither Minecraft\@Home nor this competition is affiliated with or endorsed by Microsoft or Mojang Studios.

---

### Privacy Notice

Certain information (submissions, scores, prize amounts, and publicly visible profile details) will be public. Personally identifiable information collected for legal or tax reasons will be handled privately via Discord direct messages and private channels.
