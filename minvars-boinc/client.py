import numpy as np
import pickle
import sys
import random
from typing import Sequence, Iterator, Tuple

def gf2_dependent_subset(sel, M):
    """
    Single-pass GF(2) rank + dependency detection with micro-optimizations:
      * Use argmax to find pivots.
      * Minimize per-loop overhead.
      * Process selected rows first, then candidate rows.

    Parameters
    ----------
    sel : boolean array of shape (n,)
        Indicator for which rows are selected.
    M   : boolean array of shape (n, c)
        GF(2) matrix.

    Returns
    -------
    rank_val : int
        Rank of the submatrix formed by selected rows.
    dep : boolean array of shape (n,)
        For each row k not selected (sel[k]==False), dep[k] == True if that row
        is in the span of the selected rows.
    """
    # Ensure boolean arrays
    sel = np.asarray(sel, dtype=bool)
    M   = np.asarray(M,   dtype=bool)
    n   = M.shape[0]

    # Indices of selected and candidate rows
    sel_idx  = np.flatnonzero(sel)
    cand_idx = np.flatnonzero(~sel)
    num_sel  = sel_idx.size
    num_cand = cand_idx.size

    # If no rows selected, rank is 0 and no dependencies
    if num_sel == 0:
        return 0, np.zeros(n, dtype=bool)

    # Build top block: selected rows of M
    A = M[sel_idx, :]
    top_count = A.shape[0]

    # Build bottom block: candidate rows of M (empty if none)
    C = M[cand_idx, :] if num_cand > 0 else np.zeros((0, M.shape[1]), dtype=bool)
    bottom_count = C.shape[0]

    # Combine blocks into one matrix for elimination
    B = np.vstack((A, C))
    row_count, col_count = B.shape

    # Perform Gaussian elimination over GF(2)
    pivot_row = 0
    for col in range(col_count):
        if pivot_row >= top_count:
            break

        subcol = B[pivot_row:top_count, col]
        pivot_offset = subcol.argmax()
        if not subcol[pivot_offset]:
            continue

        pivot_idx = pivot_row + pivot_offset
        if pivot_idx != pivot_row:
            B[[pivot_row, pivot_idx]] = B[[pivot_idx, pivot_row]]

        # Eliminate 1s below the pivot in this column
        if pivot_row < row_count - 1:
            below = B[pivot_row + 1:, col]
            rows_to_xor = np.flatnonzero(below) + (pivot_row + 1)
            if rows_to_xor.size > 0:
                B[rows_to_xor] ^= B[pivot_row]

        pivot_row += 1

    rank_val = pivot_row

    # Determine which candidate rows are in the span (zero rows after elimination)
    dep = np.zeros(n, dtype=bool)
    if bottom_count > 0 and rank_val > 0:
        candidate_block = B[top_count:]
        dep_mask = ~candidate_block.any(axis=1)
        dep[cand_idx[dep_mask]] = True

    return rank_val, dep

def apply_inference_rules(V: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Given:
        V: 1D binary array of length W (known vars marked 1)
        X: 2D binary array of shape (R, W) (rule antecedents)
        Y: 2D binary array of shape (R, W) (rule consequents)

    Returns:
        A binary vector of length W containing newly inferred vars:
          - For each rule i, if all X[i,j]==1 ⇒ V[j]==1, the rule is 'satisfied'.
          - OR together all Y[i] for satisfied rules.
          - Zero out any positions already 1 in V.
    """
    # 1) Check antecedents: rule i is satisfied if X[i] <= V elementwise
    #    i.e. wherever X[i]==1, V must also be 1.
    #    (X & ~V) will be 1 where X demands a var that V doesn't know; so we want no such bits.
    violated = (X & (~V.astype(bool))).any(axis=1)
    satisfied = ~violated

    # 2) Build binary vector of satisfied rules (optional to return if needed)
    #    sat_vec = satisfied.astype(int)

    # 3) OR together the consequents Y for satisfied rules
    if satisfied.any():
        or_result = np.any(Y[satisfied], axis=0)
    else:
        or_result = np.zeros_like(V, dtype=bool)

    # 4) Zero out positions already known in V
    new_inferences = or_result & (~V.astype(bool))

    # 5) Return as int array
    return new_inferences.astype(int)

def fill_inferable_symbols(curvect,fullsystem,antecedents,consequents):

    eval_count = 0

    curvect = curvect.copy()
    
    pastlinear = 0
    curlinear = curvect.sum()

    pastnonlinear = 0
    curnonlinear = curvect.sum()

    while pastlinear != curlinear or pastnonlinear != curnonlinear:
        if pastlinear != curlinear:
            rank,infervect = gf2_dependent_subset(curvect,fullsystem)
            eval_count += 1
            curvect = np.maximum(curvect,infervect)

            pastlinear = curlinear
            curlinear = curvect.sum()
            curnonlinear = curvect.sum()
            pastlinear = curlinear # since linear never needs to be run twice in a row
            
        if pastnonlinear != curnonlinear:
            infervect = apply_inference_rules(curvect,antecedents,consequents)
            eval_count += 1
            curvect = np.maximum(curvect,infervect)

            pastnonlinear = curnonlinear
            curlinear = curvect.sum()
            curnonlinear = curvect.sum()
            
    return curvect, eval_count

def generate_random_binary_vector(
    labels: np.ndarray,
    counts: Sequence[int],
    prng: random.Random
) -> np.ndarray:
    """
    Given:
      - labels: 1D array of integer labels of length N, each in [0, len(counts)-1]
      - counts: sequence of length L = number of possible labels,
                where counts[l] is the desired number of 1’s for label l
      - prng: a seeded instance of random.Random

    Returns a 1D numpy array `bits` of length N, containing 0’s and 1’s,
    such that for each label l, exactly counts[l] entries of `bits` are 1
    at the positions where labels == l.

    Input validation:
      * labels must be 1D and of integer dtype
      * counts must have length ≥ max(labels)+1
      * every counts[l] must be 0 ≤ counts[l] ≤ number of positions with labels==l
      * for any l ∈ [0..len(counts)-1] not present in labels, counts[l] must be 0
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"`labels` must be 1D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("`labels` must be of integer dtype")

    N = labels.shape[0]
    counts = list(counts)
    Lc = len(counts)

    if N == 0:
        # empty labels → ensure all counts zero
        if any(c != 0 for c in counts):
            raise ValueError("labels is empty but some counts > 0")
        return np.zeros(0, dtype=int)

    max_lbl = int(labels.max())
    min_lbl = int(labels.min())
    if min_lbl < 0:
        raise ValueError(f"found negative label {min_lbl}; labels must be in [0..{Lc-1}]")
    if max_lbl >= Lc:
        raise ValueError(
            f"labels contain value {max_lbl}, but counts length is only {Lc}"
        )

    # Precompute indices for each label
    label_to_indices = {l: np.nonzero(labels == l)[0] for l in range(Lc)}

    # Validate counts
    for l, cnt in enumerate(counts):
        # if not isinstance(cnt, int):
        #     raise TypeError(f"counts[{l}] == {cnt} is not an integer")
        if cnt < 0:
            raise ValueError(f"counts[{l}] == {cnt}: count must be ≥ 0")
        idxs = label_to_indices[l]
        if len(idxs) < cnt:
            raise ValueError(
                f"counts[{l}] == {cnt} but only {len(idxs)} entries have label {l}"
            )

    # Build the output vector
    bits = np.zeros(N, dtype=int)

    # For each label, randomly pick exactly counts[l] positions to set to 1
    for l, cnt in enumerate(counts):
        if cnt == 0:
            continue
        idxs = label_to_indices[l].tolist()
        chosen = prng.sample(idxs, cnt)
        bits[chosen] = 1

    return bits

class SwapIterator:
    """
    Iterator over all index-pairs (i, j) such that:
      - bits[i] == 1, bits[j] == 0
      - labels[i] == labels[j] > 0
    Produces each such pair exactly once, in a randomized order determined
    by the provided PRNG. Uses only O(N) extra memory.
    """
    def __init__(self,
                 bits: np.ndarray,
                 labels: np.ndarray,
                 prng: random.Random):
        if bits.shape != labels.shape:
            raise ValueError("bits and labels must have the same shape")
        if bits.ndim != 1:
            raise ValueError("only 1D arrays are supported")
        self.prng = prng

        # Build groups of indices by label
        groups = {}
        for idx, lbl in enumerate(labels):
            if lbl <= 0:
                continue
            bit = bits[idx]
            if bit not in (0, 1):
                raise ValueError(f"bits must be 0 or 1; found {bit} at position {idx}")
            if lbl not in groups:
                groups[lbl] = {"ones": [], "zeros": []}
            if bit == 1:
                groups[lbl]["ones"].append(idx)
            else:  # bit == 0
                groups[lbl]["zeros"].append(idx)

        # Keep only labels that have at least one 1 and one 0
        self.groups = {
            lbl: (data["ones"], data["zeros"])
            for lbl, data in groups.items()
            if data["ones"] and data["zeros"]
        }

        # Prepare label order
        self._labels = list(self.groups.keys())
        self.prng.shuffle(self._labels)
        self._label_iter = iter(self._labels)

        # Will be set by _prepare_next_label()
        self._done = False
        self._ones = []
        self._zeros = []
        self._i = 0
        self._j = 0
        self._prepare_next_label()

    def _prepare_next_label(self):
        """Advance to the next label group and shuffle its indices."""
        try:
            lbl = next(self._label_iter)
        except StopIteration:
            self._done = True
            return
        ones, zeros = self.groups[lbl]
        # shuffle in-place to randomize block and inner order
        self.prng.shuffle(ones)
        self.prng.shuffle(zeros)
        self._ones, self._zeros = ones, zeros
        self._i = 0
        self._j = 0

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        return self

    def __next__(self) -> Tuple[int, int]:
        if self._done:
            raise StopIteration

        i = self._ones[self._i]
        j = self._zeros[self._j]

        # advance inner index
        self._j += 1
        if self._j >= len(self._zeros):
            self._j = 0
            self._i += 1
            if self._i >= len(self._ones):
                # finished this label block → move to next
                self._prepare_next_label()

        return (i, j)

def save_checkpoint(best_vect,running_count,prng):
    np.savez_compressed(
        'checkpoint.npz',
        best_vect = best_vect,
        running_count = running_count,
        prng_state = pickle.dumps(prng.getstate())
    )

def write_progress_bar_with_stdout(value,data1,data2):
    print(f"{value:.5f} {data1} {data2}")
    
    with open('boinc_frac', 'w', newline='\n') as file:
        file.write(f"{value:.5f}")
        file.flush()

def binary_to_hex_chunked(binary_str):
    # Pad the binary string from the left so its length is a multiple of 4
    padded = binary_str.zfill((len(binary_str) + 3) // 4 * 4)
    
    hex_str = ''
    for i in range(0, len(padded), 4):
        chunk = padded[i:i+4]
        hex_digit = hex(int(chunk, 2))[2:]  # Convert to hex and remove '0x'
        hex_str += hex_digit.lower()        # Use .upper() if uppercase is preferred
    
    return hex_str

if __name__ == '__main__':
    print(sys.argv)
    config_file = sys.argv[1] # 'config-001.npz'
    max_evals = int(sys.argv[2]) # 500000
    rng_seed = int(sys.argv[3]) # 123
    local_config_file = None
    try:
        local_config_file = sys.argv[4] # input.npz
    except:
        pass

    if local_config_file:
        config = np.load(local_config_file)
    else:
        config = np.load(config_file)

    fullsystem = config['fullsystem'].astype(bool)
    antecedents = config['antecedents'].astype(bool)
    consequents = config['consequents'].astype(bool)
    init_labels = config['init_labels'].astype(int)
    init_counts = config['init_counts'].astype(int)
    swap_labels = config['swap_labels'].astype(int)
    target_mask = config['target_mask'].astype(int)

    max_possible_score = np.maximum(target_mask, 0).sum()

    running_count = 0
    prng = random.Random(rng_seed)
    
    best_vect = generate_random_binary_vector(
        labels = init_labels,
        counts = init_counts,
        prng = prng,
    )

    try:
        checkpoint = np.load('checkpoint.npz')
        best_vect = checkpoint['best_vect']
        running_count = checkpoint['running_count']
        prng_state = pickle.loads(checkpoint['prng_state'].item())
        prng.setstate(prng_state)
        print(f'loaded from checkpoint {running_count}')
    except:
        save_checkpoint(best_vect,running_count,prng)

    filled, _ = fill_inferable_symbols(best_vect,fullsystem,antecedents,consequents)
    best_score = (filled * target_mask).sum()

    swap_found = True
    while swap_found:
        swap_found = False
        for swapi,swapj in SwapIterator(best_vect, swap_labels, prng):
            if running_count > max_evals:
                break
                
            #print(swapi,swapj)
            best_vect[swapi],best_vect[swapj] = best_vect[swapj],best_vect[swapi]
        
            test_filled, eval_count = fill_inferable_symbols(best_vect,fullsystem,antecedents,consequents)
            running_count += eval_count
            
            test_score = (test_filled * target_mask).sum()
        
            if test_score >= best_score:
                swap_found = True
                best_score = test_score
                break
            
            best_vect[swapi],best_vect[swapj] = best_vect[swapj],best_vect[swapi]
        
        if not swap_found:
            break
        
        if best_score == max_possible_score:
            while True:
                scores = np.zeros(len(best_vect))
                for i in range(len(scores)):
                    if best_vect[i] == 0 or swap_labels[i] == 0:
                        continue
                    best_vect[i] = 0
                    
                    test_filled, eval_count = fill_inferable_symbols(best_vect,fullsystem,antecedents,consequents)
                    running_count += eval_count
                    
                    test_score = (test_filled * target_mask).sum()
                    scores[i] = test_score
                    best_vect[i] = 1
                    
                if max(scores) != max_possible_score or max(scores) == 0:
                    break
            
                deleteable = np.flatnonzero(scores == max_possible_score)
                best_vect[prng.choice(deleteable)] = 0
                write_progress_bar_with_stdout(.99*min(running_count/max_evals,1.01), best_score, np.bincount(swap_labels,best_vect).astype(int))
        
        save_checkpoint(best_vect,running_count,prng)
        write_progress_bar_with_stdout(.99*min(running_count/max_evals,1.01), best_score, np.bincount(swap_labels,best_vect).astype(int))

    save_checkpoint(best_vect,running_count,prng)
    
    padding = len(str(len(fullsystem)))
    with open('output.txt', 'w', newline='\n') as file:
        file.write(f'{config_file} {max_evals} {rng_seed}\n')
        file.write(f'{int(best_score):>{padding}}; ')
        file.write(' '.join(f'{x:>{padding}}' for x in np.bincount(swap_labels,best_vect).astype(int)))
        file.write('; ')
        file.write(' '.join(f'{x:>{padding}}' for x in np.bincount(init_labels,best_vect).astype(int)))
        file.write('; ')
        file.write(binary_to_hex_chunked(''.join(best_vect.astype(int).astype(str))))
        file.write('\n')
    
    with open('boinc_finish_called', 'w', newline='\n') as file:
        file.write('finished\n')
        file.flush()
    
    write_progress_bar_with_stdout(1.0, best_score, np.bincount(swap_labels,best_vect).astype(int))