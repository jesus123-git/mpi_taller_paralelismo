"""
MPI version 1: broadcast + static file distribution.

Rank 0 reads consulta.txt, broadcasts the query words, lists the file_*.txt
files and splits them into `size` contiguous blocks with (almost) the same
NUMBER of files. Every process (rank 0 included) counts the query words in
its block and the partial results are gathered and merged in rank 0.

The split ignores file sizes, so processes that happen to receive more large
files work longer than the others (load imbalance).

Run:
    mpirun -np <p> python mpi1.py
"""

import os
from collections import Counter

from mpi4py import MPI

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
QUERY_NAME = "consulta.txt"
BASELINE_CSV = "baseline_results.csv"
TOP_N = 10


def load_query(path):
    """Return the set of query words (lowercase), as the baseline does."""
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


def list_corpus_files(dataset_dir):
    """Return the sorted list of file_*.txt paths."""
    names = sorted(
        n for n in os.listdir(dataset_dir)
        if n.startswith("file_") and n.endswith(".txt")
    )
    return [os.path.join(dataset_dir, n) for n in names]


def split_static(files, parts):
    """Contiguous blocks whose number of files differs by at most one."""
    base, extra = divmod(len(files), parts)
    blocks, start = [], 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        blocks.append(files[start:end])
        start = end
    return blocks


def count_file(path, query, freq):
    """Same counting kernel as baseline_secuencial.py. Returns tokens read."""
    tokens = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = [w.lower() for w in line.split()]
            tokens += len(words)
            for w in words:
                if w in query:
                    freq[w] += 1
    return tokens


def matches_baseline(dataset_dir, freq):
    """Compare the global count with baseline_results.csv (None if absent)."""
    path = os.path.join(dataset_dir, BASELINE_CSV)
    if not os.path.isfile(path):
        return None
    expected = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            word, _, count = line.strip().rpartition(",")
            if word:
                expected[word] = int(count)
    got = {w: c for w, c in freq.items() if c > 0}
    return got == expected


def report(title, stats, freq, dataset_dir, elapsed):
    """Print per-process stats, balance metrics, top N and total time."""
    times = [s["time"] for s in stats]
    mean_t = sum(times) / len(times)
    li = max(times) / mean_t if mean_t > 0 else 1.0
    spread = (max(times) - min(times)) / mean_t if mean_t > 0 else 0.0

    print(title)
    print(f"Processes: {len(stats)}\n")
    for s in stats:
        print(
            f"PROC rank={s['rank']} files={s['files']} tokens={s['tokens']} "
            f"local_time={s['time']:.6f}"
        )
    print(f"\nLOCAL_TIME_MAX={max(times):.6f}")
    print(f"LOCAL_TIME_MIN={min(times):.6f}")
    print(f"LOCAL_TIME_MEAN={mean_t:.6f}")
    print(f"LOAD_IMBALANCE={li:.4f}")
    print(f"RELATIVE_SPREAD={spread:.4f}\n")

    print(f"Files processed: {sum(s['files'] for s in stats)}")
    print(f"Total tokens read: {sum(s['tokens'] for s in stats)}")
    print(f"Total occurrences found: {sum(freq.values())}")
    ok = matches_baseline(dataset_dir, freq)
    print(f"MATCHES_BASELINE={'not checked (no baseline CSV)' if ok is None else ok}\n")

    print(f"Top {TOP_N} query words in the corpus:")
    for word, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]:
        print(f"  {word}: {count}")

    print(f"\nEXECUTION_TIME={elapsed:.6f}")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    t_start = MPI.Wtime()

    # Steps 1-4: rank 0 reads the query, lists the files and builds the blocks.
    query = None
    blocks = None
    if rank == 0:
        try:
            query = load_query(os.path.join(DATASET_DIR, QUERY_NAME))
            blocks = split_static(list_corpus_files(DATASET_DIR), size)
        except OSError as e:
            print(f"Error: {e}", flush=True)
            comm.Abort(1)

    query = comm.bcast(query, root=0)
    my_files = comm.scatter(blocks, root=0)

    # Step 5: local counting.
    freq = Counter()
    tokens = 0
    t0 = MPI.Wtime()
    for path in my_files:
        tokens += count_file(path, query, freq)
    local_time = MPI.Wtime() - t0

    # Step 6: gather partial results in rank 0.
    partial = {
        "rank": rank,
        "files": len(my_files),
        "tokens": tokens,
        "time": local_time,
        "freq": freq,
    }
    partials = comm.gather(partial, root=0)

    # Step 7: global result.
    if rank == 0:
        global_freq = Counter()
        for p in partials:
            global_freq.update(p["freq"])
        elapsed = MPI.Wtime() - t_start
        report(
            "MPI version 1: broadcast + static distribution (equal number of files)",
            partials, global_freq, DATASET_DIR, elapsed,
        )


if __name__ == "__main__":
    main()
