"""
MPI version 2: broadcast + dynamic (on-demand) file distribution.

Version 1 gives every process the same NUMBER of files, but file sizes are
very different (2k to 100k words), so the work per process is uneven.

Here nothing is assigned in advance. Rank 0 broadcasts the query and the
list of files. Each worker asks rank 0 for a chunk of
files and, as soon as it receives one, asks for the next, so it always has
work waiting when it finishes. A process that is faster (or got lighter
files) simply takes more chunks. Chunk size is "guided": each request
receives about remaining_files / (8 * p), so chunks are large at the start
(few messages) and shrink to a single file at the end (all processes finish
almost together).

Rank 0 also works: it counts files one by one from the end of the list and
answers pending requests between files.

No file sizes are read: asking the file system for 3000 sizes turned out to
be very slow on a Docker bind mount, and the on-demand scheme does not need
them.

Run:
    mpirun -np <p> python mpi2.py
"""

import os
from collections import Counter

from mpi4py import MPI

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
QUERY_NAME = "consulta.txt"
BASELINE_CSV = "baseline_results.csv"
TOP_N = 10

# A request receives about remaining_files / (GUIDED_FACTOR * p) files.
GUIDED_FACTOR = 8

# While counting, rank 0 checks for worker requests at most once per this many
# seconds, so that coordinating does not eat into its own work.
POLL_INTERVAL = 0.002

TAG_REQUEST = 1
TAG_CHUNK = 2
TAG_RESULT = 3


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


def coordinator(comm, size, query, files):
    """
    Rank 0: hand out chunks on demand and also count files itself.
    Workers take chunks from the left of the list; rank 0 takes single files
    from the right and serves pending requests between files.
    """
    status = MPI.Status()
    left, right = 0, len(files) - 1          # files[left..right] are pending
    active_workers = size - 1

    freq = Counter()
    n_files = 0
    tokens = 0
    local_time = 0.0

    def answer(worker_rank):
        nonlocal left, active_workers
        remaining = right - left + 1
        if remaining <= 0:
            comm.send(None, dest=worker_rank, tag=TAG_CHUNK)  # no more work
            active_workers -= 1
            return
        n = max(1, remaining // (GUIDED_FACTOR * size))
        # Only the position of the chunk travels: every process already has
        # the file list, and a tiny message never blocks rank 0.
        comm.send((left, n), dest=worker_rank, tag=TAG_CHUNK)
        left += n

    def serve_pending():
        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQUEST, status=status):
            worker_rank = status.Get_source()
            comm.recv(source=worker_rank, tag=TAG_REQUEST)
            answer(worker_rank)

    last_poll = -1.0
    while True:
        now = MPI.Wtime()
        if now - last_poll >= POLL_INTERVAL:
            serve_pending()
            last_poll = now
        if left > right:
            break
        path = files[right]
        right -= 1
        t0 = MPI.Wtime()
        tokens += count_file(path, query, freq)
        local_time += MPI.Wtime() - t0
        n_files += 1

    # Nothing left: tell every worker that is still active to stop.
    while active_workers > 0:
        comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQUEST, status=status)
        answer(status.Get_source())

    partials = [{
        "rank": 0, "files": n_files, "tokens": tokens,
        "time": local_time, "freq": freq,
    }]
    for worker_rank in range(1, size):
        partials.append(comm.recv(source=worker_rank, tag=TAG_RESULT))
    return partials


def worker(comm, rank, query, files):
    """
    Ranks > 0: count chunks until rank 0 says stop. The request for the next
    chunk is sent before starting the current one, so the reply is normally
    already waiting when the current chunk is finished.
    """
    freq = Counter()
    n_files = 0
    tokens = 0
    local_time = 0.0

    comm.send(None, dest=0, tag=TAG_REQUEST)
    while True:
        chunk = comm.recv(source=0, tag=TAG_CHUNK)
        if chunk is None:
            break
        comm.send(None, dest=0, tag=TAG_REQUEST)  # ask for the next one now
        start, n = chunk
        t0 = MPI.Wtime()
        for path in files[start:start + n]:
            tokens += count_file(path, query, freq)
        local_time += MPI.Wtime() - t0
        n_files += n

    comm.send(
        {"rank": rank, "files": n_files, "tokens": tokens,
         "time": local_time, "freq": freq},
        dest=0, tag=TAG_RESULT,
    )


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    t_start = MPI.Wtime()

    query = None
    files = None
    if rank == 0:
        try:
            query = load_query(os.path.join(DATASET_DIR, QUERY_NAME))
            files = list_corpus_files(DATASET_DIR)
        except OSError as e:
            print(f"Error: {e}", flush=True)
            comm.Abort(1)

    query = comm.bcast(query, root=0)
    files = comm.bcast(files, root=0)

    if rank != 0:
        worker(comm, rank, query, files)
        return

    partials = coordinator(comm, size, query, files)

    global_freq = Counter()
    for p in partials:
        global_freq.update(p["freq"])
    elapsed = MPI.Wtime() - t_start
    report(
        "MPI version 2: broadcast + dynamic distribution (guided chunks on demand)",
        partials, global_freq, DATASET_DIR, elapsed,
    )


if __name__ == "__main__":
    main()
