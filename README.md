# Lab 2: Parallel Word Counting in a Text Corpus with MPI

Jesús Márquez
Parallel Computing, Universidad del Norte

## Problem Description

The generator script creates a folder called `dataset/`. Inside there is a file `consulta.txt` with 200 words (one per line) and 3000 text files named `file_0001.txt` to `file_3000.txt`. All the files together have about 45 million words.

The goal is to count how many times each word of `consulta.txt` appears in the 3000 files and print the 10 most frequent words.

The files do not have the same size. Looking at `generator.py`, 60% of them have between 2,000 and 5,000 words, 30% between 10,000 and 20,000, and 10% between 50,000 and 100,000. So one big file can be the same work as 20 or more small ones. This is important for the lab because it is the reason the load ends up unbalanced.

We worked in three steps: first we ran the sequential program that was given, then we wrote an MPI version with a static distribution of the files (`mpi1.py`), and finally a second MPI version that tries to balance the load better (`mpi2.py`).

Files in this repository:

| File | What it is |
|---|---|
| `generator.py` | Creates the dataset (given, not modified) |
| `baseline_secuencial.py` | Sequential solution (given, not modified) |
| `run_all.sh` | Script that runs everything (given, not modified) |
| `mpi1.py` | MPI version 1, static distribution |
| `mpi2.py` | MPI version 2, dynamic distribution |

## Environment and Execution Instructions

Everything was run with Docker using the image of the course, `augustosalazar/slim-mpi:2`, which already has Python, Open MPI and mpi4py. Nothing else has to be installed.

The commands below are written for Linux, macOS or Git Bash. In Windows CMD change `"$(pwd)":/app` to `"%cd%:/app"`, and in PowerShell to `"${PWD}:/app"`. They must be run from the folder of the repository.

### Experimental Plan

What we wanted to find out:

1. How much faster is the MPI version with static distribution compared to the sequential one?
2. Is the work really unbalanced between the processes, and how much?
3. If we balance it better, does the total time go down?

Before running anything we expected the MPI version to be faster but not 8 times faster with 8 processes, we expected some imbalance because the split is done by number of files and not by size, and we expected version 2 to be a bit faster than version 1.

To make the comparison fair we kept these things the same for the three programs:

- The dataset. `generator.py` uses a fixed seed (42), so it always creates the same files.
- The machine and the Docker image, all in one session.
- The way of counting. `mpi1.py` and `mpi2.py` count the words with the same loop as the baseline (split the line, lowercase, check if the word is in the set). We did not use a faster way of counting because then the speedup would mix two things, the parallelism and a better algorithm.
- The number of processes p. We used the value recommended by the script of the course, [detectar_procesos_mpi.py](https://github.com/augustosalazar/mpi_taller_sudoku/blob/main/detectar_procesos_mpi.py).

What we measured:

- The total time of each program, 5 times each (the lab asks for at least 3). In the MPI programs rank 0 takes the time with `MPI.Wtime()`, starting before it reads `consulta.txt` and stopping when the global result is ready. This is the same part that the baseline measures. The time is printed as `EXECUTION_TIME=...` because that is the format `run_all.sh` looks for.
- For every process, how many files it got, how many words it read and its local time (the time it spent counting its files).

With that we calculated:

- T_seq and T_p, the average time of the sequential program and of each MPI version.
- Speedup, S_p = T_seq / T_p.
- Efficiency, E_p = S_p / p.
- Load imbalance, LI = max(t_i) / mean(t_i), calculated in every run and then averaged. A value of 1.0 is perfect balance.
- Relative spread, (max(t_i) − min(t_i)) / mean(t_i).

To check that the results are correct, the MPI programs compare their full count with the file `dataset/baseline_results.csv` that the baseline saves, and print `MATCHES_BASELINE=True` or `False`. We also compared the top 10.

### Sequential baseline (what it does and how)

`baseline_secuencial.py` reads `consulta.txt` and saves the words in a set. Then it opens the `file_*.txt` files one by one. For each line it splits the words, changes them to lowercase, and if a word is in the set it adds 1 to its counter. At the end it writes all the counts to `dataset/baseline_results.csv` and prints the time and the top 10.

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```

### MPI version 1 (what it does and how)

`mpi1.py` does what the lab describes:

1. Rank 0 reads `consulta.txt`.
2. Rank 0 sends the words to all the processes with `comm.bcast`.
3. Rank 0 gets the list of `file_*.txt` files and sorts it by name.
4. Rank 0 cuts the list in p consecutive parts with the same number of files and sends one part to each process with `comm.scatter`. This is the static distribution. It does not check how big the files are.
5. Each process counts the words in its files and measures how long it took. Rank 0 also counts, it does not only coordinate.
6. The partial results go back to rank 0 with `comm.gather`.
7. Rank 0 adds the counters and prints, for each process, the number of files and the local time. Then it prints the LI, the comparison with the baseline, the top 10 and the total time.

Example with 4 processes:

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi1.py
```

`mpi2.py` is run in the same way. How it works is explained later, together with its results.

### Test procedure

We did the steps in this order, without closing the session:

1. Download `detectar_procesos_mpi.py` into the folder and run it to get p.

   ```bash
   docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py
   ```

2. Run `run_all.sh` with that p and 5 runs. The script creates the dataset again, runs the baseline, then `mpi1.py` 5 times and `mpi2.py` 5 times. It saves a log for every run in `results/logs/`.

   ```bash
   docker run --rm -e NP_LIST=8 -e RUNS=5 -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
   ```

3. `run_all.sh` only runs the baseline once, so we ran it 4 more times with the command shown above to also have 5 times for the average.

4. Take the times from the logs and calculate the metrics.

## Experimental Plan Execution

The runs were done on November 16, 2026, on this machine:

- Laptop with an AMD Ryzen 5 5500U (6 cores, 12 threads)
- Windows 11 with Docker Desktop (WSL2). Inside the container there are 12 CPUs and 7.8 GB of RAM available.
- Python 3.8.20, mpi4py 4.1.1 and Open MPI 4.1.4 (the versions of the image)
- The folder of the project is inside OneDrive and is mounted in the container with `-v`

`detectar_procesos_mpi.py` recommended **p = 8**, so all the MPI runs use 8 processes.

### Sequential baseline timing

| Run | Time (s) |
|---|---|
| 1 | 96.74 |
| 2 | 95.78 |
| 3 | 95.75 |
| 4 | 99.22 |
| 5 | 112.20 |
| **Average (T_seq)** | **99.94** |
| Std. dev. | 7.00 |

The baseline processed the 3000 files, read 44,951,458 words and found 3,631,778 occurrences of the query words. This is its top 10:

| Word | Count |
|---|---|
| a | 785774 |
| para | 392156 |
| sus | 228913 |
| otros | 105530 |
| ante | 99832 |
| unos | 88794 |
| otra | 83901 |
| vosotros | 61617 |
| mios | 58420 |
| tuya | 56635 |

### MPI version 1 timing results

| Run | Total time (s) |
|---|---|
| 1 | 19.39 |
| 2 | 17.74 |
| 3 | 16.89 |
| 4 | 16.67 |
| 5 | 17.06 |
| **Average (T_p)** | **17.55** |
| Std. dev. | 1.11 |

Speedup: S_p = 99.94 / 17.55 = **5.69**

Efficiency: E_p = 5.69 / 8 = **0.71**

In the 5 runs the count was the same as the baseline and the top 10 was also the same.

### Load imbalance evidence (per-process local times and LI)

All the processes got 375 files, but the amount of words was different:

| Rank | Files | Words | Average local time (s) |
|---|---|---|---|
| 0 | 375 | 6,002,531 | 16.84 |
| 1 | 375 | 4,801,328 | 14.63 |
| 2 | 375 | 6,180,576 | 17.47 |
| 3 | 375 | 5,976,248 | 17.00 |
| 4 | 375 | 5,281,651 | 15.75 |
| 5 | 375 | 5,535,328 | 15.87 |
| 6 | 375 | 5,409,609 | 15.82 |
| 7 | 375 | 5,764,187 | 16.58 |

| Run | max t_i (s) | min t_i (s) | mean t_i (s) | LI | Rel. spread |
|---|---|---|---|---|---|
| 1 | 19.32 | 16.30 | 18.03 | 1.072 | 0.167 |
| 2 | 17.67 | 15.14 | 16.61 | 1.063 | 0.152 |
| 3 | 16.80 | 14.35 | 15.82 | 1.062 | 0.155 |
| 4 | 16.59 | 13.07 | 14.93 | 1.111 | 0.236 |
| 5 | 16.98 | 14.27 | 15.83 | 1.073 | 0.171 |
| **Average** | | | | **1.076** | **0.176** |

Rank 2 got the most words (6.18 million) and it was the slowest process in the 5 runs. Rank 1 got the fewest (4.80 million) and it was always the fastest. Rank 2 had 29% more words than rank 1 and took 19% more time. In every run rank 1 finished about 3 seconds before the program ended and after that it was doing nothing.

### MPI version 2: how it corrects the imbalance, and its results

The problem of version 1 is that the files are assigned before knowing how much work each one is. Giving the same number of files is not the same as giving the same work.

In `mpi2.py` the files are not assigned at the beginning. The processes ask for work when they need it:

1. Rank 0 reads the query and the list of files and sends both to everyone with `comm.bcast`.
2. Each worker sends a request to rank 0, and rank 0 answers with a group of files. Since all the processes already have the list, rank 0 only sends two numbers: where the group starts and how many files it has.
3. The size of the group depends on how many files are left. Each request gets about (files left) / (8 × p). At the beginning the groups are big (46 files with p = 8), so there are few messages. At the end they are of 1 file, so no process is left with a lot of pending work while the others have finished.
4. When a worker receives a group, it asks for the next one immediately and then starts counting. That way the answer is usually already there when it finishes.
5. Rank 0 also counts. It takes files one at a time from the end of the list, and between one file and the next it checks with `Iprobe` if there are requests and answers them.
6. When there are no files left, rank 0 tells the workers to stop, receives their results and builds the global result like in version 1.

A process that got light files finishes its group early and asks for more, so the work is balanced without knowing the size of the files.

We had a previous attempt that did not work well and we think it is worth mentioning. In that one, rank 0 first asked for the size of the 3000 files to give the biggest ones first. The balance was almost perfect (LI of 1.007), but the total time was between 26 and 33 seconds, worse than version 1. The processes were counting for only 15 to 20 seconds, and the rest was lost at the start. We believe it was the reading of the 3000 file sizes, which is very slow through the Docker mount. We removed that part and the final version does not use the sizes.

Results of the final `mpi2.py` with p = 8:

| Run | Total time (s) | max t_i (s) | min t_i (s) | LI | Rel. spread |
|---|---|---|---|---|---|
| 1 | 18.15 | 17.99 | 16.98 | 1.038 | 0.058 |
| 2 | 16.04 | 15.65 | 14.68 | 1.039 | 0.065 |
| 3 | 18.49 | 18.18 | 17.41 | 1.023 | 0.043 |
| 4 | 17.73 | 17.40 | 15.80 | 1.066 | 0.098 |
| 5 | 17.22 | 16.75 | 15.99 | 1.032 | 0.047 |
| **Average** | **17.53** | | | **1.040** | **0.062** |

The average local time of rank 0 was 17.19 s. For ranks 1 to 7 it was between 16.35 s and 16.55 s. The counts matched the baseline in the 5 runs.

Both versions side by side (T_seq = 99.94 s):

| Version | p | T_p (s) | Std. dev. (s) | Speedup | Efficiency | LI | Rel. spread |
|---|---|---|---|---|---|---|---|
| mpi1 (static) | 8 | 17.55 | 1.11 | 5.69 | 0.71 | 1.076 | 0.176 |
| mpi2 (dynamic) | 8 | 17.53 | 0.96 | 5.70 | 0.71 | 1.040 | 0.062 |

## Analysis

### Parallel performance

Version 1 did improve the baseline. The time went from 99.94 s to 17.55 s, which is a speedup of 5.69. The ideal with 8 processes would be 8, so we got 71% of it.

Most of the difference comes from the fact that the processes are slower when the 8 run at the same time. If we add the local times of the 8 processes we get about 130 s, but the sequential program does the same work in about 100 s. So the same work costs 30% more in parallel. We think there are two reasons. The laptop has 6 real cores, so with 8 processes some of them share a core. And all the processes read files from the same disk through the Docker mount.

The second cause is the imbalance. The program does not end until the slowest process ends (17.47 s on average), while the average process needs 16.24 s. With a perfect balance the time would be around 16.3 s and the speedup around 6.1.

The communication is almost nothing. The total time is only 0.08 s more than the time of the slowest process, and that includes the broadcast, the scatter, the gather and adding the results.

These numbers are consistent with each other: 8 / (1.30 × 1.076) is about 5.7.

### Load balance

In version 1 the LI was between 1.06 and 1.11 in the 5 runs, with an average of 1.076, and the relative spread was 0.176. We are sure it is a real imbalance and not noise because the slowest and the fastest process were the same ones in every run, and they are exactly the ones that received the most and the fewest words.

Version 2 did reduce it. The LI went down to 1.040 and the spread to 0.062. The 7 workers ended with almost the same local time (16.35 to 16.55 s).

But the total time did not improve. It was 17.55 s before and 17.53 s now. That difference of 0.02 s means nothing when the times change by about 1 s from one run to another. So the answer is no, the reduction was not enough to give a real improvement.

Looking at the data we found two reasons. First, there was not much to win. Even removing all the imbalance of version 1 the gain would be 7% at most, about 1.2 s, which is close to the noise we have. Second, the dynamic version has a cost. Rank 0 was the slowest process in the 5 runs (17.19 s against about 16.45 s for the workers). Rank 0 can only answer requests when it finishes the file it is reading, so near the end the workers have to wait for it. What we gained with the balance we lost there.

The failed attempt showed the same thing in a stronger way. It had the best LI of all and it was the slowest. A better LI does not always mean a shorter time.

### Limitations

The biggest limitation is that reading the files takes most of the time. Our baseline takes about 100 s, and the example output in the repository of the course shows about 5 s for the same dataset. The difference is that our files are in a OneDrive folder mounted in Docker, and that is slow. So our speedup is partly measuring how well the file reading works in parallel, and on another machine the numbers could be quite different.

There is also a lot of variation between runs, around 6 or 7%. For example the fifth run of the baseline took 112 s when the others took 96 to 99 s. If we leave that run out, T_seq is 96.9 s and the speedup of version 1 is 5.52 instead of 5.69. Because of this variation we cannot say which MPI version is faster.

The programs were not run in mixed order. First all the runs of version 1 and then all of version 2, so if the laptop got hotter or something was running in the background it could affect one version more than the other.

We only tested one machine, one dataset and p = 8. We do not know what happens with other values of p or with files of other sizes.

The measured time does not include starting `mpirun` and Python, so the real speedup for a user is a bit lower.

Even with these limitations, some results are clear because the differences are large or they repeated in all the runs: version 1 is between 5 and 6 times faster than the baseline, it has an imbalance that comes from the size of the files, and version 2 reduces the LI. What we cannot conclude is that one MPI version is faster than the other.

## Conclusions

For this problem we would recommend the dynamic distribution of version 2, but being honest about what we measured. In time, both versions are the same: speedup of 5.69 and efficiency of 0.71 for the static one, and 5.70 and 0.71 for the dynamic one. The difference is in the balance, where the LI went from 1.076 to 1.040. We prefer the dynamic one because it does not need to know the size of the files and it adapts by itself if a process is slower or if the files are more unequal, and with a more unbalanced dataset the static version would get worse while the dynamic one should not. In our case the imbalance of the static version was only 7.6% and the cost of coordinating was about the same, so there was no gain in time, and for a dataset like this one the static version is also a valid option because it is simpler. What limited the efficiency in both versions was not the distribution but the fact that 8 processes running together on 6 cores are slower than one alone.
