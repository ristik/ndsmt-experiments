#!/usr/bin/env python3

import time
import os
import sys
import hashlib
import tempfile
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ndrsmt import SparseMerkleTree, verify_consistency


def to_int(aa):
    if isinstance(aa, bytes):
        return int.from_bytes(aa, byteorder="big")
    else:
        h = hashlib.sha256()
        h.update(str(aa).encode())
        return int.from_bytes(h.digest(), byteorder="big")


def run_measurement(batch_size, pre_insert_count, depth=256):
    # Create fresh tree
    smt = SparseMerkleTree(depth=depth)

    # Pre-fill to equal state - use separate key space
    pre_batch = []
    for i in range(pre_insert_count):
        rk = hash(f"pre_i{i}") % (2**depth)
        rv = to_int(f"Vpre{rk}")
        pre_batch.append((rk, rv))

    smt.batch_insert(pre_batch)

    # Now measure throughput for new batch - use different key space
    batch = []
    for i in range(batch_size):
        rk = hash(f"meas_i{i}") % (2**depth)
        rv = to_int(f"Vmeas{rk}")
        batch.append((rk, rv))

    # Get root BEFORE insert
    pre_root = smt.get_root()

    t0 = time.time()
    b, proof = smt.batch_insert(batch)
    t_insert = time.time() - t0

    # Verify - get root AFTER
    new_root = smt.get_root()

    ok = verify_consistency(proof, pre_root, new_root, b, depth)
    if not ok:
        print(
            f"WARNING: Verification failed for batch_size={batch_size}", file=sys.stderr
        )

    throughput = len(batch) / t_insert if t_insert > 0 else float("inf")
    return throughput


def main():
    batch_sizes = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    pre_insert_count = 1000000  # Same pre-inserted leaves for each measurement
    depth = 256
    num_trials = 3

    print(
        f"Pre-inserting {pre_insert_count} leaves before each measurement",
        file=sys.stderr,
    )
    print(f"{'batch_size':>12} {'throughput (leaves/sec)':>25}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    results = {}
    for bs in batch_sizes:
        trials = []
        for trial in range(num_trials):
            throughput = run_measurement(bs, pre_insert_count, depth)
            trials.append(throughput)

        avg_throughput = sum(trials) / len(trials)
        results[bs] = avg_throughput
        print(f"{bs:>12} {avg_throughput:>25.0f}", file=sys.stderr)

    # Plot - batch size on x-axis (equal spacing), throughput on y-axis
    plt.figure(figsize=(10, 6))

    batch_sizes = list(results.keys())
    throughputs = list(results.values())

    # Use categorical x-axis with equal spacing
    x_positions = range(len(batch_sizes))

    plt.plot(x_positions, throughputs, "bo-", linewidth=2, markersize=8)

    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Throughput (new leaves/sec)", fontsize=12)
    plt.title(
        f"ndrsmt: Batch Size vs Throughput\n(Pre-filled with {pre_insert_count} leaves)",
        fontsize=14,
    )

    # Set x-ticks to batch sizes with equal spacing
    plt.xticks(x_positions, batch_sizes)

    # Start y-axis (throughput) at zero
    plt.ylim(bottom=0)

    plt.grid(True, alpha=0.3)

    # Add annotations
    for i, (bs, tp) in enumerate(results.items()):
        plt.annotate(
            f"{tp:.0f}",
            (i, tp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("graph3_batch_vs_throughput.png", dpi=150)
    plt.close()
    print("\nSaved graph3_batch_vs_throughput.png", file=sys.stderr)


if __name__ == "__main__":
    main()
