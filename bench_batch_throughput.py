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

from ndrsmt2 import SparseMerkleTree as SMT_ndrsmt, verify_consistency as vc_ndrsmt
from ndrsmt3o import SparseMerkleTree as SMT_ndrsmt2, verify_consistency as vc_ndrsmt2


def to_int(aa):
    if isinstance(aa, bytes):
        return int.from_bytes(aa, byteorder="big")
    else:
        h = hashlib.sha256()
        h.update(str(aa).encode())
        return int.from_bytes(h.digest(), byteorder="big")


def run_measurement(SMT, verify_consistency, batch_size, pre_insert_count, depth=256):
    smt = SMT(depth=depth)

    # Pre-fill - use separate key space
    pre_batch = []
    for i in range(pre_insert_count):
        rk = hash(f"pre_i{i}") % (2**depth)
        rv = f"Vpre{rk}".encode(encoding="utf-8")
        pre_batch.append((rk, rv))

    smt.batch_insert(pre_batch)

    # Measure throughput for new batch - use different key space
    batch = []
    for i in range(batch_size):
        rk = hash(f"meas_i{i}") % (2**depth)
        rv = f"Vmeas{rk}".encode(encoding="utf-8")
        batch.append((rk, rv))

    pre_root = smt.get_root()

    t0 = time.time()
    b, proof = smt.batch_insert(batch)
    t_insert = time.time() - t0

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

    implementations = [
        ("ndrsmt2",  SMT_ndrsmt,  vc_ndrsmt,  "red",    "^", "-"),
        ("ndrsmt3o", SMT_ndrsmt2, vc_ndrsmt2, "purple", "D", "--"),
    ]

    print(
        f"Pre-inserting {pre_insert_count} leaves before each measurement",
        file=sys.stderr,
    )

    all_results = {}
    for impl_name, SMT, verify_consistency, _, _, _ in implementations:
        print(f"\n{'batch_size':>12} {'throughput (leaves/sec)':>25}  [{impl_name}]", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        all_results[impl_name] = {}
        for bs in batch_sizes:
            trials = [run_measurement(SMT, verify_consistency, bs, pre_insert_count, depth)
                      for _ in range(num_trials)]
            avg = sum(trials) / len(trials)
            all_results[impl_name][bs] = avg
            print(f"{bs:>12} {avg:>25.0f}", file=sys.stderr)

    # Plot
    plt.figure(figsize=(10, 6))
    x_positions = range(len(batch_sizes))

    for impl_name, _, _, color, marker, ls in implementations:
        throughputs = [all_results[impl_name][bs] for bs in batch_sizes]
        plt.plot(x_positions, throughputs, marker=marker, color=color,
                 linestyle=ls, linewidth=2, markersize=8, label=impl_name, alpha=0.85)

    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Throughput (new leaves/sec)", fontsize=12)
    plt.title(
        f"Batch Size vs Throughput\n(Pre-filled with {pre_insert_count} leaves)",
        fontsize=14,
    )
    plt.xticks(x_positions, batch_sizes)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graph3_batch_vs_throughput.png", dpi=150)
    plt.close()
    print("\nSaved graph3_batch_vs_throughput.png", file=sys.stderr)


if __name__ == "__main__":
    main()
