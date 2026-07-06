#!/usr/bin/env python3

import hashlib
import importlib
import inspect
import itertools
import os
import shutil
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pympler import asizeof

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================

# Just add your module names here. The script will automatically import
# `SparseMerkleTree` and `verify_consistency` from each module.
MODULES_TO_BENCH = [
    "ndrsmt",
    "ndrsmt3o",
    "rsmt6",
    "rsmt6a",
]

BATCH_SIZES = [10000]
NUM_ROUNDS = 30
DEPTH = 256

# ==========================================


def to_int(aa):
    if isinstance(aa, (list, tuple)):
        return [to_int(a) for a in aa]
    elif isinstance(aa, bytes):
        return int.from_bytes(aa, byteorder="big")
    else:
        h = hashlib.sha256()
        h.update(str(aa).encode())
        return int.from_bytes(h.digest(), byteorder="big")


def run_benchmark(
    module_name, SMT, verify_consistency, depth=256, batch_size=1000, num_rounds=60
):
    # Dynamically check if the SMT class accepts a 'db_path' argument (stateful DB backed)
    sig = inspect.signature(SMT.__init__)
    if "db_path" in sig.parameters:
        tmpdir = tempfile.mkdtemp()
        smt = SMT(depth=depth, db_path=tmpdir)
        cleanup = lambda: shutil.rmtree(tmpdir)
    else:
        smt = SMT(depth=depth)
        cleanup = None

    total_leaves = 0

    results = {
        "total_leaves": [],
        "insert_speed": [],
        "verify_speed": [],
        "ins_per_sec": [],
        "memory_mb": [],
        "proof_size_mb": [],
    }

    try:
        for rnd in range(num_rounds):
            batch = []
            for i in range(batch_size):
                rk = hash(f"r{rnd}_i{i}") % (2**depth)
                rv = hashlib.sha256(b"Value" + rk.to_bytes(32)).digest()
                batch.append((rk, rv))

            old_root = smt.get_root()

            t0 = time.time()
            b, proof = smt.batch_insert(batch)
            t_insert = time.time() - t0

            new_root = smt.get_root()

            t0 = time.time()
            ok = verify_consistency(proof, old_root, new_root, b, depth)
            t_verify = time.time() - t0
            assert ok, f"Verification failed at round {rnd}"

            total_leaves += len(batch)
            ips = len(batch) / t_insert if t_insert > 0 else float("inf")

            mem_mb = asizeof.asizeof(smt) / 1024 / 1024
            prf_size = asizeof.asizeof(proof) / 1024 / 1024

            results["total_leaves"].append(total_leaves)
            results["insert_speed"].append(t_insert)
            results["verify_speed"].append(t_verify)
            results["ins_per_sec"].append(ips)
            results["memory_mb"].append(mem_mb)
            results["proof_size_mb"].append(prf_size)
    finally:
        if cleanup:
            cleanup()

    return results


def main():
    # Load implementations dynamically
    implementations = []
    for mod_name in MODULES_TO_BENCH:
        try:
            mod = importlib.import_module(mod_name)
            SMT = getattr(mod, "SparseMerkleTree")
            vc = getattr(mod, "verify_consistency")
            implementations.append((mod_name, SMT, vc))
        except Exception as e:
            print(f"Failed to load module '{mod_name}': {e}", file=sys.stderr)
            sys.exit(1)

    all_results = {}

    for bs in BATCH_SIZES:
        print(f"\n=== Batch size: {bs} ===", file=sys.stderr)
        all_results[bs] = {}

        for name, SMT, verify_consistency in implementations:
            print(f"Running {name}...", file=sys.stderr)
            results = run_benchmark(
                name,
                SMT,
                verify_consistency,
                depth=DEPTH,
                batch_size=bs,
                num_rounds=NUM_ROUNDS,
            )
            all_results[bs][name] = results

    # Generate dynamic styles mapping
    color_cycle = itertools.cycle(
        ["red", "purple", "green", "orange", "blue", "brown", "magenta", "cyan"]
    )
    marker_cycle = itertools.cycle(["^", "D", "s", "o", "v", "<", ">", "p", "*"])
    linestyle_cycle = itertools.cycle(
        ["-.", (0, (3, 1, 1, 1)), "--", ":", "-", (0, (5, 1))]
    )

    colors = {}
    markers = {}
    linestyles = {}
    for name in MODULES_TO_BENCH:
        colors[name] = next(color_cycle)
        markers[name] = next(marker_cycle)
        linestyles[name] = next(linestyle_cycle)

    # Plot 1: Throughput
    plt.figure(figsize=(12, 8))
    for bs in BATCH_SIZES:
        for name, _, _ in implementations:
            x = all_results[bs][name]["total_leaves"]
            y = all_results[bs][name]["ins_per_sec"]
            plt.plot(
                x,
                y,
                marker=markers[name],
                color=colors[name],
                linestyle="-" if bs == BATCH_SIZES[0] else "--",
                label=f"{name} (batch={bs})",
                alpha=0.7,
            )

    plt.xlabel("Total Leaves (pre-existing)")
    plt.ylabel("Insertions per second (tx/s)")
    plt.title("Batch Insertion Throughput vs Tree Size")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graph1_throughput.png", dpi=150)
    plt.close()
    print("Saved graph1_throughput.png", file=sys.stderr)

    # Plot 2: Metrics Subplots
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    for name, _, _ in implementations:
        x = all_results[BATCH_SIZES[0]][name]["total_leaves"]
        y = all_results[BATCH_SIZES[0]][name]["insert_speed"]
        plt.plot(x, y, marker=markers[name], color=colors[name], label=name)
    plt.xlabel("Total Leaves")
    plt.ylabel("Time (seconds)")
    plt.title("Batch Insertion Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    for name, _, _ in implementations:
        x = all_results[BATCH_SIZES[0]][name]["total_leaves"]
        y = all_results[BATCH_SIZES[0]][name]["verify_speed"]
        plt.plot(x, y, marker=markers[name], color=colors[name], label=name)
    plt.xlabel("Total Leaves")
    plt.ylabel("Time (seconds)")
    plt.title("Proof Verification Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    for name, _, _ in implementations:
        x = all_results[BATCH_SIZES[0]][name]["total_leaves"]
        y = all_results[BATCH_SIZES[0]][name]["proof_size_mb"]
        plt.plot(
            x,
            y,
            marker=markers[name],
            color=colors[name],
            label=name,
            linewidth=2,
            linestyle=linestyles[name],
        )
    plt.xlabel("Total Leaves")
    plt.ylabel("Size (MB)")
    plt.title("Proof Size")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    for name, _, _ in implementations:
        x = all_results[BATCH_SIZES[0]][name]["total_leaves"]
        y = all_results[BATCH_SIZES[0]][name]["memory_mb"]
        plt.plot(x, y, marker=markers[name], color=colors[name], label=name)
    plt.xlabel("Total Leaves")
    plt.ylabel("Memory (MB)")
    plt.title("Total Memory Usage")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph2_metrics.png", dpi=150)
    plt.close()
    print("Saved graph2_metrics.png", file=sys.stderr)

    # Summary Output
    print("\n=== Summary ===", file=sys.stderr)
    for bs in BATCH_SIZES:
        print(f"\nBatch size {bs}:", file=sys.stderr)
        for name, _, _ in implementations:
            r = all_results[bs][name]
            final_ins = r["ins_per_sec"][-1]
            final_mem = r["memory_mb"][-1]
            final_prf = r["proof_size_mb"][-1]
            print(
                f"  {name}: {final_ins:.0f} tx/s, {final_mem:.1f} MB memory, {final_prf:.2f} MB proof",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
