#!/usr/bin/env python3

import time
import os
import sys
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pympler import asizeof

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nc_ndsmt import SparseMerkleTree as SMT_ncndsmt, verify_consistency as vc_ncndsmt
from ndsmt import SparseMerkleTree as SMT_ndsmt, verify_consistency as vc_ndsmt
from ndsmt_opt import (
    SparseMerkleTree as SMT_ndsmt_opt,
    verify_consistency as vc_ndsmt_opt,
)
from ndrsmt import SparseMerkleTree as SMT_ndrsmt, verify_consistency as vc_ndrsmt


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
    module_name, SMT, verify_consistency, depth=256, batch_size=10000, num_rounds=60
):
    smt = SMT(depth=depth)
    total_leaves = 0

    results = {
        "total_leaves": [],
        "insert_speed": [],
        "verify_speed": [],
        "ins_per_sec": [],
        "memory_mb": [],
        "proof_size_mb": [],
    }

    for rnd in range(num_rounds):
        batch = []
        for i in range(batch_size):
            rk = hash(f"r{rnd}_i{i}") % (2**depth)
            rv = to_int(f"V{rk}")
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

    return results


def main():
    # batch_sizes = [100, 500, 1000, 5000]
    batch_sizes = [1000]
    num_rounds = 30
    depth = 256

    implementations = [
        ("nc-ndsmt", SMT_ncndsmt, vc_ncndsmt),
        ("ndsmt", SMT_ndsmt, vc_ndsmt),
        ("ndsmt_opt", SMT_ndsmt_opt, vc_ndsmt_opt),
        ("ndrsmt", SMT_ndrsmt, vc_ndrsmt),
    ]

    all_results = {}

    for bs in batch_sizes:
        print(f"\n=== Batch size: {bs} ===", file=sys.stderr)
        all_results[bs] = {}

        for name, SMT, verify_consistency in implementations:
            print(f"Running {name}...", file=sys.stderr)
            results = run_benchmark(
                name,
                SMT,
                verify_consistency,
                depth=depth,
                batch_size=bs,
                num_rounds=num_rounds,
            )
            all_results[bs][name] = results

    colors = {
        "nc-ndsmt": "orange",
        "ndsmt": "blue",
        "ndsmt_opt": "green",
        "ndrsmt": "red",
    }
    markers = {"nc-ndsmt": "x", "ndsmt": "o", "ndsmt_opt": "s", "ndrsmt": "^"}
    linestyles = {"nc-ndsmt": ":", "ndsmt": "-", "ndsmt_opt": "--", "ndrsmt": "-."}

    plt.figure(figsize=(12, 8))
    for bs in batch_sizes:
        for name in ["nc-ndsmt", "ndsmt", "ndsmt_opt", "ndrsmt"]:
            x = all_results[bs][name]["total_leaves"]
            y = all_results[bs][name]["ins_per_sec"]
            plt.plot(
                x,
                y,
                marker=markers[name],
                color=colors[name],
                linestyle="-" if bs == batch_sizes[0] else "--",
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

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    for name in ["nc-ndsmt", "ndsmt", "ndsmt_opt", "ndrsmt"]:
        x = all_results[batch_sizes[0]][name]["total_leaves"]
        y = all_results[batch_sizes[0]][name]["insert_speed"]
        plt.plot(x, y, marker=markers[name], color=colors[name], label=name)
    plt.xlabel("Total Leaves")
    plt.ylabel("Time (seconds)")
    plt.title("Batch Insertion Speed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    for name in ["nc-ndsmt", "ndsmt", "ndsmt_opt", "ndrsmt"]:
        x = all_results[batch_sizes[0]][name]["total_leaves"]
        y = all_results[batch_sizes[0]][name]["verify_speed"]
        plt.plot(x, y, marker=markers[name], color=colors[name], label=name)
    plt.xlabel("Total Leaves")
    plt.ylabel("Time (seconds)")
    plt.title("Proof Verification Speed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    for name in ["nc-ndsmt", "ndsmt", "ndsmt_opt", "ndrsmt"]:
        x = all_results[batch_sizes[0]][name]["total_leaves"]
        y = all_results[batch_sizes[0]][name]["proof_size_mb"]
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
    for name in ["nc-ndsmt", "ndsmt", "ndsmt_opt", "ndrsmt"]:
        x = all_results[batch_sizes[0]][name]["total_leaves"]
        y = all_results[batch_sizes[0]][name]["memory_mb"]
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

    print("\n=== Summary ===", file=sys.stderr)
    for bs in batch_sizes:
        print(f"\nBatch size {bs}:", file=sys.stderr)
        for name in ["nc-ndsmt", "ndsmt", "ndsmt_opt", "ndrsmt"]:
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
