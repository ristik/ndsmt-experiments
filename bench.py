#!/usr/bin/env python3

import time
import os
import sys
import hashlib
import psutil
import importlib

def get_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, (tuple, list)):
        for item in obj:
            size += get_size(item, seen)
    return size


def to_int(aa):
    if isinstance(aa, (list, tuple)):
        return [to_int(a) for a in aa]
    elif isinstance(aa, bytes):
        return int.from_bytes(aa, byteorder="big")
    else:
        # Ensure 256-bit (32-byte) output by using SHA-256 hash
        h = hashlib.sha256()
        h.update(str(aa).encode())
        return int.from_bytes(h.digest(), byteorder="big")


def run_benchmark(depth=256, batch_size=10000, num_rounds=60):

    smt = SparseMerkleTree(depth=depth)
    total_leaves = 0

    print(
        f"{'round':>5} {'batch':>5} {'total':>7} {'insert_s':>9} "
        f"{'verify_s':>9} {'ins/s':>7} {'mem_MB':>8} {'prf_MB':>10}"
    )
    print("-" * 76)

    for rnd in range(num_rounds):
        batch = []
        for i in range(batch_size):
            rk = hash(f"r{rnd}_i{i}") % (2**depth)
            # rv = to_int(f"V{rk}
            rv = b"value"
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

        # Measure memory usage
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024

        # Measure proof size
        prf_size = get_size(proof) / 1024 / 1024

        print(
            f"{rnd:5d} {len(batch):5d} {total_leaves:7d} {t_insert:9.3f} "
            f"{t_verify:9.3f} {ips:7.0f} {mem_mb:8.1f} {prf_size:8.1f}"
        )

    # print(f"\nFinal: {total_leaves} leaves", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: bench.py <module_name> [batch_size] [num_rounds] [depth]")
        sys.exit(0)
    if len(sys.argv) < 2:
        print(
            "Usage: bench.py <module_name> [batch_size] [num_rounds] [depth]",
            file=sys.stderr,
        )
        sys.exit(1)
    module_name = sys.argv[1]
    mod = importlib.import_module(module_name)
    SparseMerkleTree = mod.SparseMerkleTree
    verify_consistency = mod.verify_consistency

    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    num_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    run_benchmark(depth=depth, batch_size=batch_size, num_rounds=num_rounds)
