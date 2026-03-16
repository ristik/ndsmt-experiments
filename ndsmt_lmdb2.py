#!/usr/bin/env python3
"""
NOTE: caching as done here is not the best idea.
This one tries to implement caching of certain number of SMT layers closer
to the root.

LMDB-backed Sparse Merkle Tree with pass-through branch compression.

Storage layout:
  leaves:   leaf_key  (32 B big-endian) → leaf_data  (variable bytes)
  branches: level (2 B) + node_key (32 B) → node_hash (32 B SHA-256 digest)
            Only stored when BOTH children are non-empty.
  meta:     b'root' → root_hash (32 B)

Close-to-root cache analysis:
  With the old scheme the cache saved O(D - cache_level) SHA-256 ops + O(2)
  cursor reads.  With pass-through the SHA-256 saving is gone; chain hashes
  are trivially the same as the binary node hash already in branches_db.
  The cache still saves O(2) cursor reads per sibling lookup and one branches_db
  read -- worthwhile for hot upper-level nodes that are siblings of many batch
  keys across rounds.  Cache invalidation is unchanged.
"""

import hashlib
import sys
import json
import time
import os
import shutil
import bisect
from collections import deque
import lmdb

# ---------------------------------------------------------------------------
# Minimal CBOR encoder (handles None, unsigned int, bytes, list)
# ---------------------------------------------------------------------------

def cbor_encode(value):
    """Encode a value as CBOR bytes. Supports None, int >= 0, bytes, list."""
    if value is None:
        return b'\xf6'                                      # CBOR null
    if isinstance(value, int):
        if value < 0:
            raise ValueError("negative integers not supported")
        if value < 2**64:
            return _cbor_head(0, value)                     # major 0: uint
        # Tag 2: positive bignum as byte string
        n = (value.bit_length() + 7) // 8
        return b'\xc2' + cbor_encode(value.to_bytes(n, 'big'))
    if isinstance(value, (bytes, bytearray)):
        return _cbor_head(2, len(value)) + value            # major 2: bstr
    if isinstance(value, (list, tuple)):
        body = b''.join(cbor_encode(v) for v in value)
        return _cbor_head(4, len(value)) + body             # major 4: array
    raise TypeError(f"cbor_encode: unsupported {type(value)}")

def _cbor_head(major, n):
    m = major << 5
    if n < 24:       return bytes([m | n])
    if n < 0x100:    return bytes([m | 24, n])
    if n < 0x10000:  return bytes([m | 25]) + n.to_bytes(2, 'big')
    if n < 2**32:    return bytes([m | 26]) + n.to_bytes(4, 'big')
    return bytes([m | 27]) + n.to_bytes(8, 'big')

# ---------------------------------------------------------------------------
# SMT node hashing  (matches ndsmt2.py)
# ---------------------------------------------------------------------------

EMPTY = None  # ⊥ -- empty node / absent value

def path_at_level(key, level, depth):
    """
    Path segment for node (level, key).
    Encodes position as sentinel-terminated bit-string: key | (1 << (depth - level)).
    'key' is the node index at this level (= leaf_key >> level).
    """
    return key | (1 << (depth - level))

def hash_leaf(path_segment, data):
    """smt_leaf_hash(p, d) = SHA-256(CBOR([p, d]))"""
    return hashlib.sha256(cbor_encode([path_segment, data])).digest()

def hash_branch(path_segment, h_left, h_right):
    """
    smt_branch_hash with pass-through compression:
      - Both empty  → EMPTY
      - One empty   → the other child (pass-through; no hash computed)
      - Both present → SHA-256(CBOR([p, h_L, h_R]))
    """
    if h_left is EMPTY:
        return h_right
    if h_right is EMPTY:
        return h_left
    return hashlib.sha256(cbor_encode([path_segment, h_left, h_right])).digest()

# ---------------------------------------------------------------------------
# LMDB key encoding helpers
# ---------------------------------------------------------------------------

def _enc_key(key):
    """Encode an integer tree key as 32 big-endian bytes (LMDB dict key)."""
    return key.to_bytes(32, 'big')

def _dec_key(b):
    """Decode 32 big-endian bytes to integer tree key."""
    return int.from_bytes(b, 'big')

def _bkey(level, key):
    """LMDB key for a branch node: 2-byte level + 32-byte node key."""
    return level.to_bytes(2, 'big') + _enc_key(key)

# ---------------------------------------------------------------------------
# LMDB-backed Sparse Merkle Tree
# ---------------------------------------------------------------------------

class SparseMerkleTree:
    def __init__(self, db_path, depth=256, map_size=1024**4, cache_levels=32):
        self.depth = depth
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.env = lmdb.open(db_path, max_dbs=3, map_size=map_size)
        self.leaves_db   = self.env.open_db(b'leaves')
        self.branches_db = self.env.open_db(b'branches')
        self.meta_db     = self.env.open_db(b'meta')

        # Persistent cache for upper-level node hashes.
        # With pass-through, saves O(2) cursor reads per sibling lookup
        self._node_cache = {}           # (level, key) → hash bytes
        self._cache_min_level = depth - cache_levels

        # Batch proof cache: deque of (proof, batch_key_set) for recent rounds.
        self._batch_proof_cache = deque(maxlen=2)

    def close(self):
        self.env.close()

    def get_root(self):
        with self.env.begin() as txn:
            data = txn.get(b'root', db=self.meta_db)
            return bytes(data) if data else EMPTY

    # -------------------------------------------------------------------
    # Leaf range helpers -- O(1) LMDB cursor access
    # -------------------------------------------------------------------

    def _get_first_leaf(self, txn, lo, hi):
        """First leaf key in [lo, hi]. Returns (int_key, bytes_data) or None."""
        cur = txn.cursor(db=self.leaves_db)
        try:
            if not cur.set_range(_enc_key(lo)):
                return None
            k = _dec_key(cur.key())
            return (k, bytes(cur.value())) if k <= hi else None
        finally:
            cur.close()

    def _get_last_leaf(self, txn, lo, hi):
        """Last leaf key in [lo, hi]. Returns (int_key, bytes_data) or None."""
        cur = txn.cursor(db=self.leaves_db)
        try:
            bound = _enc_key(hi + 1) if hi < (1 << self.depth) - 1 else None
            if bound and cur.set_range(bound):
                if not cur.prev():
                    return None
            elif not cur.last():
                return None
            k = _dec_key(cur.key())
            return (k, bytes(cur.value())) if lo <= k <= hi else None
        finally:
            cur.close()

    # -------------------------------------------------------------------
    # Node hash lookup -- O(3) DB reads via first/last leaf + XOR trick
    # -------------------------------------------------------------------

    def _get_existing_hash(self, txn, level, key):
        """
        Hash of node (level, key) in the current (pre-batch) tree state.

        With pass-through compression, a chain node's hash equals the
        child's hash unchanged -- so:
          Single leaf subtree → hash is just the leaf hash at any level.
          Multi-leaf subtree  → hash is the highest binary node's hash.

        Strategy:
          1. Persistent cache                   → O(1)
          2. Branch DB lookup (binary node)     → O(1)
          3. First + last leaf in subtree       → O(2) cursor reads
             a. Single leaf → leaf hash         (no chain computation)
             b. Multiple leaves → XOR trick →
                highest binary node → DB read   → O(1)
        """
        nc = self._node_cache
        if level >= self._cache_min_level:
            cached = nc.get((level, key))
            if cached is not None:
                return cached

        if level == 0:
            data = txn.get(_enc_key(key), db=self.leaves_db)
            if data is None:
                return EMPTY
            return hash_leaf(path_at_level(key, 0, self.depth), bytes(data))

        # Check branch DB (stored binary node)
        data = txn.get(_bkey(level, key), db=self.branches_db)
        if data is not None:
            h = bytes(data)
            if level >= self._cache_min_level:
                nc[(level, key)] = h
            return h

        # Find first and last leaf in this subtree
        lo = key << level
        hi = ((key + 1) << level) - 1
        first = self._get_first_leaf(txn, lo, hi)
        if first is None:
            return EMPTY

        last = self._get_last_leaf(txn, lo, hi)

        if first[0] == last[0]:
            # Single leaf: hash is the leaf hash, propagated by pass-through.
            leaf_key, leaf_data = first
            result = hash_leaf(path_at_level(leaf_key, 0, self.depth), leaf_data)
        else:
            # Multiple leaves: XOR of first and last gives the divergence level
            # = the highest binary node.  Its hash propagates up unchanged.
            xor_val = first[0] ^ last[0]
            b_level = xor_val.bit_length()          # level of highest binary node
            b_key   = first[0] >> b_level

            data = txn.get(_bkey(b_level, b_key), db=self.branches_db)
            assert data is not None, f"Missing binary node ({b_level}, {b_key})"
            result = bytes(data)
            # result is valid at all levels >= b_level (pass-through)

        if level >= self._cache_min_level:
            nc[(level, key)] = result
        return result

    # -------------------------------------------------------------------
    # Batch insert with consistency-proof generation
    # -------------------------------------------------------------------

    def batch_insert(self, batch):
        """
        Insert (key, data_bytes) pairs and return (new_items, proof).

        new_items: sorted list of actually inserted (key, data_bytes) pairs.
        proof:     list of length depth; proof[level] = [(sibling_key, hash), ...]
                   Only non-empty siblings appear (binary branch levels only).
                   Proof is O(b·log N) entries, not O(b·D).

        Algorithm:
          Phase A -- independent per-key processing up to first_merge_level (fml):
            With pass-through, chain levels produce no hash change.  We skip
            directly to levels with a non-empty sibling (binary branch points),
            compute hash_branch there, and continue.  No chain-hash loop needed.

          Phase B -- level-by-level merge from fml to root:
            Standard bottom-up propagation; hash_branch handles pass-through
            automatically (chain parents carry the child hash unchanged).
            Includes early termination when a single key remains.

        Total: O(b·D) worst-case; O(b·log b + b·log N) typical.
        """
        depth = self.depth

        with self.env.begin() as rtxn:
            # Filter already-present keys; dict deduplicates batch
            new_items_dict = {}
            for key, data in batch:
                if rtxn.get(_enc_key(key), db=self.leaves_db) is not None:
                    print(f"Leaf {key} already set, skipping.", file=sys.stderr)
                elif key not in new_items_dict:
                    new_items_dict[key] = data

            if not new_items_dict:
                return [], [[] for _ in range(depth)]

            new_items  = sorted(new_items_dict.items())
            sorted_keys = [k for k, _ in new_items]

            proof        = [[] for _ in range(depth)]
            branches_out = {}   # (level, key) → hash bytes  (binary nodes only)

            # Level-0 leaf hashes
            cur = {k: hash_leaf(path_at_level(k, 0, depth), v)
                   for k, v in new_items}

            # first_merge_level (fml): minimum level where two batch keys
            # share a parent.  Below fml every key's path is independent.
            fml = depth
            if len(sorted_keys) >= 2:
                for i in range(len(sorted_keys) - 1):
                    bl = (sorted_keys[i] ^ sorted_keys[i + 1]).bit_length()
                    fml = min(fml, bl)

            # Preload all existing leaf keys for fast bisect-based range checks.
            ex_keys = []
            with rtxn.cursor(db=self.leaves_db) as c:
                for k_raw in c.iternext(keys=True, values=False):
                    ex_keys.append(_dec_key(k_raw))

            # Helper: fast non-empty check via bisect, DB read only if needed
            def _sib_hash_fast(level, key):
                lo = key << level
                hi = lo + (1 << level) - 1
                idx = bisect.bisect_left(ex_keys, lo)
                if idx >= len(ex_keys) or ex_keys[idx] > hi:
                    return EMPTY
                return self._get_existing_hash(rtxn, level, key)

            # ------------------------------------------------------------------
            # Phase A: levels 0 … fml-2  (independent per-key)
            #
            # Pass-through means we skip chain levels entirely: h is unchanged
            # at levels with an empty sibling.  We jump directly to each
            # binary branch point (sib_hash is not EMPTY) and compute the
            # new branch hash there.
            # ------------------------------------------------------------------
            start_level = 0
            if fml >= 2:
                cur_at_merge = {}

                for bk in sorted_keys:
                    h = cur[bk]  # leaf hash; propagates through chain levels

                    # Collect all levels in [0, fml-1) where an existing leaf
                    # in bk's subtree creates a non-empty sibling.
                    lo_fml = (bk >> fml) << fml
                    hi_fml = lo_fml + (1 << fml) - 1
                    sib_levels = set()
                    idx = bisect.bisect_left(ex_keys, lo_fml)
                    while idx < len(ex_keys) and ex_keys[idx] <= hi_fml:
                        ek = ex_keys[idx]
                        if ek != bk:
                            sl = (ek ^ bk).bit_length() - 1
                            if sl < fml - 1:
                                sib_levels.add(sl)
                        idx += 1

                    # For each binary branch point (ascending): fetch sibling,
                    # compute branch hash.  Chain levels between them are free.
                    for sl in sorted(sib_levels):
                        sib_key  = (bk >> sl) ^ 1
                        sib_hash = self._get_existing_hash(rtxn, sl, sib_key)
                        if sib_hash is EMPTY:
                            continue    # still a chain level -- h unchanged
                        proof[sl].append((sib_key, sib_hash))
                        pk     = bk >> (sl + 1)
                        p_path = path_at_level(pk, sl + 1, depth)
                        if (bk >> sl) & 1 == 0:
                            h = hash_branch(p_path, h, sib_hash)
                        else:
                            h = hash_branch(p_path, sib_hash, h)
                        branches_out[(sl + 1, pk)] = h
                    # Chain levels from last binary branch to fml-1 are free.
                    cur_at_merge[bk >> (fml - 1)] = h

                cur = cur_at_merge
                start_level = fml - 1

            # ------------------------------------------------------------------
            # Phase B: levels start_level … depth-1  (merging, set-based)
            # ------------------------------------------------------------------
            for level in range(start_level, depth):
                nxt     = {}
                parents = {k >> 1 for k in cur}

                for pk in parents:
                    lk, rk = pk << 1, pk << 1 | 1
                    l_in, r_in = lk in cur, rk in cur

                    lh = cur[lk] if l_in else _sib_hash_fast(level, lk)
                    rh = cur[rk] if r_in else _sib_hash_fast(level, rk)

                    if l_in and not r_in and rh is not EMPTY:
                        proof[level].append((rk, rh))
                    if r_in and not l_in and lh is not EMPTY:
                        proof[level].append((lk, lh))

                    ph = hash_branch(path_at_level(pk, level + 1, depth), lh, rh)
                    nxt[pk] = ph

                    if lh is not EMPTY and rh is not EMPTY:
                        branches_out[(level + 1, pk)] = ph

                cur = nxt

                # Early termination: single path remaining to root.
                # Chain levels above the last binary branch are free (pass-through).
                if len(cur) == 1:
                    only_key, only_hash = next(iter(cur.items()))
                    for rlevel in range(level + 1, depth):
                        sib_key  = only_key ^ 1
                        sib_hash = _sib_hash_fast(rlevel, sib_key)
                        if sib_hash is EMPTY:
                            only_key = only_key >> 1
                            continue    # chain level: pass-through, skip
                        proof[rlevel].append((sib_key, sib_hash))
                        pk = only_key >> 1
                        p_path = path_at_level(pk, rlevel + 1, depth)
                        if only_key & 1 == 0:
                            only_hash = hash_branch(p_path, only_hash, sib_hash)
                        else:
                            only_hash = hash_branch(p_path, sib_hash, only_hash)
                        branches_out[(rlevel + 1, pk)] = only_hash
                        only_key = pk
                    new_root = only_hash
                    break
            else:
                assert len(cur) == 1
                new_root = next(iter(cur.values()))

        # Atomic write
        with self.env.begin(write=True) as wtxn:
            for k, v in new_items:
                wtxn.put(_enc_key(k), v, db=self.leaves_db)
            for (lvl, key), h in branches_out.items():
                wtxn.put(_bkey(lvl, key), h, db=self.branches_db)
            wtxn.put(b'root', new_root, db=self.meta_db)

        # Invalidate cached nodes on paths from new leaves to root
        nc = self._node_cache
        if nc:
            cmin = self._cache_min_level
            for k, _ in new_items:
                for L in range(cmin, depth):
                    nc.pop((L, k >> L), None)

        for lp in proof:
            lp.sort()

        self._batch_proof_cache.append((proof, {k for k, _ in new_items}))
        return new_items, proof

    def get_cached_proof(self, leaf_key):
        """Return a cached batch proof containing leaf_key, or None."""
        for proof, batch_keys in reversed(self._batch_proof_cache):
            if leaf_key in batch_keys:
                return proof
        return None


# ---------------------------------------------------------------------------
# Standalone proof verification  (matches ndsmt2.py)
# ---------------------------------------------------------------------------

def smt_compute_tree_root(proof, batch, depth):
    """
    Recompute SMT root from a leaf batch and proof siblings.
    """
    nodes = []
    for key, data in batch:
        if data is EMPTY:
            nodes.append((key, EMPTY))
        else:
            nodes.append((key, hash_leaf(path_at_level(key, 0, depth), data)))

    for level in range(depth):
        next_nodes = []
        lp = proof[level]
        i, j = 0, 0

        while i < len(nodes):
            k, k_val = nodes[i]
            sibling = k ^ 1
            parent  = k >> 1

            if (k & 1 == 0
                    and i + 1 < len(nodes)
                    and nodes[i + 1][0] == sibling):
                i += 1
                sib_val = nodes[i][1]
            elif j < len(lp) and lp[j][0] == sibling:
                sib_val = lp[j][1]
                j += 1
            else:
                sib_val = EMPTY

            p_path = path_at_level(parent, level + 1, depth)
            if k & 1 == 0:
                p_val = hash_branch(p_path, k_val, sib_val)
            else:
                p_val = hash_branch(p_path, sib_val, k_val)

            next_nodes.append((parent, p_val))
            i += 1

        nodes = next_nodes

    assert len(nodes) == 1, f"Expected 1 root, got {len(nodes)}"
    return nodes[0][1]


def verify_consistency(proof, old_root, new_root, batch, depth):
    """
    Verify a non-deletion (consistency) proof.

    Step 1: Compute root with empty leaves (B_⊥) → must equal old_root.
    Step 2: Compute root with actual leaves (B)   → must equal new_root.
    """
    if not batch:
        return old_root == new_root

    batch_empty = [(key, EMPTY) for key, _ in batch]
    r1 = smt_compute_tree_root(proof, batch_empty, depth)
    if r1 != old_root:
        print(f"Consistency step 1 failed:\n"
              f"  computed: {r1!r}\n  expected: {old_root!r}", file=sys.stderr)
        return False

    r2 = smt_compute_tree_root(proof, batch, depth)
    if r2 != new_root:
        print(f"Consistency step 2 failed:\n"
              f"  computed: {r2!r}\n  expected: {new_root!r}", file=sys.stderr)
        return False

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def db_size_mb(path):
    return sum(os.path.getsize(os.path.join(path, f))
               for f in os.listdir(path)
               if os.path.isfile(os.path.join(path, f))) / 1024**2


# ---------------------------------------------------------------------------
# Demo / correctness tests + performance benchmark
# ---------------------------------------------------------------------------

def main():
    depth = 256
    db_path = '/tmp/smt_lmdb2_test'
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    smt = SparseMerkleTree(db_path, depth=depth)

    # ---- Test 1: small batch with adjacent keys ----
    keys   = [1, 3, 2]
    values = [b'value1', b'value3', b'value2']
    batch  = zip(keys, values)

    old_root = smt.get_root()
    uniq_batch, proof = smt.batch_insert(batch)
    new_root = smt.get_root()
    assert verify_consistency(proof, old_root, new_root, uniq_batch, depth)
    total_leaves = len(keys)

    # ---- Test 2: pre-fill 5000 items, including a duplicate ----
    batch = {}
    for i in range(5000):
        rk = hash("a" + str(i)) % (2 ** depth)
        batch[rk] = f"Val {rk}".encode()
    batch[3] = b"double three"

    print(f"Pre-filling SMT with {len(batch)} items.", file=sys.stderr)
    old_root = new_root
    uniq_batch, proof = smt.batch_insert(batch.items())
    new_root = smt.get_root()
    assert verify_consistency(proof, old_root, new_root, uniq_batch, depth)
    total_leaves += len(uniq_batch)

    # ---- Test 3: proving batch ----
    batch = {}
    for i in range(5000):
        rk = hash("b" + str(i)) % (2 ** depth)
        batch[rk] = f"Val {rk}".encode()

    old_root = new_root
    print(f"Inserting batch of {len(batch)} items.", file=sys.stderr)
    uniq_batch, proof = smt.batch_insert(batch.items())
    new_root = smt.get_root()
    assert verify_consistency(proof, old_root, new_root, uniq_batch, depth)
    total_leaves += len(uniq_batch)

    print("All consistency proofs verified.", file=sys.stderr)

    # ---- Performance benchmark ----
    print("\n--- Performance Benchmark ---", file=sys.stderr)
    for rnd in range(2000):
        batch = []
        for i in range(1_000):
            rk = hash(f"perf_{rnd}_{i}") % (2 ** depth)
            batch.append((rk, f"V{i}".encode()))

        old_root = smt.get_root()
        t0 = time.time()
        new_items, proof = smt.batch_insert(batch)
        dt = time.time() - t0
        new_root = smt.get_root()

        assert verify_consistency(proof, old_root, new_root, new_items, depth)
        total_leaves += len(new_items)
        print(f"  round {rnd:2d}: +{len(new_items):5d} leaves  "
              f"{dt:.3f}s  total={total_leaves:7d}  "
              f"db={db_size_mb(db_path):.1f} MB", file=sys.stderr)

    smt.close()
    shutil.rmtree(db_path)
    print("\nAll tests passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
