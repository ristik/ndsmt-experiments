# Not path-compressed Sparse Merkle Tree (SMT) with non-deletion (consistency) proofs.
# Here:
#    node_hash(l, r) := ⊥ if l == ⊥ AND r == ⊥ else H(l, r)
#
# bad:  one hashing step per tree layer
#       (O(depth) instead of O(log(capacity)) leaf add complexity)
# good: all keys are directly encoded as hash path shape.
#
# Naive one but included for completeness

import hashlib
import sys
import json

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
# SMT node hashing
# ---------------------------------------------------------------------------

EMPTY = None  # ⊥ — empty node / absent value

def path_at_level(key, level, depth):
    """
    Path segment for node (level, key) in the path-compressed SMT.

    Each edge carries a bit-string label. We encode bit-strings as integers
    with a sentinel high bit:

        path = key | (1 << (depth - level))

    The sentinel delimits the variable-length bit-string:
      Root  (level = depth):  path = 1             (empty ε, sentinel only)
      Leaf  (level = 0):      path = key | 2^depth (full key + sentinel)

    'key' is the node's positional index at this level:
      leaf_key >> level  (the upper bits of the original leaf key).
    """
    return key | (1 << (depth - level))


def hash_leaf(path_segment, data):
    """smt_leaf_hash(p, d) = SHA-256(CBOR([p, d]))"""
    return hashlib.sha256(cbor_encode([path_segment, data])).digest()


def hash_branch(path_segment, h_left, h_right):
    """
    smt_branch_hash(p, h_L, h_R) = SHA-256(CBOR([p, h_L, h_R]))
    Returns EMPTY when **both** children are empty.
    """
    if h_left is EMPTY and h_right is EMPTY:
        return EMPTY
    return hashlib.sha256(cbor_encode([path_segment, h_left, h_right])).digest()


# ---------------------------------------------------------------------------
# Sparse Merkle Tree (in-memory, dictionary-backed)
# ---------------------------------------------------------------------------

class SparseMerkleTree:
    def __init__(self, depth=256):
        self.depth = depth
        self.nodes = {}  # (level, key) → hash bytes;  absent = EMPTY

    def get_root(self):
        return self.nodes.get((self.depth, 0), EMPTY)

    def _get(self, level, key):
        return self.nodes.get((level, key), EMPTY)

    def _set(self, level, key, value):
        if value is EMPTY:
            self.nodes.pop((level, key), None)
        else:
            self.nodes[(level, key)] = value

    def batch_insert(self, batch):
        """
        Insert a sorted batch of (key, data_bytes) pairs and return a
        consistency proof enabling verification of the root transition.

        The proof is a list of length `depth`, where proof[level] contains
        (sibling_key, sibling_hash) pairs — the siblings NOT computable
        from the batch alone.

        Algorithm (bottom-up, one level at a time):
          1. Hash and store all new leaves at level 0.
          2. Maintain a sorted list of affected keys at the current level.
          3. At each level, walk the sorted list pairing siblings:
             - If both siblings affected: no proof entry needed.
             - Otherwise: record the existing sibling hash in the proof.
          4. Compute and store parent hashes; advance to next level.
          5. Early-terminate when only one affected key remains.

        Complexity: O(b·d) worst-case where b = |batch|, d = depth.
        In practice O(b·log(b) + d) because paths merge.
        """
        depth = self.depth

        # Filter already-existing keys; same filtered set must be used for validation
        # Alternative would be failing the entire batch
        new_items_dict = {}   # dict filters duplicate keys in batch
        for key, data in batch:
            if (0, key) in self.nodes:
                print(f"Leaf {key} already set, skipping.", file=sys.stderr)
            else:
                new_items_dict[key] = data

        proof = [[] for _ in range(depth)]
        if not new_items_dict:
            return ([], proof)

        new_items = sorted(new_items_dict.items())
        affected = set()
        # Level 0: compute and store leaf hashes
        for key, data in new_items:
            self._set(0, key, hash_leaf(path_at_level(key, 0, depth), data))
            affected.add(key)

        # Bottom-up propagation
        for level in range(depth):
            parents = set()

            for k in affected:
                sibling = k ^ 1

                # If sibling is not in the affected set, we need its hash for the proof
                if sibling not in affected:
                    sib_hash = self._get(level, sibling)
                    if sib_hash is not EMPTY:
                        proof[level].append((sibling, sib_hash))

                parents.add(k >> 1)

            # Recompute parent hashes
            for p in parents:
                h_l = self._get(level, p << 1)
                h_r = self._get(level, p << 1 | 1)
                self._set(level + 1, p,
                          hash_branch(path_at_level(p, level + 1, depth),
                                      h_l, h_r))

            affected = parents

        # Sort each proof level for deterministic output
        for lp in proof:
            lp.sort()

        return (new_items, proof)


# ---------------------------------------------------------------------------
# Standalone proof verification (no tree state needed)
# ---------------------------------------------------------------------------

def smt_compute_tree_root(proof, batch, depth):
    """
    Recompute SMT root from a leaf batch and proof siblings.
    Corresponds to smt_compute_tree_root in the specification.

    Processes level by level: at each level, pairs batch nodes with
    siblings (from the batch itself, from the proof, or EMPTY)
    to compute parent hashes.
    """
    # Leaf layer
    nodes = []
    for key, data in batch:
        if data is EMPTY:
            nodes.append((key, EMPTY))
        else:
            nodes.append((key, hash_leaf(path_at_level(key, 0, depth), data)))

    # Level by level to root
    for level in range(depth):
        next_nodes = []
        lp = proof[level]
        i, j = 0, 0

        while i < len(nodes):
            k, k_val = nodes[i]
            sibling = k ^ 1
            parent = k >> 1

            # Find sibling hash: batch > proof > empty
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

    assert len(nodes) == 1, f"Expected 1 root node, got {len(nodes)}"
    return nodes[0][1]


def verify_consistency(proof, old_root, new_root, batch, depth):
    """
    Verify a non-deletion (consistency) proof.

    Step 1: Compute root with empty leaves (B_⊥) → must equal old_root.
            This proves all inserted positions were previously empty.
    Step 2: Compute root with actual leaves (B) → must equal new_root.
            This proves new leaves are correctly placed and nothing else changed.
    At both steps, exactly the same proof must be used.
    """
    if not batch:
        return old_root == new_root

    # Step 1: positions were empty before insertion
    batch_empty = [(key, EMPTY) for key, _ in batch]
    r1 = smt_compute_tree_root(proof, batch_empty, depth)
    if r1 != old_root:
        print(f"Consistency step 1 failed:\n"
              f"  computed: {r1!r}\n  expected: {old_root!r}", file=sys.stderr)
        return False

    # Step 2: new state matches claimed root
    r2 = smt_compute_tree_root(proof, batch, depth)
    if r2 != new_root:
        print(f"Consistency step 2 failed:\n"
              f"  computed: {r2!r}\n  expected: {new_root!r}", file=sys.stderr)
        return False

    return True


# ---------------------------------------------------------------------------
# Demo / test / quick witness gen
# ---------------------------------------------------------------------------

def main():
    depth = 256
    smt = SparseMerkleTree(depth)

    # --- test: small batch with adjacent keys ---
    keys = [1, 3, 2]
    values = [b'value1', b'value3', b'value2']
    batch = zip(keys, values)

    old_root = smt.get_root()
    uniq_batch, proof = smt.batch_insert(batch)
    new_root = smt.get_root()
    assert verify_consistency(proof, old_root, new_root, uniq_batch, depth)

    # --- pre-fill tree ---
    batch = {}
    for i in range(5000):
        rk = hash("a" + str(i)) % (2 ** depth)
        batch[rk] = f"Val {rk}".encode()
    batch[3] = b"double three"  # and a duplicate

    print(f"Pre-filling SMT with {len(batch)} items.", file=sys.stderr)
    old_root = new_root
    uniq_batch, proof = smt.batch_insert(batch.items())
    new_root = smt.get_root()
    assert verify_consistency(proof, old_root, new_root, uniq_batch, depth)

    # --- proving batch ---
    batch = {}
    for i in range(5000):
        rk = hash("b" + str(i)) % (2 ** depth)
        batch[rk] = f"Val {rk}".encode()

    old_root = new_root
    print(f"Inserting batch of {len(batch)} items.", file=sys.stderr)
    uniq_batch, proof = smt.batch_insert(batch.items())
    new_root = smt.get_root()
    assert verify_consistency(proof, old_root, new_root, uniq_batch, depth)

    print("All consistency proofs verified.", file=sys.stderr)

    # --- JSON witness output ---
    def hexify(v):
        if v is None:
            return None
        if isinstance(v, bytes):
            return v.hex()
        return v

    witness = {
        "old_root": hexify(old_root),
        "new_root": hexify(new_root),
        "batch": [[k, hexify(v)] for k, v in uniq_batch],
        "proof": [[[k, hexify(v)] for k, v in lp] for lp in proof],
        "depth": depth,
    }
    # print(json.dumps(witness, indent=4))


if __name__ == "__main__":
    main()
