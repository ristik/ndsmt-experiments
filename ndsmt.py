# Path-compressed Sparse Merkle Tree (SMT) with non-deletion (consistency) proofs.
# Tree format matches the paper "Unicity Infrastructure - the Aggregation Layer
# technical report" , https://github.com/unicitynetwork/aggr-layer-paper/
#
# Hash encoding uses CBOR arrays for unambiguous domain separation:
#   Leaf:   H = SHA-256(CBOR([path, data]))
#   Branch: H = SHA-256(CBOR([path, h_left, h_right]))
# where path = key | (1 << (depth - level)) is a sentinel-encoded bit-string
# identifying the node's position in the tree.

import hashlib
import json
import sys

# ---------------------------------------------------------------------------
# Minimal CBOR encoder (handles None, unsigned int, bytes, list)
# ---------------------------------------------------------------------------


def cbor_encode(value):
    """Encode a value as CBOR bytes. Supports None, int >= 0, bytes, list."""
    if value is None:
        return b"\xf6"  # CBOR null
    if isinstance(value, int):
        if value < 0:
            raise ValueError("negative integers not supported")
        if value < 2**64:
            return _cbor_head(0, value)  # major 0: uint
        # Tag 2: positive bignum as byte string
        n = (value.bit_length() + 7) // 8
        return b"\xc2" + cbor_encode(value.to_bytes(n, "big"))
    if isinstance(value, (bytes, bytearray)):
        return _cbor_head(2, len(value)) + value  # major 2: bstr
    if isinstance(value, (list, tuple)):
        body = b"".join(cbor_encode(v) for v in value)
        return _cbor_head(4, len(value)) + body  # major 4: array
    raise TypeError(f"cbor_encode: unsupported {type(value)}")


def _cbor_head(major, n):
    m = major << 5
    if n < 24:
        return bytes([m | n])
    if n < 0x100:
        return bytes([m | 24, n])
    if n < 0x10000:
        return bytes([m | 25]) + n.to_bytes(2, "big")
    if n < 2**32:
        return bytes([m | 26]) + n.to_bytes(4, "big")
    return bytes([m | 27]) + n.to_bytes(8, "big")


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
    smt_branch_hash(p, h_L, h_R)
    Implements pass-through (branch compression) for empty subtrees.
    """
    if h_left is EMPTY:
        return h_right
    if h_right is EMPTY:
        return h_left
    return hashlib.sha256(cbor_encode([path_segment, h_left, h_right])).digest()


class Node:
    """Memory-efficient internal node for the compressed SMT."""

    __slots__ = ["level", "key", "hash", "left", "right"]

    def __init__(self, level, key, hash_val, left=None, right=None):
        self.level = level
        self.key = key
        self.hash = hash_val
        self.left = left
        self.right = right


class SparseMerkleTree:
    def __init__(self, depth=256):
        self.depth = depth
        self.root = None  # Points to the root Node

    def get_root(self):
        return self.root.hash if self.root else EMPTY

    def _get_leaf(self, key):
        """O(log N) top-down search for an existing leaf."""
        curr = self.root
        while curr is not None:
            if curr.level == 0:
                return curr.hash if curr.key == key else EMPTY

            # If the key doesn't share the prefix of this compressed node, it's not here
            if (key >> curr.level) != curr.key:
                return EMPTY

            # Check the bit at the level just below this node
            bit = (key >> (curr.level - 1)) & 1
            curr = curr.right if bit else curr.left

        return EMPTY

    def batch_insert(self, batch):
        """
        Top-down recursive batch insertion.
        Time Complexity: O(b * d) recursion steps, but only O(b * log N) hashes!
        """
        # 1. Deduplicate and filter existing keys
        new_items_dict = {}
        for key, data in batch:
            if self._get_leaf(key) is not EMPTY:
                print(f"Leaf {key} already set, skipping.", file=sys.stderr)
            else:
                new_items_dict[key] = data

        proof = [[] for _ in range(self.depth)]
        if not new_items_dict:
            return ([], proof)

        new_items = sorted(new_items_dict.items())

        # Precompute leaf hashes
        batch_leaves = [
            (k, hash_leaf(path_at_level(k, 0, self.depth), d)) for k, d in new_items
        ]

        # 2. Insert recursively
        self.root = self._insert(self.root, batch_leaves, self.depth, proof)

        # Sort each proof level for deterministic output
        for lp in proof:
            lp.sort()

        return (new_items, proof)

    def _insert(self, node, batch, level, proof):
        """
        Recursively pushes a batch of leaves down the tree.
        Creates branches ONLY when paths diverge.
        """
        if not batch:
            return node

        # Fast-forward: Empty subtree, just drop the leaf here.
        if node is None and len(batch) == 1:
            k, h = batch[0]
            return Node(0, k, h)

        # Reached the absolute bottom
        if level == 0:
            k, h = batch[0]
            return Node(0, k, h)

        # Split batch based on the bit at (level - 1)
        left_batch, right_batch = [], []
        for item in batch:
            if (item[0] >> (level - 1)) & 1:
                right_batch.append(item)
            else:
                left_batch.append(item)

        # Determine where the existing node belongs
        left_node, right_node = None, None
        if node is not None:
            if node.level == level:
                left_node = node.left
                right_node = node.right
            else:
                # Node is compressed (skips this level). Route it to the correct side.
                node_bit = (node.key >> (level - 1 - node.level)) & 1
                if node_bit == 1:
                    right_node = node
                else:
                    left_node = node

        # Identify the common prefix for keys at this level
        prefix = batch[0][0] >> level if batch else node.key >> (level - node.level)
        left_key = (prefix << 1) | 0
        right_key = (prefix << 1) | 1

        # Process Left Side
        if not left_batch and left_node:
            # Batch only went right. The left node is untouched. Add it to proof!
            proof[level - 1].append((left_key, left_node.hash))
            new_left = left_node
        else:
            new_left = self._insert(left_node, left_batch, level - 1, proof)

        # Process Right Side
        if not right_batch and right_node:
            # Batch only went left. The right node is untouched. Add it to proof!
            proof[level - 1].append((right_key, right_node.hash))
            new_right = right_node
        else:
            new_right = self._insert(right_node, right_batch, level - 1, proof)

        # --- BRANCH COMPRESSION (PASS-THROUGH) ---
        if new_left is None and new_right is None:
            return None
        if new_left is None:
            return new_right  # Skip creating a branch!
        if new_right is None:
            return new_left  # Skip creating a branch!

        # Both children exist, so we MUST create a cryptographic branch here
        h = hash_branch(
            path_at_level(prefix, level, self.depth), new_left.hash, new_right.hash
        )
        return Node(level, prefix, h, new_left, new_right)


# ---------------------------------------------------------------------------
# Standalone proof verification
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
            if k & 1 == 0 and i + 1 < len(nodes) and nodes[i + 1][0] == sibling:
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
        print(
            f"Consistency step 1 failed:\n  computed: {r1!r}\n  expected: {old_root!r}",
            file=sys.stderr,
        )
        return False

    # Step 2: new state matches claimed root
    r2 = smt_compute_tree_root(proof, batch, depth)
    if r2 != new_root:
        print(
            f"Consistency step 2 failed:\n  computed: {r2!r}\n  expected: {new_root!r}",
            file=sys.stderr,
        )
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
    values = [b"value1", b"value3", b"value2"]
    batch = zip(keys, values)

    old_root = smt.get_root()
    uniq_batch, proof = smt.batch_insert(batch)
    new_root = smt.get_root()
    assert verify_consistency(proof, old_root, new_root, uniq_batch, depth)

    # --- pre-fill tree ---
    batch = {}
    for i in range(5000):
        rk = hash("a" + str(i)) % (2**depth)
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
        rk = hash("b" + str(i)) % (2**depth)
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
