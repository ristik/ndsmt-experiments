# Radix Sparse Merkle Tree v2 (RSMT2) with Consistency Proofs
#
# Leaf-anchored topology: each leaf hashes the full 256-bit key, making
# leaves the spatial anchors. Internal nodes hash only their children:
#   H_node(lh, rh) = SHA256(0x01 || lh || rh)
#   H_leaf(key, value) = SHA256(0x00 || key_32B || value)
#
# Because neither leaf nor node hashes depend on the tree's edge structure
# (path / common prefix), inserting a new key only creates new hashes for
# the new leaf and new internal nodes. Pre-existing node hashes are strictly
# read-only -- splitting an edge above them does not change their hash.
#
# Internal nodes always have exactly two children (radix tree invariant:
# nodes are only created at bifurcation points). This avoids the single-
# child vulnerability where H(child) would equal the child's hash.

# Consistency Proof
# -----------------
# Demonstrates that batch B was correctly inserted into tree state ρ_0,
# producing ρ_1, without modifying or deleting existing records.
#
# Proof format (flat opcode stream from pre-order traversal):
#   'S'   Unchanged subtree.   Stream: ['S', hash].
#   'N'   Branch node.         Stream: ['N']. Two children follow (left, right).
#   'L'   New leaf inserted.   Stream: ['L', key].
#
# No BL (border leaf) or BNS (border node shortened) opcodes are needed.
# When an insertion splits an existing edge, the existing subtree's hash is
# unchanged (it includes neither the edge path nor its position), so it
# appears as a simple 'S'.
#
# Verification -- Eval(π, B) --> (h_0, h_1):
#   'S': pop hash --> (hash, hash)
#   'N': eval left --> (lh_0, lh_1), eval right --> (rh_0, rh_1)
#        h_0: both null --> null
#             exactly one null --> pass-through (the non-null hash)
#             both non-null --> H_node(lh_0, rh_0)
#        h_1: H_node(lh_1, rh_1)
#   'L': pop key, look up value from B
#        --> (null, H_leaf(key, value))
#
# Valid iff:
#   (1) π and B are fully consumed without leftovers,
#   (2) (h_0, h_1) == (ρ_0, ρ_1).
#
# Security argument:
# - S hashes originate from the existing tree; any tampering causes ρ_0 mismatch.
# - L keys are matched against the verifier's own batch; values cannot be faked.
# - The N structure must simultaneously reconstruct both ρ_0 and ρ_1 from
#   collision-resistant hashes, binding the topology cryptographically.
# - Pre-existing leaves hash the full key (absolute anchor); their hashes
#   never change regardless of tree restructuring, preventing silent deletion.

# Inclusion Proof
# ---------------
# A presence bitmap (256-bit int) marks the bit positions of branch nodes
# along the root-to-leaf path. Sibling hashes are listed leaf-to-root.
# Verification:
#   current = H_leaf(key, value)
#   for each set bit i in bitmap from highest to lowest:
#       sibling = next sibling hash
#       if key bit i == 0: current = H_node(current, sibling)
#       else:              current = H_node(sibling, current)
#   assert current == root

# Possible optimization we're postponing for now:
# Do not provide explicit inclusion batch B to verifier, instead do
#   'L'   New leaf inserted.   Stream: ['L', key, value].

import hashlib
import sys

# ---------------------------------------------------------------------------
# Hashing -- domain-separated, no CBOR
# ---------------------------------------------------------------------------

KEY_BYTES = 32  # 256-bit keys
EMPTY = None

def hash_leaf(key, value):
    """SHA256(0x00 || key_32B || value) -- position-independent leaf hash."""
    return hashlib.sha256(b'\x00' + key.to_bytes(KEY_BYTES, 'big') + value).digest()

def hash_node(lh, rh):
    """SHA256(0x01 || lh || rh) -- children-only internal hash."""
    return hashlib.sha256(b'\x01' + lh + rh).digest()

# ---------------------------------------------------------------------------
# Path utilities (for tree navigation only, not hashing)
# ---------------------------------------------------------------------------

def path_len(p):
    """Number of payload bits (excluding sentinel)."""
    return p.bit_length() - 1

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class LeafBranch:
    """Leaf node: stores key, value, path (navigation), and cached hash."""
    __slots__ = ['path', 'key', 'value', '_hash']

    def __init__(self, path, key, value):
        self.path  = path
        self.key   = key
        self.value = value
        self._hash = hash_leaf(key, value)  # full key, position-independent

    def get_hash(self):
        return self._hash
    # No rehash needed -- hash does not depend on path.


class NodeBranch:
    """Internal node: path (navigation) + left/right children.
    Always has exactly two children (radix tree invariant)."""
    __slots__ = ['path', 'left', 'right', '_hash']

    def __init__(self, path, left, right):
        self.path  = path
        self.left  = left
        self.right = right
        self._hash = None

    def get_hash(self):
        if self._hash is None:
            self._hash = hash_node(self.left.get_hash(), self.right.get_hash())
        return self._hash

# ---------------------------------------------------------------------------
# Radix Sparse Merkle Tree
# ---------------------------------------------------------------------------

class SparseMerkleTree:
    """
    Radix SMT v2. LSB-first path consumption.
    Append-only (no deletion, no value changes). In-memory.
    """
    def __init__(self, depth=256):
        self.depth = depth
        self.root  = None

    def get_root(self):
        return self.root.get_hash() if self.root else None

    # ------------------------------------------------------------------
    # Public: batch insert with consistency proof
    # ------------------------------------------------------------------

    def batch_insert(self, batch):
        """
        Insert (key, value) pairs. Duplicates / pre-existing keys skipped.
        Returns (items, proof) where items = sorted inserted pairs.
        """
        new_items = {}
        for key, data in batch:
            if key in new_items:
                print(f"Duplicate key {key} in batch, skipping.", file=sys.stderr)
                continue
            if self._find_leaf(key) is not None:
                print(f"Key {key} already exists, skipping.", file=sys.stderr)
                continue
            new_items[key] = data

        if not new_items:
            return [], ['S', None]

        items = sorted(new_items.items())
        proof_out = []
        self.root = self._insert_proof(self.root, items, 0, proof_out)
        return items, proof_out

    # ------------------------------------------------------------------
    # Public: inclusion proof (bitmap + siblings)
    # ------------------------------------------------------------------

    def generate_proof(self, key):
        """Returns (bitmap, siblings) or None if key absent.
        bitmap: 256-bit int with bits set at branch positions.
        siblings: hashes ordered leaf-to-root."""
        return self._collect_proof(self.root, key, 0)

    def _collect_proof(self, node, key, start_bit):
        if node is None:
            return None
        if isinstance(node, LeafBranch):
            if node.key != key:
                return None
            return (0, [])

        n    = path_len(node.path)
        kpfx = (key >> start_bit) & ((1 << n) - 1)
        npfx = node.path & ((1 << n) - 1)
        if kpfx != npfx:
            return None

        bit       = start_bit + n
        direction = (key >> bit) & 1

        if direction:
            result       = self._collect_proof(node.right, key, bit)
            sibling_hash = node.left.get_hash()
        else:
            result       = self._collect_proof(node.left, key, bit)
            sibling_hash = node.right.get_hash()

        if result is None:
            return None

        bitmap, siblings = result
        siblings.append(sibling_hash)
        bitmap |= (1 << bit)
        return (bitmap, siblings)

    # ------------------------------------------------------------------
    # Internal: key lookup
    # ------------------------------------------------------------------

    def _find_leaf(self, key):
        node = self.root
        bit  = 0
        while node is not None:
            if isinstance(node, LeafBranch):
                return node if node.key == key else None
            n    = path_len(node.path)
            kpfx = (key >> bit) & ((1 << n) - 1)
            npfx = node.path    & ((1 << n) - 1)
            if kpfx != npfx:
                return None
            bit += n
            direction = (key >> bit) & 1
            node = node.right if direction else node.left
        return None

    # ------------------------------------------------------------------
    # Internal: path helper
    # ------------------------------------------------------------------

    def _rem(self, key, start_bit):
        return (1 << (self.depth - start_bit)) | (key >> start_bit)

    @staticmethod
    def _first_split(keys, start_bit):
        """First bit >= start_bit where the sorted keys disagree."""
        result = None
        for i in range(len(keys) - 1):
            xor = (keys[i] ^ keys[i + 1]) >> start_bit
            if xor:
                pos = start_bit + (xor & -xor).bit_length() - 1
                if result is None or pos < result:
                    result = pos
        return result

    # ------------------------------------------------------------------
    # Internal: build fresh subtree from batch items
    # frozen: {key: hash} for existing leaves (emitted as 'S')
    # ------------------------------------------------------------------

    def _build_batch_proof(self, batch, start_bit, proof_out, frozen=None):
        if not batch:
            proof_out.extend(['S', None])
            return None

        if len(batch) == 1:
            k, v = batch[0]
            path = self._rem(k, start_bit)
            leaf = LeafBranch(path, k, v)
            if frozen and k in frozen:
                proof_out.extend(['S', frozen[k]])
            else:
                proof_out.extend(['L', k])
            return leaf

        keys  = [k for k, _ in batch]
        split = self._first_split(keys, start_bit)

        n_common = split - start_bit
        cbits    = (keys[0] >> start_bit) & ((1 << n_common) - 1)
        cp       = (1 << n_common) | cbits

        lb = [(k, v) for k, v in batch if not ((k >> split) & 1)]
        rb = [(k, v) for k, v in batch if      (k >> split) & 1 ]

        proof_out.append('N')
        ln = self._build_batch_proof(lb, split, proof_out, frozen)
        rn = self._build_batch_proof(rb, split, proof_out, frozen)
        return NodeBranch(cp, ln, rn)

    # ------------------------------------------------------------------
    # Internal: insert batch into existing subtree with consistency proof
    # ------------------------------------------------------------------

    def _insert_proof(self, node, batch, start_bit, proof_out):
        if not batch:
            proof_out.extend(['S', node.get_hash() if node else None])
            return node

        if node is None:
            return self._build_batch_proof(batch, start_bit, proof_out)

        if isinstance(node, LeafBranch):
            filtered = [(k, v) for k, v in batch if k != node.key]
            if len(filtered) < len(batch):
                print(f"Key {node.key} already exists, skipping.", file=sys.stderr)
            if not filtered:
                proof_out.extend(['S', node.get_hash()])
                return node

            # Merge existing leaf with new items; existing leaf is frozen (hash unchanged)
            all_items = sorted([(node.key, node.value)] + filtered,
                               key=lambda x: x[0])
            frozen = {node.key: node.get_hash()}
            return self._build_batch_proof(all_items, start_bit, proof_out, frozen=frozen)

        # NodeBranch: check how batch aligns with this node's path
        n_path      = path_len(node.path)
        node_prefix = node.path & ((1 << n_path) - 1)

        first_div = n_path
        for k, _ in batch:
            item_pfx = (k >> start_bit) & ((1 << n_path) - 1)
            xor      = item_pfx ^ node_prefix
            if xor:
                low = (xor & -xor).bit_length() - 1
                if low < first_div:
                    first_div = low

        if first_div < n_path:
            return self._node_split_proof(node, batch, start_bit, first_div, proof_out)

        # Path fully matches -- recurse into children
        split       = start_bit + n_path
        batch_left  = [(k, v) for k, v in batch if not ((k >> split) & 1)]
        batch_right = [(k, v) for k, v in batch if      (k >> split) & 1 ]

        proof_out.append('N')
        new_left  = self._insert_proof(node.left,  batch_left,  split, proof_out)
        new_right = self._insert_proof(node.right, batch_right, split, proof_out)

        node.left  = new_left
        node.right = new_right
        node._hash = None
        return node

    def _node_split_proof(self, node, batch, start_bit, first_div, proof_out):
        n_path      = path_len(node.path)
        node_prefix = node.path & ((1 << n_path) - 1)

        n_common    = first_div
        common_bits = node_prefix & ((1 << n_common) - 1)
        new_cp      = (1 << n_common) | common_bits
        new_split   = start_bit + n_common

        old_dir     = (node_prefix >> n_common) & 1

        # Shorten node's path. Hash is unchanged (doesn't include path).
        new_path = node.path >> n_common
        if new_path == 0:
            new_path = 1
        node.path = new_path

        batch_left  = [(k, v) for k, v in batch if not ((k >> new_split) & 1)]
        batch_right = [(k, v) for k, v in batch if      (k >> new_split) & 1 ]

        proof_out.append('N')
        if old_dir == 0:
            new_left  = self._insert_proof(node,  batch_left,  new_split, proof_out)
            new_right = self._insert_proof(None,  batch_right, new_split, proof_out)
        else:
            new_left  = self._insert_proof(None,  batch_left,  new_split, proof_out)
            new_right = self._insert_proof(node,  batch_right, new_split, proof_out)

        return NodeBranch(new_cp, new_left, new_right)


# ---------------------------------------------------------------------------
# Proof evaluation
# ---------------------------------------------------------------------------

def synchronized_proof_eval(proof_iterator, batch_dict):
    """Computes (h_0, h_1) -- pre- and post-insertion hashes, in one pass."""
    try:
        tag = next(proof_iterator)
    except StopIteration:
        return None, None

    if tag == 'S':
        h = next(proof_iterator)
        return h, h

    if tag == 'N':
        lh0, lh1 = synchronized_proof_eval(proof_iterator, batch_dict)
        rh0, rh1 = synchronized_proof_eval(proof_iterator, batch_dict)

        # h_0: pass-through for pre-insertion state
        if lh0 is None and rh0 is None:
            h0 = None
        elif lh0 is None:
            h0 = rh0
        elif rh0 is None:
            h0 = lh0
        else:
            h0 = hash_node(lh0, rh0)

        # h_1: always combine both children
        h1 = hash_node(lh1, rh1)
        return h0, h1

    if tag == 'L':
        k = next(proof_iterator)
        v = batch_dict.pop(k, None)
        if v is None:
            raise ValueError(f"Proof requires key {k} not found in batch")
        return None, hash_leaf(k, v)

    raise ValueError(f"Unknown tag: {tag}")


# ---------------------------------------------------------------------------
# Consistency (non-deletion) proof verification
# ---------------------------------------------------------------------------

def verify_consistency(proof, old_root, new_root, batch, _=None):
    if not batch:
        return old_root == new_root

    batch_dict = {k: v for k, v in batch}
    proof_iter = iter(proof)

    try:
        r0, r1 = synchronized_proof_eval(proof_iter, batch_dict)
    except Exception as e:
        print(f"Consistency verification failed: {e}", file=sys.stderr)
        return False

    try:
        next(proof_iter)
        print("Proof not fully consumed.", file=sys.stderr)
        return False
    except StopIteration:
        pass

    if batch_dict:
        print(f"{len(batch_dict)} batch elements not consumed.", file=sys.stderr)
        return False

    if r0 != old_root:
        print(f"r0 mismatch:\n  computed: {r0.hex() if r0 else None}\n  expected: {old_root.hex() if old_root else None}", file=sys.stderr)
        return False

    if r1 != new_root:
        print(f"r1 mismatch:\n  computed: {r1.hex() if r1 else None}\n  expected: {new_root.hex() if new_root else None}", file=sys.stderr)
        return False

    return True


# ---------------------------------------------------------------------------
# Inclusion proof verification
# ---------------------------------------------------------------------------

def verify_proof(key, value, bitmap, siblings, root, depth):
    """Verify inclusion proof: bitmap marks branch positions, siblings leaf-to-root."""
    current = hash_leaf(key, value)

    # Extract set bit positions from bitmap, process deepest (highest) first
    bits = []
    b = bitmap
    while b:
        pos = (b & -b).bit_length() - 1
        bits.append(pos)
        b &= b - 1
    bits.sort(reverse=True)

    if len(bits) != len(siblings):
        return False

    for idx, i in enumerate(bits):
        sibling = siblings[idx]
        if (key >> i) & 1:
            current = hash_node(sibling, current)
        else:
            current = hash_node(current, sibling)

    return current == root


# ---------------------------------------------------------------------------
# Demo / test
# ---------------------------------------------------------------------------

def main():
    import time

    depth = 256
    smt   = SparseMerkleTree(depth)

    def check_batch(label, smt, batch):
        old_root = smt.get_root()
        t0       = time.perf_counter()
        items, proof = smt.batch_insert(batch)
        dt       = time.perf_counter() - t0
        new_root = smt.get_root()

        assert verify_consistency(proof, old_root, new_root, items), \
            f"Consistency proof failed for {label}"

        # Spot-check individual inclusion proofs
        sample = items[:5] + items[-5:]
        for k, v in sample:
            result = smt.generate_proof(k)
            assert result is not None, f"key {k} not found"
            bitmap, siblings = result
            assert verify_proof(k, v, bitmap, siblings, new_root, depth), \
                f"inclusion proof failed for key {k}"

        print(f"{label}: {len(items)} inserted in {dt:.3f}s, "
              f"root={new_root.hex()[:16] if new_root else 'None'}…, "
              f"consistency+inclusion OK", file=sys.stderr)
        return items

    # --- small batch ---
    check_batch("Small batch", smt, [(1, b'v1'), (3, b'v3'), (2, b'v2')])

    # --- duplicate rejection ---
    smt.batch_insert([(1, b'dup1'), (99, b'new99')])
    assert smt._find_leaf(1)  is not None
    assert smt._find_leaf(99) is not None

    # --- large pre-fill ---
    batch = {}
    for i in range(5000):
        rk = hash("a" + str(i)) % (2 ** depth)
        batch[rk] = f"Val {rk}".encode()
    batch[3] = b"dup three"
    check_batch("Pre-fill", smt, batch.items())

    # --- second large batch ---
    batch2 = {hash("b" + str(i)) % (2 ** depth): f"Val2 {i}".encode()
              for i in range(5000)}
    check_batch("Second batch", smt, batch2.items())

    print("All consistency and inclusion proofs verified.", file=sys.stderr)


if __name__ == "__main__":
    main()
