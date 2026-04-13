# Radix Sparse Merkle Tree v3 - simplifying the consistency proof ops
#
# Consistency Proof Optimizations:
# - Synchronized processing of batch items and proof elements: Keys are not
#   transmitted in the proof stream. The verifier
#   routes the known batch locally using the 'depth' provided by 'N' opcodes.
# - Subtree 'B'uilding: Purely new subtrees are summarized by a single 'B' opcode.
#   The verifier locally recreates the subtree hash directly from the batch partition.

import hashlib
import sys

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

KEY_BYTES = 32


def hash_leaf(key, value):
    return hashlib.sha256(b"\x00" + key.to_bytes(KEY_BYTES, "big") + value).digest()


def hash_node(lh, rh, depth):
    return hashlib.sha256(b"\x01" + depth.to_bytes(2, "big") + lh + rh).digest()


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def path_len(p):
    return p.bit_length() - 1


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class LeafBranch:
    __slots__ = ["path", "key", "value", "_hash"]

    def __init__(self, path, key, value):
        self.path = path
        self.key = key
        self.value = value
        self._hash = hash_leaf(key, value)

    def get_hash(self):
        return self._hash


class NodeBranch:
    __slots__ = ["path", "left", "right", "depth", "_hash"]

    def __init__(self, path, left, right, depth):
        self.path = path
        self.left = left
        self.right = right
        self.depth = depth
        self._hash = None

    def get_hash(self):
        if self._hash is None:
            self._hash = hash_node(
                self.left.get_hash(), self.right.get_hash(), self.depth
            )
        return self._hash


# ---------------------------------------------------------------------------
# Radix Sparse Merkle Tree
# ---------------------------------------------------------------------------


class SparseMerkleTree:
    def __init__(self, depth=256):
        self.depth = depth
        self.root = None

    def get_root(self):
        return self.root.get_hash() if self.root else None

    def batch_insert(self, batch):
        new_items = {}
        for key, data in batch:
            if key in new_items or self._find_leaf(key) is not None:
                continue
            new_items[key] = data

        if not new_items:
            return [], []

        items = sorted(new_items.items())
        proof_out = []
        self.root = self._insert_proof(self.root, items, 0, proof_out)
        return items, proof_out

    # ------------------------------------------------------------------
    # Proof Generation Helper Math
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
    # Tree Building & Proof Emission
    # ------------------------------------------------------------------

    def _build_pure_subtree(self, batch, start_bit):
        """Builds a purely new subtree natively (no proofs emitted)."""
        if len(batch) == 1:
            k, v = batch[0]
            return LeafBranch(self._rem(k, start_bit), k, v)

        keys = [k for k, _ in batch]
        split = self._first_split(keys, start_bit)

        n_common = split - start_bit
        cbits = (keys[0] >> start_bit) & ((1 << n_common) - 1)
        cp = (1 << n_common) | cbits

        lb = [(k, v) for k, v in batch if not ((k >> split) & 1)]
        rb = [(k, v) for k, v in batch if (k >> split) & 1]

        return NodeBranch(
            cp,
            self._build_pure_subtree(lb, split),
            self._build_pure_subtree(rb, split),
            split,
        )

    def _build_mixed_subtree(self, batch, start_bit, proof_out, frozen):
        """Builds a subtree that mixes an existing 'frozen' leaf with new batch items."""
        # If no frozen items route here, it's a pure new branch! Emit 'B'.
        if not any(k in frozen for k, _ in batch):
            proof_out.append("B")
            return self._build_pure_subtree(batch, start_bit)

        # If it's a single item and frozen is present, it MUST be the frozen leaf.
        if len(batch) == 1:
            k, v = batch[0]
            proof_out.extend(["S", frozen[k]])
            return LeafBranch(self._rem(k, start_bit), k, v)

        keys = [k for k, _ in batch]
        split = self._first_split(keys, start_bit)

        n_common = split - start_bit
        cbits = (keys[0] >> start_bit) & ((1 << n_common) - 1)
        cp = (1 << n_common) | cbits

        lb = [(k, v) for k, v in batch if not ((k >> split) & 1)]
        rb = [(k, v) for k, v in batch if (k >> split) & 1]

        proof_out.extend(["N", split])
        ln = self._build_mixed_subtree(lb, split, proof_out, frozen)
        rn = self._build_mixed_subtree(rb, split, proof_out, frozen)
        return NodeBranch(cp, ln, rn, split)

    def _insert_proof(self, node, batch, start_bit, proof_out):
        if not batch:
            proof_out.extend(["S", node.get_hash() if node else None])
            return node

        if node is None:
            proof_out.append("B")
            return self._build_pure_subtree(batch, start_bit)

        if isinstance(node, LeafBranch):
            # Existing leaf split by new batch items. Mix them and emit proof.
            all_items = sorted([(node.key, node.value)] + batch, key=lambda x: x[0])
            frozen = {node.key: node.get_hash()}
            return self._build_mixed_subtree(all_items, start_bit, proof_out, frozen)

        # NodeBranch logic
        n_path = path_len(node.path)
        node_prefix = node.path & ((1 << n_path) - 1)

        first_div = n_path
        for k, _ in batch:
            item_pfx = (k >> start_bit) & ((1 << n_path) - 1)
            xor = item_pfx ^ node_prefix
            if xor:
                low = (xor & -xor).bit_length() - 1
                if low < first_div:
                    first_div = low

        if first_div < n_path:
            return self._node_split_proof(node, batch, start_bit, first_div, proof_out)

        split = start_bit + n_path
        batch_left = [(k, v) for k, v in batch if not ((k >> split) & 1)]
        batch_right = [(k, v) for k, v in batch if (k >> split) & 1]

        proof_out.extend(["N", split])
        new_left = self._insert_proof(node.left, batch_left, split, proof_out)
        new_right = self._insert_proof(node.right, batch_right, split, proof_out)

        node.left = new_left
        node.right = new_right
        node._hash = None
        return node

    def _node_split_proof(self, node, batch, start_bit, first_div, proof_out):
        n_path = path_len(node.path)
        node_prefix = node.path & ((1 << n_path) - 1)

        n_common = first_div
        common_bits = node_prefix & ((1 << n_common) - 1)
        new_cp = (1 << n_common) | common_bits
        new_split = start_bit + n_common
        old_dir = (node_prefix >> n_common) & 1

        new_path = node.path >> n_common
        if new_path == 0:
            new_path = 1
        node.path = new_path

        batch_left = [(k, v) for k, v in batch if not ((k >> new_split) & 1)]
        batch_right = [(k, v) for k, v in batch if (k >> new_split) & 1]

        proof_out.extend(["N", new_split])
        if old_dir == 0:
            new_left = self._insert_proof(node, batch_left, new_split, proof_out)
            new_right = self._insert_proof(None, batch_right, new_split, proof_out)
        else:
            new_left = self._insert_proof(None, batch_left, new_split, proof_out)
            new_right = self._insert_proof(node, batch_right, new_split, proof_out)

        return NodeBranch(new_cp, new_left, new_right, new_split)

    def _find_leaf(self, key):
        # lookup helper - identical to previous
        node = self.root
        bit = 0
        while node is not None:
            if isinstance(node, LeafBranch):
                return node if node.key == key else None
            n = path_len(node.path)
            kpfx = (key >> bit) & ((1 << n) - 1)
            npfx = node.path & ((1 << n) - 1)
            if kpfx != npfx:
                return None
            bit += n
            node = node.right if ((key >> bit) & 1) else node.left
        return None


# ---------------------------------------------------------------------------
# Verifier Functions
# ---------------------------------------------------------------------------


def _build_pure_hashes(batch, start_bit):
    """Local verifier reconstruction for the 'B' opcode."""
    if len(batch) == 1:
        return hash_leaf(batch[0][0], batch[0][1])

    keys = [k for k, _ in batch]
    split = SparseMerkleTree._first_split(keys, start_bit)

    lb = [(k, v) for k, v in batch if not ((k >> split) & 1)]
    rb = [(k, v) for k, v in batch if (k >> split) & 1]

    return hash_node(
        _build_pure_hashes(lb, split), _build_pure_hashes(rb, split), split
    )


def synchronized_proof_eval(proof_iterator, batch, start_bit=0):
    try:
        tag = next(proof_iterator)
    except StopIteration:
        return None, None

    if tag == "S":
        # Critical verification check: An 'S' branch is unmodified tree state.
        # If any new batch elements routed here, it means the proof omitted their insertion!
        if batch:
            raise ValueError("Proof assigned batch items to an unmodified 'S' node!")
        h = next(proof_iterator)
        return h, h

    if tag == "B":
        if not batch:
            raise ValueError(
                "Proof requested pure build 'B', but no items routed here."
            )
        h1 = _build_pure_hashes(batch, start_bit)
        return None, h1

    if tag == "N":
        depth = next(proof_iterator)

        # Partition batch on the fly
        batch_left = [(k, v) for k, v in batch if not ((k >> depth) & 1)]
        batch_right = [(k, v) for k, v in batch if (k >> depth) & 1]

        lh0, lh1 = synchronized_proof_eval(proof_iterator, batch_left, depth)
        rh0, rh1 = synchronized_proof_eval(proof_iterator, batch_right, depth)

        # h_0 (Pre-state hash resolution)
        if lh0 is None and rh0 is None:
            h0 = None
        elif lh0 is None:
            h0 = rh0
        elif rh0 is None:
            h0 = lh0
        else:
            h0 = hash_node(lh0, rh0, depth)

        # h_1 (Post-state hash resolution)
        h1 = hash_node(lh1, rh1, depth)
        return h0, h1

    raise ValueError(f"Unknown tag: {tag}")


def verify_consistency(proof, old_root, new_root, batch, _=None):
    if not batch:
        return old_root == new_root

    proof_iter = iter(proof)
    try:
        r0, r1 = synchronized_proof_eval(proof_iter, batch, 0)
    except Exception as e:
        print(f"Consistency verification failed: {e}", file=sys.stderr)
        return False

    try:
        next(proof_iter)
        print("Proof not fully consumed.", file=sys.stderr)
        return False
    except StopIteration:
        pass

    if r0 != old_root:
        print("r0 mismatch", file=sys.stderr)
        return False
    if r1 != new_root:
        print("r1 mismatch", file=sys.stderr)
        return False

    return True
