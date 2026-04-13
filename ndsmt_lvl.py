# ===========================================================================
# Optimized Path-Compressed Full Sparse Merkle Tree
# ===========================================================================
#
# This is a classical full SMT with implicit blank leaves and path compression.
# The committed tree semantics are still the fixed full binary tree over the
# entire key space; path compression is only an implementation optimization.
# Because the tree shape is fixed by key positions and blank leaves are
# implicit, canonicality comes for free: there is no separate prover-chosen
# radix topology to authenticate.
#
# Hash encoding uses fixed-width binary frames for comparability with the other
# prototypes in this repository:
#   Leaf:   H = SHA-256(0x00 || path || data)
#   Branch: H = SHA-256(0x01 || path || h_left || h_right)
#
# `path` is the sentinel-encoded node position, serialized to
# ceil((depth + 1) / 8) bytes:
#   path = key | (1 << (depth - level))
#
# The sentinel makes the encoded path self-delimiting with respect to the tree
# level. Domain separators 0x00 / 0x01 separate leaves and branches.
#
# Branch compression / implicit blanks:
#   merge_branch(path, left, right) returns:
#     - right, if left  is EMPTY
#     - left,  if right is EMPTY
#     - H_branch(path, left, right), otherwise
#
# ---------------------------------------------------------------------------
# Consistency Proof Format
# ---------------------------------------------------------------------------
# A consistency proof is a levelized array of untouched sibling subtrees:
#
#   proof = [level_0, level_1, ..., level_{depth-1}]
#
# where each `level_i` is a sorted list of pairs:
#
#   (node_key, subtree_hash)
#
# Semantics:
# - `i = 0` is the leaf level, `i = depth - 1` is the top branch level.
# - `node_key` is the positional index of that node at level `i`
#   (equivalently, original_leaf_key >> i).
# - `subtree_hash` is the root hash of the untouched sibling subtree at that
#   position. Untouched empty subtrees are implicit and omitted.
#
# Verification recomputes both roots in one pass:
# - old root: every batch position is treated as EMPTY
# - new root: every batch position is treated as the supplied leaf value
#
# The same untouched sibling proof is used for both passes.
#
# ---------------------------------------------------------------------------
# Inclusion Proof Format
# ---------------------------------------------------------------------------
# An inclusion proof is a leaf-to-root list over non-empty sibling levels only:
#
#   cert = [(level, sibling_hash), ...]
#
# with strictly increasing `level`.
#
# Missing levels imply an EMPTY sibling and therefore pass-through compression.
# The verifier reconstructs the root from (key, value), the proof list, and the
# known tree depth.

import hashlib
from bisect import bisect_left

EMPTY = None


def _path_size_bytes(depth):
    return (depth + 8) // 8


def path_at_level(key, level, depth):
    """
    Return the fixed-width byte encoding of the sentinel-encoded path segment
    for node (level, key) in the compressed full SMT.
    """
    path = key | (1 << (depth - level))
    return path.to_bytes(_path_size_bytes(depth), "big")


def hash_leaf(path_segment, data):
    return hashlib.sha256(b"\x00" + path_segment + data).digest()


def hash_branch(path_segment, h_left, h_right):
    return hashlib.sha256(b"\x01" + path_segment + h_left + h_right).digest()


def merge_branch(path_segment, h_left, h_right):
    if h_left is EMPTY:
        return h_right
    if h_right is EMPTY:
        return h_left
    return hash_branch(path_segment, h_left, h_right)


def _find_key_index(batch, key):
    lo, hi = 0, len(batch)
    while lo < hi:
        mid = (lo + hi) >> 1
        if batch[mid][0] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(batch) and batch[lo][0] == key:
        return lo
    return None


def _normalize_batch(batch, depth):
    try:
        items = list(batch)
    except TypeError:
        return None

    if not items:
        return []

    try:
        items.sort(key=lambda x: x[0])
    except Exception:
        return None

    prev_key = None
    upper = 1 << depth
    for entry in items:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return None
        key, data = entry
        if not isinstance(key, int) or key < 0 or key >= upper:
            return None
        if prev_key is not None and key == prev_key:
            return None
        if not isinstance(data, (bytes, bytearray)):
            return None
        prev_key = key

    return [(key, bytes(data)) for key, data in items]


def _validate_levelized_proof(proof, depth):
    if not isinstance(proof, list) or len(proof) != depth:
        return None, None

    normalized = []
    work_levels = []
    for level, entries in enumerate(proof):
        if not isinstance(entries, list):
            return None, None

        normalized_level = []
        prev_key = -1
        max_key = 1 << (depth - level)
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                return None, None
            key, h = entry
            if not isinstance(key, int) or key < 0 or key >= max_key or key <= prev_key:
                return None, None
            if not isinstance(h, (bytes, bytearray)) or len(h) != 32:
                return None, None
            normalized_level.append((key, bytes(h)))
            prev_key = key

        if normalized_level:
            work_levels.append(level)
        normalized.append(normalized_level)

    return normalized, work_levels


class Node:
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
        self.root = None

    def get_root(self):
        return self.root.hash if self.root else EMPTY

    def _get_leaf(self, key):
        curr = self.root
        while curr is not None:
            if curr.level == 0:
                return curr.hash if curr.key == key else EMPTY

            if (key >> curr.level) != curr.key:
                return EMPTY

            bit = (key >> (curr.level - 1)) & 1
            curr = curr.right if bit else curr.left

        return EMPTY

    def batch_insert(self, batch):
        """
        Top-down recursive batch insertion with path compression.

        Input duplicates are deduplicated with last-value-wins semantics.
        Keys that already exist in the tree are skipped during descent.
        """
        new_items_dict = {}
        for key, data in batch:
            new_items_dict[key] = data

        proof = [[] for _ in range(self.depth)]
        if not new_items_dict:
            return [], proof

        candidates = sorted(new_items_dict.items())
        skipped_existing = set()
        self.root, _ = self._insert(
            self.root, candidates, self.depth, proof, skipped_existing
        )

        if skipped_existing:
            inserted = [
                (key, data) for key, data in candidates if key not in skipped_existing
            ]
        else:
            inserted = candidates

        return inserted, proof

    def _insert(self, node, batch, level, proof, skipped_existing):
        if not batch:
            return node, False

        if node is not None and node.level == 0:
            idx = None
            if batch[0][0] <= node.key <= batch[-1][0]:
                idx = _find_key_index(batch, node.key)
            if idx is not None:
                skipped_existing.add(node.key)
                if len(batch) == 1:
                    return node, False
                batch = batch[:idx] + batch[idx + 1 :]
                if not batch:
                    return node, False

        if node is None and len(batch) == 1:
            key, data = batch[0]
            return Node(
                0, key, hash_leaf(path_at_level(key, 0, self.depth), data)
            ), True

        if level == 0:
            key, data = batch[0]
            return Node(
                0, key, hash_leaf(path_at_level(key, 0, self.depth), data)
            ), True

        if node is None and len(batch) > 1:
            split_level = (batch[0][0] ^ batch[-1][0]).bit_length()
            if split_level < level:
                return self._insert(None, batch, split_level, proof, skipped_existing)
        elif node is not None and node.level < level:
            first_key, last_key = batch[0][0], batch[-1][0]
            node_full = node.key << node.level
            xor = (
                (first_key ^ last_key)
                | (first_key ^ node_full)
                | (last_key ^ node_full)
            )
            mask = ((1 << level) - 1) & ~((1 << node.level) - 1)
            effective = xor & mask
            if effective:
                effective_level = effective.bit_length()
                if effective_level < level:
                    return self._insert(
                        node, batch, effective_level, proof, skipped_existing
                    )
            else:
                return self._insert(node, batch, node.level, proof, skipped_existing)

        lo, hi = 0, len(batch)
        while lo < hi:
            mid = (lo + hi) >> 1
            if (batch[mid][0] >> (level - 1)) & 1:
                hi = mid
            else:
                lo = mid + 1
        left_batch = batch[:lo]
        right_batch = batch[lo:]

        left_node, right_node = None, None
        if node is not None:
            if node.level == level:
                left_node = node.left
                right_node = node.right
            else:
                node_bit = (node.key >> (level - 1 - node.level)) & 1
                if node_bit:
                    right_node = node
                else:
                    left_node = node

        prefix = batch[0][0] >> level
        left_key = prefix << 1
        right_key = left_key | 1

        if left_batch:
            new_left, left_changed = self._insert(
                left_node, left_batch, level - 1, proof, skipped_existing
            )
        else:
            new_left, left_changed = left_node, False

        if right_batch:
            new_right, right_changed = self._insert(
                right_node, right_batch, level - 1, proof, skipped_existing
            )
        else:
            new_right, right_changed = right_node, False

        if not left_changed and not right_changed:
            return node, False

        if not left_changed and left_node is not None:
            proof[level - 1].append((left_key, left_node.hash))
        if not right_changed and right_node is not None:
            proof[level - 1].append((right_key, right_node.hash))

        if new_left is None:
            return new_right, True
        if new_right is None:
            return new_left, True

        parent_path = path_at_level(prefix, level, self.depth)
        parent_hash = hash_branch(parent_path, new_left.hash, new_right.hash)
        return Node(level, prefix, parent_hash, new_left, new_right), True

    def inclusion_cert(self, key):
        curr = self.root
        if curr is None:
            return None

        siblings = []
        while curr is not None:
            if curr.level == 0:
                if curr.key != key:
                    return None
                siblings.reverse()
                return siblings

            if (key >> curr.level) != curr.key:
                return None

            level = curr.level - 1
            bit = (key >> level) & 1
            if bit:
                if curr.left is not None:
                    siblings.append((level, curr.left.hash))
                curr = curr.right
            else:
                if curr.right is not None:
                    siblings.append((level, curr.right.hash))
                curr = curr.left

        return None


def smt_compute_tree_root(proof, batch, depth):
    sorted_batch = _normalize_batch(batch, depth)
    normalized_proof, proof_levels = _validate_levelized_proof(proof, depth)
    if sorted_batch is None or normalized_proof is None:
        raise ValueError("invalid batch or proof")

    if not sorted_batch:
        if proof_levels:
            raise ValueError("non-empty proof for empty batch")
        return EMPTY

    nodes = [
        (key, EMPTY if data is EMPTY else hash_leaf(path_at_level(key, 0, depth), data))
        for key, data in sorted_batch
    ]

    work_levels = set(proof_levels)
    for i in range(len(nodes) - 1):
        xor = nodes[i][0] ^ nodes[i + 1][0]
        if xor:
            work_levels.add(xor.bit_length() - 1)
    work_levels_sorted = sorted(work_levels)

    level = 0
    wl_idx = bisect_left(work_levels_sorted, 0)
    while level < depth:
        while wl_idx < len(work_levels_sorted) and work_levels_sorted[wl_idx] < level:
            wl_idx += 1
        next_work = (
            work_levels_sorted[wl_idx] if wl_idx < len(work_levels_sorted) else depth
        )

        if next_work > level:
            skip = next_work - level
            nodes = [(key >> skip, val) for key, val in nodes]
            level = next_work
            if level >= depth:
                break

        lp = normalized_proof[level]
        next_nodes = []
        i = j = 0

        while i < len(nodes):
            key, key_val = nodes[i]
            sibling = key ^ 1
            parent = key >> 1

            if (key & 1) == 0 and i + 1 < len(nodes) and nodes[i + 1][0] == sibling:
                i += 1
                sib_val = nodes[i][1]
            elif j < len(lp) and lp[j][0] == sibling:
                sib_val = lp[j][1]
                j += 1
            else:
                sib_val = EMPTY

            parent_path = path_at_level(parent, level + 1, depth)
            if key & 1:
                parent_val = merge_branch(parent_path, sib_val, key_val)
            else:
                parent_val = merge_branch(parent_path, key_val, sib_val)

            next_nodes.append((parent, parent_val))
            i += 1

        if j != len(lp):
            raise ValueError("unused proof entries")

        nodes = next_nodes
        level += 1

    if len(nodes) != 1:
        raise ValueError(f"expected 1 root node, got {len(nodes)}")
    return nodes[0][1]


def verify_consistency(proof, old_root, new_root, batch, depth):
    normalized_batch = _normalize_batch(batch, depth)
    normalized_proof, proof_levels = _validate_levelized_proof(proof, depth)
    if normalized_batch is None or normalized_proof is None:
        return False

    if not normalized_batch:
        return old_root == new_root and not proof_levels

    nodes = []
    for key, data in normalized_batch:
        new_h = hash_leaf(path_at_level(key, 0, depth), data)
        nodes.append((key, EMPTY, new_h))

    work_levels = set(proof_levels)
    for i in range(len(nodes) - 1):
        xor = nodes[i][0] ^ nodes[i + 1][0]
        if xor:
            work_levels.add(xor.bit_length() - 1)
    work_levels_sorted = sorted(work_levels)

    level = 0
    wl_idx = 0
    while level < depth:
        while wl_idx < len(work_levels_sorted) and work_levels_sorted[wl_idx] < level:
            wl_idx += 1
        next_work = (
            work_levels_sorted[wl_idx] if wl_idx < len(work_levels_sorted) else depth
        )

        if next_work > level:
            skip = next_work - level
            nodes = [(key >> skip, old_h, new_h) for key, old_h, new_h in nodes]
            level = next_work
            if level >= depth:
                break

        lp = normalized_proof[level]
        next_nodes = []
        i = j = 0

        while i < len(nodes):
            key, key_old, key_new = nodes[i]
            sibling = key ^ 1
            parent = key >> 1

            if (key & 1) == 0 and i + 1 < len(nodes) and nodes[i + 1][0] == sibling:
                i += 1
                sib_old = nodes[i][1]
                sib_new = nodes[i][2]
            elif j < len(lp) and lp[j][0] == sibling:
                sib_old = sib_new = lp[j][1]
                j += 1
            else:
                sib_old = sib_new = EMPTY

            parent_path = path_at_level(parent, level + 1, depth)
            if key & 1:
                parent_old = merge_branch(parent_path, sib_old, key_old)
                parent_new = merge_branch(parent_path, sib_new, key_new)
            else:
                parent_old = merge_branch(parent_path, key_old, sib_old)
                parent_new = merge_branch(parent_path, key_new, sib_new)

            next_nodes.append((parent, parent_old, parent_new))
            i += 1

        if j != len(lp):
            return False

        nodes = next_nodes
        level += 1

    if len(nodes) != 1:
        return False

    return nodes[0][1] == old_root and nodes[0][2] == new_root


def verify_inclusion(cert, root_hash, key, value, depth):
    if cert is None or root_hash is EMPTY or not isinstance(cert, list):
        return False
    if not isinstance(key, int) or key < 0 or key >= (1 << depth):
        return False
    if not isinstance(value, (bytes, bytearray)):
        return False

    h = hash_leaf(path_at_level(key, 0, depth), bytes(value))
    current_key = key
    current_level = 0
    prev_level = -1

    for entry in cert:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return False
        level, sibling_hash = entry
        if (
            not isinstance(level, int)
            or not 0 <= level < depth
            or level <= prev_level
            or level < current_level
            or not isinstance(sibling_hash, (bytes, bytearray))
            or len(sibling_hash) != 32
        ):
            return False

        current_key >>= level - current_level
        parent = current_key >> 1
        parent_path = path_at_level(parent, level + 1, depth)
        sibling_hash = bytes(sibling_hash)

        if current_key & 1:
            h = hash_branch(parent_path, sibling_hash, h)
        else:
            h = hash_branch(parent_path, h, sibling_hash)

        current_key = parent
        current_level = level + 1
        prev_level = level

    if current_level < depth:
        current_key >>= depth - current_level

    return current_key == 0 and h == root_hash
