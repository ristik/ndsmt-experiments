# ===========================================================================
# Optimized Path-Compressed Full Sparse Merkle Tree v2
# ===========================================================================
#
# This is the same classical full SMT semantics as `ndsmt_lvl.py`: implicit
# blank leaves over the full key space, with path compression used only as an
# in-memory optimization. Canonicality therefore comes for free. The operator
# cannot choose an alternative tree shape: the committed structure is the fixed
# full binary tree indexed by key positions.
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
# Domain separators 0x00 / 0x01 distinguish leaves and branches. The sentinel
# bit makes the path self-delimiting with respect to the tree level.
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
# An explicit non-empty skeleton proof is a flat post-order opcode stream:
#
#   'S', level, key, hash   : untouched non-empty subtree rooted at (level, key)
#   'L'                     : next verifier-sorted batch leaf
#   'B', level, key         : pop right then left and merge them under branch
#                             (level, key)
#
# Stack items are:
#   (level, key, old_hash, new_hash)
#
# Semantics:
# - `level = 0` is a leaf, `level = depth` is the root.
# - `key` is the positional index at that level (original_leaf_key >> level).
# - `S` contributes the same subtree hash to old and new state.
# - `L` contributes EMPTY in the old state and the batch leaf hash in the new
#   state.
# - `B` applies `merge_branch(...)` independently to the old and new states.
#   Its children may start at any lower levels as long as they promote through
#   empty siblings into the left / right child slots of `(level, key)`.
#
# The verifier accepts only if the stack finishes with exactly one item whose
# `(old_hash, new_hash)` matches the committed `(old_root, new_root)`. Missing
# levels above the top non-empty subtree are implicit pass-through compression.
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


def _validate_skeleton_proof(proof, depth):
    if not isinstance(proof, list):
        return None

    normalized = []
    i = 0
    while i < len(proof):
        tag = proof[i]
        i += 1
        if tag == "L":
            normalized.append(("L",))
        elif tag == "S":
            if i + 2 >= len(proof):
                return None
            level, key, h = proof[i], proof[i + 1], proof[i + 2]
            i += 3
            if (
                not isinstance(level, int)
                or not 0 <= level <= depth
                or not isinstance(key, int)
                or key < 0
                or key >= (1 << (depth - level))
                or not isinstance(h, (bytes, bytearray))
                or len(h) != 32
            ):
                return None
            normalized.append(("S", level, key, bytes(h)))
        elif tag == "B":
            if i + 1 >= len(proof):
                return None
            level, key = proof[i], proof[i + 1]
            i += 2
            if (
                not isinstance(level, int)
                or not 1 <= level <= depth
                or not isinstance(key, int)
                or key < 0
                or key >= (1 << (depth - level))
            ):
                return None
            normalized.append(("B", level, key))
        else:
            return None

    return normalized


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

    def batch_insert(self, batch):
        new_items_dict = {}
        for key, data in batch:
            new_items_dict[key] = data

        levelized = [[] for _ in range(self.depth)]
        if not new_items_dict:
            return [], []

        candidates = sorted(new_items_dict.items())
        skipped_existing = set()
        self.root, _ = self._insert(
            self.root, candidates, self.depth, levelized, skipped_existing
        )

        if skipped_existing:
            inserted = [
                (key, data) for key, data in candidates if key not in skipped_existing
            ]
        else:
            inserted = candidates

        proof = _levelized_to_skeleton(levelized, inserted, self.depth)
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


class _ProgItem:
    __slots__ = ["current_level", "current_key", "prog"]

    def __init__(self, current_level, current_key, prog):
        self.current_level = current_level
        self.current_key = current_key
        self.prog = prog


def _levelized_to_skeleton(proof, batch, depth):
    normalized_batch = _normalize_batch(batch, depth)
    normalized_proof, proof_levels = _validate_levelized_proof(proof, depth)
    if normalized_batch is None or normalized_proof is None:
        raise ValueError("invalid intermediate proof or batch")

    if not normalized_batch:
        return []

    nodes = [_ProgItem(0, key, ["L"]) for key, _ in normalized_batch]

    work_levels = set(proof_levels)
    for i in range(len(normalized_batch) - 1):
        xor = normalized_batch[i][0] ^ normalized_batch[i + 1][0]
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
            for node in nodes:
                node.current_level = next_work
                node.current_key >>= skip
            level = next_work
            if level >= depth:
                break

        lp = normalized_proof[level]
        next_nodes = []
        i = j = 0

        while i < len(nodes):
            node = nodes[i]
            sibling = node.current_key ^ 1
            parent = node.current_key >> 1

            if (
                (node.current_key & 1) == 0
                and i + 1 < len(nodes)
                and nodes[i + 1].current_key == sibling
            ):
                left = node
                right = nodes[i + 1]
                i += 2
                prog = left.prog + right.prog + ["B", level + 1, parent]
                next_nodes.append(_ProgItem(level + 1, parent, prog))
            elif j < len(lp) and lp[j][0] == sibling:
                proof_item = _ProgItem(level, sibling, ["S", level, sibling, lp[j][1]])
                j += 1
                if node.current_key & 1:
                    left, right = proof_item, node
                else:
                    left, right = node, proof_item
                prog = left.prog + right.prog + ["B", level + 1, parent]
                next_nodes.append(_ProgItem(level + 1, parent, prog))
                i += 1
            else:
                node.current_level = level + 1
                node.current_key = parent
                next_nodes.append(node)
                i += 1

        if j != len(lp):
            raise ValueError("unused intermediate proof entries")

        nodes = next_nodes
        level += 1

    if len(nodes) != 1:
        raise ValueError(f"expected 1 root program, got {len(nodes)}")

    return nodes[0].prog


def verify_consistency(proof, old_root, new_root, batch, depth):
    normalized_batch = _normalize_batch(batch, depth)
    normalized_proof = _validate_skeleton_proof(proof, depth)
    if normalized_batch is None or normalized_proof is None:
        return False

    if not normalized_batch:
        return old_root == new_root and normalized_proof == []

    stack = []
    bi = 0

    try:
        for op in normalized_proof:
            tag = op[0]

            if tag == "S":
                _, level, key, h = op
                stack.append((level, key, h, h))

            elif tag == "L":
                key, data = normalized_batch[bi]
                bi += 1
                h = hash_leaf(path_at_level(key, 0, depth), data)
                stack.append((0, key, EMPTY, h))

            elif tag == "B":
                _, level, key = op
                right_level, right_key, right_old, right_new = stack.pop()
                left_level, left_key, left_old, left_new = stack.pop()

                if level <= left_level or level <= right_level:
                    return False

                left_child = key << 1
                right_child = left_child | 1
                if (left_key >> ((level - 1) - left_level)) != left_child:
                    return False
                if (right_key >> ((level - 1) - right_level)) != right_child:
                    return False

                parent_path = path_at_level(key, level, depth)
                parent_old = merge_branch(parent_path, left_old, right_old)
                parent_new = merge_branch(parent_path, left_new, right_new)
                stack.append((level, key, parent_old, parent_new))

            else:
                return False

    except (IndexError, TypeError, ValueError):
        return False

    if bi != len(normalized_batch) or len(stack) != 1:
        return False

    _, _, old_h, new_h = stack[0]
    return old_h == old_root and new_h == new_root


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
