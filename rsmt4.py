# ===========================================================================
# Radix Sparse Merkle Tree v4 (RSMT4)
# ===========================================================================
# This is a brute-force attempt to maintain the same optimal inclusion proof format
# and internal node hashing with short input (==> one hash permutation per internal
# node in ZK proving circuit)
# Unfortunately, the compromise is linear batch insertion+proof generation effort.
#
# Leaf hashes remain anchored to full keys:
#   H_leaf(key, value) = SHA256(0x00 || key_32B || value)
#
# Internal nodes hash exactly two children and their split depth:
#   H_node(lh, rh, depth) = SHA256(0x01 || depth_1B || lh || rh)
#
# Inclusion proofs stay unchanged from ndrsmt3o:
#   - bitmap: 256-bit integer with bits set at branch depths
#   - siblings: sibling hashes ordered leaf-to-root
#
# ---------------------------------------------------------------------------
# Consistency Proof
# ---------------------------------------------------------------------------
#
# The v3 proof could preserve an old subtree hash opaquely under a new parent,
# reconstruct both roots, and still accept a malformed post-state whose radix
# topology was not canonical for the committed keys.
#
# RSMT4 fixes this without changing the tree hash function or the inclusion
# proof format. The proof opens exactly the old boundary leaves needed to pin
# every changed post-state split.
#
# Proof format (LSB-first post-order stream):
#   'S', hash         : opaque unchanged old subtree
#   'E', key, value   : exposed pre-state boundary leaf (present both before
#                       and after the batch)
#   'L'               : new leaf consumed from the verifier-sorted batch
#   'N', depth        : internal node in the opened proof skeleton
#
# Verification stack entry:
#   (h0, h1, lo, hi, delta)
#
#   h0, h1 : pre-state / post-state subtree hashes
#   lo, hi : authenticated leftmost / rightmost leaf keys of the opened
#            subtree fragment, or None if not known
#   delta  : whether the post-state subtree contains any new leaf
#
# A changed node (delta = True) must prove its split depth canonically from
# authenticated boundary keys:
#   depth == lowest_differing_bit(left.hi, right.lo)
#
# and the bit at that depth must route left.hi to the 0-side and right.lo to
# the 1-side.
#
# The prover opens only the minimal touched frontier:
#   - for every changed node, expose the old rightmost leaf of its left child
#     if that leaf is pre-existing
#   - for every changed node, expose the old leftmost leaf of its right child
#     if that leaf is pre-existing
#
# Every other unchanged old subtree remains a single 'S' hash.

import hashlib

KEY_BYTES = 32
DEPTH_BYTES = [d.to_bytes(1, "big") for d in range(256)]
BIT_REVERSE_TABLE = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def get_sort_key(k):
    """
    LSB-first lexicographical sort key.
    """
    return k.to_bytes(KEY_BYTES, "big")[::-1].translate(BIT_REVERSE_TABLE)


def hash_leaf(key, value):
    return hashlib.sha256(b"\x00" + key.to_bytes(KEY_BYTES, "big") + value).digest()


def hash_node(lh, rh, depth):
    return hashlib.sha256(b"\x01" + DEPTH_BYTES[depth] + lh + rh).digest()


def path_len(p):
    return p.bit_length() - 1


def first_diff_bit(a, b):
    xor = a ^ b
    if xor == 0:
        return None
    return (xor & -xor).bit_length() - 1


class LeafBranch:
    __slots__ = ["key", "value", "_hash"]

    def __init__(self, key, value):
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

        items = list(new_items.items())
        items.sort(key=lambda x: get_sort_key(x[0]))

        self.root = self._insert(self.root, items, 0, len(items), 0)

        new_keys = {k for k, _ in items}
        exposed_old_keys = set()
        self._collect_required_old(self.root, new_keys, exposed_old_keys)

        active = {}
        self._mark_active(self.root, new_keys, exposed_old_keys, active)

        proof_out = []
        self._emit_consistency(self.root, new_keys, active, proof_out)
        return items, proof_out

    def _build_subtree(self, batch, start, end, start_bit):
        if end - start == 1:
            k, v = batch[start]
            return LeafBranch(k, v)

        xor = (batch[start][0] ^ batch[end - 1][0]) >> start_bit
        split = start_bit + (xor & -xor).bit_length() - 1

        low, high = start, end
        while low < high:
            mid = (low + high) // 2
            if (batch[mid][0] >> split) & 1:
                high = mid
            else:
                low = mid + 1
        mid = low

        n_common = split - start_bit
        cp = (1 << n_common) | ((batch[start][0] >> start_bit) & ((1 << n_common) - 1))

        ln = self._build_subtree(batch, start, mid, split)
        rn = self._build_subtree(batch, mid, end, split)
        return NodeBranch(cp, ln, rn, split)

    def _insert(self, node, batch, start, end, start_bit):
        if start == end:
            return node

        if node is None:
            return self._build_subtree(batch, start, end, start_bit)

        if isinstance(node, LeafBranch):
            mixed = batch[start:end] + [(node.key, node.value)]
            mixed.sort(key=lambda x: get_sort_key(x[0]))
            return self._build_subtree(mixed, 0, len(mixed), start_bit)

        n_path = path_len(node.path)
        node_prefix = node.path & ((1 << n_path) - 1)

        first_div = n_path
        xor_start = ((batch[start][0] >> start_bit) & ((1 << n_path) - 1)) ^ node_prefix
        if xor_start:
            first_div = min(first_div, (xor_start & -xor_start).bit_length() - 1)
        xor_end = ((batch[end - 1][0] >> start_bit) & ((1 << n_path) - 1)) ^ node_prefix
        if xor_end:
            first_div = min(first_div, (xor_end & -xor_end).bit_length() - 1)

        if first_div < n_path:
            return self._node_split(node, batch, start, end, start_bit, first_div)

        split = start_bit + n_path

        low, high = start, end
        while low < high:
            mid = (low + high) // 2
            if (batch[mid][0] >> split) & 1:
                high = mid
            else:
                low = mid + 1
        mid = low

        new_left = self._insert(node.left, batch, start, mid, split)
        new_right = self._insert(node.right, batch, mid, end, split)

        node.left = new_left
        node.right = new_right
        node._hash = None
        return node

    def _node_split(self, node, batch, start, end, start_bit, first_div):
        n_path = path_len(node.path)
        node_prefix = node.path & ((1 << n_path) - 1)

        n_common = first_div
        new_cp = (1 << n_common) | (node_prefix & ((1 << n_common) - 1))
        new_split = start_bit + n_common
        old_dir = (node_prefix >> n_common) & 1

        new_path = node.path >> n_common
        node.path = new_path if new_path != 0 else 1

        low, high = start, end
        while low < high:
            mid = (low + high) // 2
            if (batch[mid][0] >> new_split) & 1:
                high = mid
            else:
                low = mid + 1
        mid = low

        if old_dir == 0:
            new_left = self._insert(node, batch, start, mid, new_split)
            new_right = self._insert(None, batch, mid, end, new_split)
        else:
            new_left = self._insert(None, batch, start, mid, new_split)
            new_right = self._insert(node, batch, mid, end, new_split)

        return NodeBranch(new_cp, new_left, new_right, new_split)

    def _collect_required_old(self, node, new_keys, required_old_keys):
        # Returns the leftmost and rightmost keys of this subtree in tree order,
        # together with whether the subtree contains any newly inserted leaf.
        if isinstance(node, LeafBranch):
            is_new = node.key in new_keys
            return node.key, node.key, is_new

        left_lo, left_hi, left_changed = self._collect_required_old(
            node.left, new_keys, required_old_keys
        )
        right_lo, right_hi, right_changed = self._collect_required_old(
            node.right, new_keys, required_old_keys
        )

        changed = left_changed or right_changed
        if changed:
            if left_hi not in new_keys:
                required_old_keys.add(left_hi)
            if right_lo not in new_keys:
                required_old_keys.add(right_lo)

        return left_lo, right_hi, changed

    def _mark_active(self, node, new_keys, exposed_old_keys, active):
        if isinstance(node, LeafBranch):
            flag = node.key in new_keys or node.key in exposed_old_keys
        else:
            left_flag = self._mark_active(node.left, new_keys, exposed_old_keys, active)
            right_flag = self._mark_active(
                node.right, new_keys, exposed_old_keys, active
            )
            flag = left_flag or right_flag

        active[node] = flag
        return flag

    def _emit_consistency(self, node, new_keys, active, proof_out):
        if not active[node]:
            proof_out.extend(["S", node.get_hash()])
            return

        if isinstance(node, LeafBranch):
            if node.key in new_keys:
                proof_out.append("L")
            else:
                proof_out.extend(["E", node.key, node.value])
            return

        self._emit_consistency(node.left, new_keys, active, proof_out)
        self._emit_consistency(node.right, new_keys, active, proof_out)
        proof_out.extend(["N", node.depth])

    def _find_leaf(self, key):
        node = self.root
        bit = 0
        while node is not None:
            if isinstance(node, LeafBranch):
                return node if node.key == key else None
            n = path_len(node.path)
            if ((key >> bit) & ((1 << n) - 1)) != (node.path & ((1 << n) - 1)):
                return None
            bit += n
            node = node.right if ((key >> bit) & 1) else node.left
        return None

    def inclusion_cert(self, key):
        node = self.root
        if node is None:
            return None

        bitmap = 0
        siblings = []
        bit = 0

        while isinstance(node, NodeBranch):
            n = path_len(node.path)
            prefix = node.path & ((1 << n) - 1)
            if ((key >> bit) & ((1 << n) - 1)) != prefix:
                return None
            bit += n
            depth = node.depth
            if (key >> bit) & 1:
                siblings.append(node.left.get_hash())
                node = node.right
            else:
                siblings.append(node.right.get_hash())
                node = node.left
            bitmap |= 1 << depth

        if not isinstance(node, LeafBranch) or node.key != key:
            return None

        return {"bitmap": bitmap, "siblings": siblings}


def verify_consistency(proof, old_root, new_root, batch, _=None):
    if not batch:
        return old_root == new_root

    sorted_batch = sorted(batch, key=lambda x: get_sort_key(x[0]))
    stack = []
    pi = 0
    bi = 0

    try:
        while pi < len(proof):
            tag = proof[pi]
            pi += 1

            if tag == "S":
                h = proof[pi]
                pi += 1
                stack.append((h, h, None, None, False))

            elif tag == "E":
                key = proof[pi]
                value = proof[pi + 1]
                pi += 2
                h = hash_leaf(key, value)
                stack.append((h, h, key, key, False))

            elif tag == "L":
                key, value = sorted_batch[bi]
                bi += 1
                h = hash_leaf(key, value)
                stack.append((None, h, key, key, True))

            elif tag == "N":
                depth = proof[pi]
                pi += 1

                if not isinstance(depth, int) or not 0 <= depth < 256:
                    return False

                rh0, rh1, rlo, rhi, rdelta = stack.pop()
                lh0, lh1, llo, lhi, ldelta = stack.pop()

                delta = ldelta or rdelta
                if delta:
                    if lhi is None or rlo is None:
                        return False
                    split = first_diff_bit(lhi, rlo)
                    if split != depth:
                        return False
                    if ((lhi >> depth) & 1) != 0:
                        return False
                    if ((rlo >> depth) & 1) != 1:
                        return False

                if lh0 is None and rh0 is None:
                    h0 = None
                elif lh0 is None:
                    h0 = rh0
                elif rh0 is None:
                    h0 = lh0
                else:
                    h0 = hash_node(lh0, rh0, depth)

                h1 = hash_node(lh1, rh1, depth)
                stack.append((h0, h1, llo, rhi, delta))

            else:
                return False

    except (IndexError, ValueError, TypeError):
        return False

    if pi != len(proof) or bi != len(sorted_batch) or len(stack) != 1:
        return False

    r0, r1, _, _, _ = stack[0]
    return r0 == old_root and r1 == new_root


def verify_inclusion(cert, root_hash, key, value):
    bitmap = cert["bitmap"]
    siblings = list(cert["siblings"])

    h = hash_leaf(key, value)
    j = len(siblings)

    for d in range(255, -1, -1):
        if not (bitmap >> d) & 1:
            continue
        j -= 1
        if j < 0:
            return False
        sibling = siblings[j]
        if (key >> d) & 1:
            h = hash_node(sibling, h, d)
        else:
            h = hash_node(h, sibling, d)

    return j == 0 and h == root_hash
