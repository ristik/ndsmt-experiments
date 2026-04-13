# ===========================================================================
# Radix Sparse Merkle Tree v5a (RSMT5a)
# ===========================================================================
#
# This variant keeps the RSMT5 tree commitment and consistency proof exactly
# the same, but uses a simpler inclusion proof format:
#   [(sibling_hash, depth, sibling_lo, sibling_hi), ...]
# in leaf-to-root order.
#
# Compared with RSMT5:
# - no bitmap is carried in the inclusion proof
# - each sibling operand carries its explicit depth
# - inclusion verification iterates directly over proof elements instead of
#   scanning all 256 depth positions


KEY_BYTES = 32

DEPTH_BYTES = [d.to_bytes(1, "big") for d in range(256)]
BIT_REVERSE_TABLE = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def get_sort_key(k):
    """
    Converts integer key to LSB-first lexicographical sort key.
    (Reverse byte order, then reverse bits in each byte.)
    """
    return k.to_bytes(KEY_BYTES, "big")[::-1].translate(BIT_REVERSE_TABLE)


def _key_bytes(key):
    return key.to_bytes(KEY_BYTES, "big")


def hash_leaf(key, value):
    return hashlib.sha256(b"\x00" + _key_bytes(key) + value).digest()


def hash_node(lh, rh, depth, lo, hi):
    return hashlib.sha256(
        b"\x01" + DEPTH_BYTES[depth] + lh + rh + _key_bytes(lo) + _key_bytes(hi)
    ).digest()


def path_len(p):
    return p.bit_length() - 1


def first_diff_bit(a, b):
    xor = a ^ b
    if xor == 0:
        return None
    return (xor & -xor).bit_length() - 1


def _is_canonical_split(left_hi, right_lo, depth):
    if left_hi is None or right_lo is None:
        return False
    split = first_diff_bit(left_hi, right_lo)
    return (
        split == depth
        and ((left_hi >> depth) & 1) == 0
        and ((right_lo >> depth) & 1) == 1
    )


def _branch_lo(node):
    if node is None:
        return None
    if isinstance(node, LeafBranch):
        return node.key
    return node.lo


def _branch_hi(node):
    if node is None:
        return None
    if isinstance(node, LeafBranch):
        return node.key
    return node.hi


class LeafBranch:
    __slots__ = ["key", "value", "_hash"]

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self._hash = hash_leaf(key, value)

    def get_hash(self):
        return self._hash


class NodeBranch:
    __slots__ = ["path", "left", "right", "depth", "lo", "hi", "_hash"]

    def __init__(self, path, left, right, depth):
        self.path = path
        self.left = left
        self.right = right
        self.depth = depth
        self.lo = _branch_lo(left)
        self.hi = _branch_hi(right)
        self._hash = None

    def get_hash(self):
        if self._hash is None:
            self._hash = hash_node(
                self.left.get_hash(),
                self.right.get_hash(),
                self.depth,
                self.lo,
                self.hi,
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

        proof_out = []
        self.root = self._insert_proof(self.root, items, 0, len(items), 0, proof_out)
        return items, proof_out

    def _build_subtree(self, batch, start, end, start_bit, proof_out, frozen):
        if end - start == 1:
            key, value = batch[start]
            if key in frozen:
                proof_out.extend(["S", frozen[key], key, key])
            else:
                proof_out.append("L")
            return LeafBranch(key, value)

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

        left = self._build_subtree(batch, start, mid, split, proof_out, frozen)
        right = self._build_subtree(batch, mid, end, split, proof_out, frozen)
        proof_out.extend(["N", split])
        return NodeBranch(cp, left, right, split)

    def _emit_subtree(self, node, proof_out):
        if node is None:
            proof_out.extend(["S", None])
            return
        proof_out.extend(["S", node.get_hash(), _branch_lo(node), _branch_hi(node)])

    def _insert_proof(self, node, batch, start, end, start_bit, proof_out):
        if start == end:
            self._emit_subtree(node, proof_out)
            return node

        if node is None:
            return self._build_subtree(batch, start, end, start_bit, proof_out, {})

        if isinstance(node, LeafBranch):
            frozen = {node.key: node.get_hash()}
            mixed = batch[start:end] + [(node.key, node.value)]
            mixed.sort(key=lambda x: get_sort_key(x[0]))
            return self._build_subtree(
                mixed, 0, len(mixed), start_bit, proof_out, frozen
            )

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
            return self._node_split_proof(
                node, batch, start, end, start_bit, first_div, proof_out
            )

        split = start_bit + n_path

        low, high = start, end
        while low < high:
            mid = (low + high) // 2
            if (batch[mid][0] >> split) & 1:
                high = mid
            else:
                low = mid + 1
        mid = low

        new_left = self._insert_proof(node.left, batch, start, mid, split, proof_out)
        new_right = self._insert_proof(node.right, batch, mid, end, split, proof_out)
        proof_out.extend(["N", split])

        node.left = new_left
        node.right = new_right
        node.lo = _branch_lo(new_left)
        node.hi = _branch_hi(new_right)
        node._hash = None
        return node

    def _node_split_proof(
        self, node, batch, start, end, start_bit, first_div, proof_out
    ):
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
            new_left = self._insert_proof(node, batch, start, mid, new_split, proof_out)
            new_right = self._insert_proof(None, batch, mid, end, new_split, proof_out)
        else:
            new_left = self._insert_proof(None, batch, start, mid, new_split, proof_out)
            new_right = self._insert_proof(node, batch, mid, end, new_split, proof_out)
        proof_out.extend(["N", new_split])

        return NodeBranch(new_cp, new_left, new_right, new_split)

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
                sibling = node.left
                siblings.append(
                    (
                        sibling.get_hash(),
                        depth,
                        _branch_lo(sibling),
                        _branch_hi(sibling),
                    )
                )
                node = node.right
            else:
                sibling = node.right
                siblings.append(
                    (
                        sibling.get_hash(),
                        depth,
                        _branch_lo(sibling),
                        _branch_hi(sibling),
                    )
                )
                node = node.left

        if not isinstance(node, LeafBranch) or node.key != key:
            return None

        siblings.reverse()
        return siblings


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
                if h is None:
                    stack.append((None, None, None, None, None, None))
                else:
                    lo = proof[pi]
                    hi = proof[pi + 1]
                    pi += 2
                    if not isinstance(lo, int) or not isinstance(hi, int):
                        return False
                    stack.append((h, lo, hi, h, lo, hi))

            elif tag == "L":
                key, value = sorted_batch[bi]
                bi += 1
                h = hash_leaf(key, value)
                stack.append((None, None, None, h, key, key))

            elif tag == "N":
                depth = proof[pi]
                pi += 1
                if not isinstance(depth, int) or not 0 <= depth < 256:
                    return False

                rh0, rlo0, rhi0, rh1, rlo1, rhi1 = stack.pop()
                lh0, llo0, lhi0, lh1, llo1, lhi1 = stack.pop()

                if lh1 is None or rh1 is None:
                    return False
                if not _is_canonical_split(lhi1, rlo1, depth):
                    return False

                h1 = hash_node(lh1, rh1, depth, llo1, rhi1)
                lo1 = llo1
                hi1 = rhi1

                if lh0 is None and rh0 is None:
                    h0 = None
                    lo0 = None
                    hi0 = None
                elif lh0 is None:
                    h0 = rh0
                    lo0 = rlo0
                    hi0 = rhi0
                elif rh0 is None:
                    h0 = lh0
                    lo0 = llo0
                    hi0 = lhi0
                else:
                    if not _is_canonical_split(lhi0, rlo0, depth):
                        return False
                    h0 = hash_node(lh0, rh0, depth, llo0, rhi0)
                    lo0 = llo0
                    hi0 = rhi0

                stack.append((h0, lo0, hi0, h1, lo1, hi1))

            else:
                return False
    except (IndexError, TypeError, ValueError):
        return False

    if pi != len(proof) or bi != len(sorted_batch) or len(stack) != 1:
        return False

    h0, _, _, h1, _, _ = stack[0]
    return h0 == old_root and h1 == new_root


def verify_inclusion(cert, root_hash, key, value):
    if cert is None or root_hash is None or not isinstance(cert, list):
        return False

    h = hash_leaf(key, value)
    lo = key
    hi = key
    prev_depth = 256

    for sibling in cert:
        if not isinstance(sibling, (list, tuple)) or len(sibling) != 4:
            return False

        sib_hash, depth, sib_lo, sib_hi = sibling
        if not isinstance(depth, int) or not 0 <= depth < 256:
            return False
        if depth >= prev_depth:
            return False

        if (key >> depth) & 1:
            if not _is_canonical_split(sib_hi, lo, depth):
                return False
            h = hash_node(sib_hash, h, depth, sib_lo, hi)
            lo = sib_lo
        else:
            if not _is_canonical_split(hi, sib_lo, depth):
                return False
            h = hash_node(h, sib_hash, depth, lo, sib_hi)
            hi = sib_hi

        prev_depth = depth

    return h == root_hash
