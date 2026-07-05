# ===========================================================================
# Radix Sparse Merkle Tree v6a (RSMT6a) - compact v6 consistency proofs
# ===========================================================================
#
# This is a proof-format and verifier-state optimization of RSMT6. Tree
# hashes, roots, insertion semantics, and inclusion/non-inclusion proofs are
# unchanged. Only consistency proofs differ. Lineage:
#
#   v3  (ndrsmt3.py / ndrsmt3o.py): depth-committing nodes; consistency
#       proof enforces STRUCTURAL preservation only.
#   v4  (rsmt4.py): canonicality via opened boundary leaves ("witness
#       route") -- sound but widens the touched frontier.
#   v5  (rsmt5.py / rsmt5a.py): canonicality via committed traversal
#       ranges [lo, hi] in node hashes.
#   v6  (rsmt6.py): canonicality (and more) via the committed region
#       (absolute key prefix addressing the node) plus local
#       edge-coherence checks in the verifier. The "commitment route".
#   v6a (this file): removes the redundant region operand from every N
#       opcode and derives it from authenticated child advice.
#
# Why v3 is not enough (the shadow-insertion attack)
# --------------------------------------------------
# v3's verifier binds hash structure but not key positions. Concretely,
# with a pre-state containing recorded (k, v) under certified root r0,
# the 3-opcode stream
#       S(r0), L, N(d*)          with bit d* of k equal to 1
# VERIFIES as a v3 consistency proof for batch {(k, v')}: the old side
# pass-through reproduces r0, the new root hangs the whole old tree on
# the 0-side of a new junction while the key-directed descent for k goes
# to the 1-side. Under the new certified root, (k, v') is provable and
# (k, v) is orphaned. Across rounds this is equivocation on key k --
# a double-spend enabler -- and every v3 check passes, for the BFT Core
# and for the recursive full-history audit alike. (Attack reproduced
# against ndrsmt3o.verify_consistency; see the paper, Sec. 5.2.)
#
# v6/v6a reject the attack: a preserved subtree meeting a NEW junction must
# be presented opened (O / OL below), and edge coherence forces the new
# junction's region p and sides to satisfy  p||0 <= region(left child),
# p||1 <= region(right child), child depth > junction depth. Since the
# preserved subtree's region is a prefix of k, the new leaf with the
# same key k cannot be placed on the other side.
#
# Hashes
# ------
#   H_leaf(key, value)     = SHA256(0x00 || key_32B || value)
#   H_node(d, p, lh, rh)   = SHA256(0x01 || d_1B || region_32B || lh || rh)
#
# where region p is the d-bit key prefix addressing the node (packed
# left-aligned into 32 bytes; injective together with d). A leaf's
# region is its full key. The region, like the depth, is an ABSOLUTE
# property of a node: splitting an edge above a node changes neither, so
# v3's insert-immutability of pre-existing hashes is preserved.
# Inclusion proofs do not change: the verifier derives each expected
# region from the queried key itself.
#
# Consistency proof format (flat post-order opcode stream)
# --------------------------------------------------------
#   'S',  hash                      : opaque preserved subtree. Only
#                                     admissible where the parent
#                                     junction is pre-existing.
#   'O',  depth, region, lh, rh    : preserved junction, opened one
#                                     level (hashed by the verifier, so
#                                     the annotations are collision-
#                                     bound to the digest).
#   'OL', key, value               : preserved leaf, opened.
#   'L'                            : new leaf; (key, value) is consumed
#                                     from the sorted batch.
#   'N',  depth                    : junction over the two preceding stack
#                                     entries. Its region is derived from
#                                     authenticated child advice.
#
# The verifier runs a stack machine over
#       (old_digest, new_digest, advice_depth, advice_region).
# At N(d), every child carrying advice yields the same parent region p and
# must extend p||side at a depth greater than d. At least one child must carry
# advice. If N is absent from the pre-state (one old child digest is None),
# both children must carry advice, preventing an opaque S from being attached
# through a new edge. Checking every available advised edge also makes v6's
# new/old stack flag redundant.
#
# Soundness reduces directly to v6: expand any accepted v6a proof by inserting
# the uniquely derived p after every N(d). The resulting v6 proof has the same
# hash evaluation and passes every v6-required edge check. Completeness
# holds because every N emitted on a changed frontier has an advised child.
#
# Security Framework:
#
# Model. Hash trees T ::= Leaf(k, v) | Node(d, p, l, r); dig(.) as above,
# dig(empty) = None. Well-formed: at every junction, dep(child) > d and
# p||0 <= reg(left), p||1 <= reg(right)  (reg of a leaf is its key).
# Well-formed trees have pairwise distinct leaf keys and hence represent
# a partial map map(T). Canonical: every junction's region is the
# longest common prefix of the keys below it; canonical trees are unique
# per key set. A certified history is r_0 = None and rounds
# (B_i, pi_i, r_i) each accepted by verify_consistency.
#
# Definition (Append-only consistency). There exist well-formed trees
#   T_0 = empty, ..., T_n  with  dig(T_i) = r_i  and
#   map(T_i) = map(T_{i-1})  disjoint-union  B_i        for every round.
#
# Theorem 1 (Round soundness). If T is well-formed with dig(T) = r_{i-1}
# and the verifier accepts (pi, r_{i-1}, r_i, B), then (unless a
# collision can be extracted) there is a well-formed T' with dig(T') = r_i
# and map(T') = map(T) disjoint-union B. In particular the keys of B are
# fresh: NO ACCEPTING RUN EXISTS for a batch re-recording a present key.
# If T is canonical, T' is canonical.
#
# Theorem 2 (History soundness). Every accepted certified history is
# append-only consistent, with all T_i canonical -- or a collision is
# computable from its transcript.
#
# Corollary 1 (Unicity). Across ALL certified roots of a history, no key
# is ever bound to two different values: verifying inclusion proofs for
# (k, v) against r_i and (k, v') against r_j with v != v' yield a
# collision. (This is the global no-equivocation property that v3
# provided only per-root, not across rounds.)
#
# Corollary 2 (No false non-inclusion). A verifying inclusion proof for
# (k, v) against r_i together with a verifying non-inclusion witness for
# k against r_j, j >= i, yields a collision.
#
# Corollary 3 (Service completeness). In the canonical T_i, every
# recorded binding has a verifying inclusion proof and every absent key
# a verifying non-inclusion witness; producing them needs only the tree
# data (availability, not integrity).
#
# Assumptions: collision resistance of the hash; trusted genesis
# r_0 = None; root authenticity certified externally (Consensus Layer).
# No assumptions about the operator (trustless setup).
# ---------------------------------------------------------------------------

import hashlib

KEY_BYTES = 32
KEY_BITS = KEY_BYTES * 8  # 256; also the "depth" of a leaf

DEPTH_BYTES = [d.to_bytes(1, "big") for d in range(256)]


# ---------------------------------------------------------------------------
# Bit / prefix utilities (plain MSB-first order)
# ---------------------------------------------------------------------------


def key_bit(k, depth):
    """Bit of key k at position `depth`, counted from the MSB."""
    return (k >> (KEY_BITS - 1 - depth)) & 1


def prefix(k, d):
    """The d-bit region prefix of key k (as an integer with d bits)."""
    return k >> (KEY_BITS - d) if d > 0 else 0


def first_divergence(a, b):
    """First bit position (from MSB) where 256-bit integers a, b differ."""
    x = a ^ b
    return KEY_BITS - x.bit_length()  # x != 0 expected


# ---------------------------------------------------------------------------
# Hashers
# ---------------------------------------------------------------------------


def hash_leaf(key, value):
    return hashlib.sha256(b"\x00" + key.to_bytes(KEY_BYTES, "big") + value).digest()


def hash_node(depth, region, lh, rh):
    # Region packed left-aligned into 32 bytes; injective together with depth.
    packed = (region << (KEY_BITS - depth)).to_bytes(KEY_BYTES, "big")
    return hashlib.sha256(b"\x01" + DEPTH_BYTES[depth] + packed + lh + rh).digest()


# ---------------------------------------------------------------------------
# Node types (absolute depth and region stored per junction)
# ---------------------------------------------------------------------------


class LeafBranch:
    __slots__ = ["key", "value", "_hash"]

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self._hash = hash_leaf(key, value)

    def get_hash(self):
        return self._hash


class NodeBranch:
    __slots__ = ["depth", "region", "left", "right", "_hash"]

    def __init__(self, depth, region, left, right):
        self.depth = depth
        self.region = region
        self.left = left
        self.right = right
        self._hash = None

    def get_hash(self):
        if self._hash is None:
            self._hash = hash_node(
                self.depth, self.region, self.left.get_hash(), self.right.get_hash()
            )
        return self._hash


# ---------------------------------------------------------------------------
# Radix Sparse Merkle Tree with coherent consistency proofs
# ---------------------------------------------------------------------------


class SparseMerkleTree:
    def __init__(self, depth=KEY_BITS):
        if depth != KEY_BITS:
            raise ValueError("RSMT6 uses fixed 256-bit keys")
        self.root = None

    def get_root(self):
        return self.root.get_hash() if self.root else None

    # -- insertion with proof ------------------------------------------------

    def batch_insert(self, batch):
        """Insert new (key, value) pairs; keys already present are skipped
        (the honest dedup -- an adversarial re-record is REJECTED by the
        verifier, see Theorem 1). Returns (applied_items, proof_stream)."""
        new_items = {}
        for key, value in batch:
            if key in new_items or self._find_leaf(key) is not None:
                continue
            new_items[key] = value
        items = sorted(new_items.items())
        if not items:
            return [], []

        proof = []
        self.root = self._insert(self.root, items, 0, len(items), False, proof)
        return items, proof

    def _emit_preserved(self, node, parent_new, out):
        """A subtree untouched this round: opaque under an old junction,
        opened one level under a new one (the split edge)."""
        if not parent_new:
            out.extend(["S", node.get_hash()])
        elif isinstance(node, LeafBranch):
            out.extend(["OL", node.key, node.value])
        else:
            out.extend(
                ["O", node.depth, node.region,
                 node.left.get_hash(), node.right.get_hash()]
            )

    def _build(self, items, lo, hi, frozen, out):
        """Fresh subtree over sorted items; every junction is new. `frozen`
        maps a key to its pre-existing LeafBranch (leaf-merge case)."""
        if hi - lo == 1:
            k, v = items[lo]
            if k in frozen:
                out.extend(["OL", k, v])
                return frozen[k]
            out.append("L")
            return LeafBranch(k, v)
        split = first_divergence(items[lo][0], items[hi - 1][0])
        region = prefix(items[lo][0], split)
        mid = self._partition(items, lo, hi, split)
        ln = self._build(items, lo, mid, frozen, out)
        rn = self._build(items, mid, hi, frozen, out)
        out.extend(["N", split])
        return NodeBranch(split, region, ln, rn)

    def _partition(self, items, lo, hi, depth):
        while lo < hi:
            mid = (lo + hi) // 2
            if key_bit(items[mid][0], depth):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _insert(self, node, items, lo, hi, parent_new, out):
        if lo == hi:
            self._emit_preserved(node, parent_new, out)
            return node

        if node is None:
            return self._build(items, lo, hi, {}, out)

        if isinstance(node, LeafBranch):
            # Keys are pre-filtered distinct from node.key: merge and rebuild.
            merged = sorted(items[lo:hi] + [(node.key, node.value)])
            return self._build(merged, 0, len(merged), {node.key: node}, out)

        # Do the batch extremes diverge from this junction's region?
        d_div = node.depth
        for probe in (items[lo][0], items[hi - 1][0]):
            x = prefix(probe, node.depth) ^ node.region
            if x:
                d_div = min(d_div, node.depth - x.bit_length())
        if d_div < node.depth:
            return self._split_edge(node, items, lo, hi, d_div, out)

        mid = self._partition(items, lo, hi, node.depth)
        new_left = self._insert(node.left, items, lo, mid, False, out)
        new_right = self._insert(node.right, items, mid, hi, False, out)
        out.extend(["N", node.depth])
        node.left, node.right, node._hash = new_left, new_right, None
        return node

    def _split_edge(self, node, items, lo, hi, d_div, out):
        """New junction at depth d_div above `node` (canonical edge split)."""
        region = node.region >> (node.depth - d_div)
        node_side = (node.region >> (node.depth - d_div - 1)) & 1
        mid = self._partition(items, lo, hi, d_div)
        if node_side == 0:
            ln = self._insert(node, items, lo, mid, True, out)
            rn = self._build(items, mid, hi, {}, out)
        else:
            ln = self._build(items, lo, mid, {}, out)
            rn = self._insert(node, items, mid, hi, True, out)
        out.extend(["N", d_div])
        return NodeBranch(d_div, region, ln, rn)

    # -- queries ---------------------------------------------------------------

    def _find_leaf(self, key):
        node = self.root
        while node is not None:
            if isinstance(node, LeafBranch):
                return node if node.key == key else None
            if prefix(key, node.depth) != node.region:
                return None
            node = node.right if key_bit(key, node.depth) else node.left
        return None

    def inclusion_cert(self, key):
        """Root-to-leaf walk; bitmap of junction depths + sibling hashes.
        Regions are NOT included -- the verifier derives them from the key."""
        node = self.root
        bitmap = 0
        siblings = []
        while isinstance(node, NodeBranch):
            if prefix(key, node.depth) != node.region:
                return None
            bitmap |= 1 << node.depth
            if key_bit(key, node.depth):
                siblings.append(node.left.get_hash())
                node = node.right
            else:
                siblings.append(node.right.get_hash())
                node = node.left
        if not isinstance(node, LeafBranch) or node.key != key:
            return None
        return {"bitmap": bitmap, "siblings": siblings}

    def non_inclusion_witness(self, key):
        """Chain of openings along the key-directed descent, ending at the
        first node whose region is not a prefix of the key (or a leaf with
        a different key). None root is a witness for every key."""
        if self.root is None:
            return []
        chain = []
        node = self.root
        while True:
            if isinstance(node, LeafBranch):
                chain.append(("LEAF", node.key, node.value))
                return chain if node.key != key else None
            chain.append(
                ("NODE", node.depth, node.region,
                 node.left.get_hash(), node.right.get_hash())
            )
            if prefix(key, node.depth) != node.region:
                return chain  # divergence junction: terminal
            node = node.right if key_bit(key, node.depth) else node.left


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


def verify_consistency(proof, old_root, new_root, batch, _=None):
    """Compact v6 stack machine.

    N carries only a depth. Its region is derived from authenticated child
    advice. Any accepting proof expands to an accepting v6 proof by inserting
    the derived region at each N, so v6's append-only and canonicality results
    apply unchanged.
    """
    if not batch:
        return old_root == new_root and not proof

    try:
        sorted_batch = sorted(batch)
    except TypeError:
        return False
    for i in range(1, len(sorted_batch)):  # strictly increasing keys
        if sorted_batch[i - 1][0] >= sorted_batch[i][0]:
            return False

    stack = []
    pi = 0
    bi = 0
    try:
        while pi < len(proof):
            tag = proof[pi]
            pi += 1

            if tag == "S":
                h = proof[pi]; pi += 1
                if not isinstance(h, bytes) or len(h) != 32:
                    return False
                stack.append((h, h, None, None))

            elif tag == "O":
                d, p, lh, rh = proof[pi:pi + 4]; pi += 4
                if not isinstance(d, int) or not 0 <= d < KEY_BITS:
                    return False
                if not isinstance(p, int) or not 0 <= p < (1 << d if d else 1):
                    return False
                if not isinstance(lh, bytes) or len(lh) != 32:
                    return False
                if not isinstance(rh, bytes) or len(rh) != 32:
                    return False
                h = hash_node(d, p, lh, rh)
                stack.append((h, h, d, p))

            elif tag == "OL":
                k, v = proof[pi:pi + 2]; pi += 2
                if not isinstance(k, int) or not 0 <= k < (1 << KEY_BITS):
                    return False
                if not isinstance(v, bytes):
                    return False
                h = hash_leaf(k, v)
                stack.append((h, h, KEY_BITS, k))

            elif tag == "L":
                k, v = sorted_batch[bi]; bi += 1
                if not isinstance(k, int) or not 0 <= k < (1 << KEY_BITS):
                    return False
                if not isinstance(v, bytes):
                    return False
                stack.append((None, hash_leaf(k, v), KEY_BITS, k))

            elif tag == "N":
                d = proof[pi]; pi += 1
                if not isinstance(d, int) or not 0 <= d < KEY_BITS:
                    return False
                rh0, rh1, rdelta, rrho = stack.pop()
                lh0, lh1, ldelta, lrho = stack.pop()

                # Derive p from every available child descriptor. Descriptors
                # must agree and each described edge must be coherent.
                p = None
                for delta, rho, side in (
                    (ldelta, lrho, 0),
                    (rdelta, rrho, 1),
                ):
                    if delta is None:
                        continue
                    if delta <= d:
                        return False
                    candidate = rho >> (delta - d)
                    if ((rho >> (delta - d - 1)) & 1) != side:
                        return False
                    if p is not None and p != candidate:
                        return False
                    p = candidate
                if p is None:
                    return False

                is_new = lh0 is None or rh0 is None
                if is_new and (ldelta is None or rdelta is None):
                    return False

                # four-way pre-state rule
                if lh0 is None and rh0 is None:
                    h0 = None
                elif lh0 is None:
                    h0 = rh0                    # pass-through
                elif rh0 is None:
                    h0 = lh0                    # pass-through
                else:
                    h0 = hash_node(d, p, lh0, rh0)

                h1 = hash_node(d, p, lh1, rh1)
                stack.append((h0, h1, d, p))

            else:
                return False
    except (IndexError, OverflowError, ValueError, TypeError):
        return False

    if pi != len(proof) or bi != len(sorted_batch) or len(stack) != 1:
        return False
    h0, h1 = stack[0][0], stack[0][1]
    return h0 == old_root and h1 == new_root


def verify_inclusion(cert, root_hash, key, value):
    """Regions recomputed from the key: prefix(key, d) at each junction."""
    bitmap = cert["bitmap"]
    siblings = list(cert["siblings"])

    h = hash_leaf(key, value)
    j = len(siblings)
    for d in range(KEY_BITS):          # deepest junction combines first
        dd = KEY_BITS - 1 - d
        if not (bitmap >> dd) & 1:
            continue
        j -= 1
        if j < 0:
            return False
        s = siblings[j]
        if key_bit(key, dd):
            h = hash_node(dd, prefix(key, dd), s, h)
        else:
            h = hash_node(dd, prefix(key, dd), h, s)
    return j == 0 and h == root_hash


def verify_non_inclusion(chain, root_hash, key):
    """Openings chain from the root along key-directed descent; terminal is
    a junction whose region is not a prefix of the key, or a foreign leaf.
    Empty chain is valid iff the tree is empty."""
    if root_hash is None:
        return chain == []
    if not chain:
        return False
    expected = root_hash
    last = len(chain) - 1
    for i, item in enumerate(chain):
        if item[0] == "LEAF":
            _, k, v = item
            return i == last and hash_leaf(k, v) == expected and k != key
        _, d, p, lh, rh = item
        if hash_node(d, p, lh, rh) != expected:
            return False
        if prefix(key, d) != p:
            return i == last          # divergence junction: valid terminal
        expected = rh if key_bit(key, d) else lh
    return False                       # chain ended without a terminal


# ---------------------------------------------------------------------------
# Self-tests: round-trip, the v3 attack (must be rejected), tampering
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    rng = random.Random(6)

    # -- valid multi-round history ----------------------------------------
    tree = SparseMerkleTree()
    recorded = {}
    root = tree.get_root()
    assert root is None                      # genesis
    for rnd in range(8):
        batch = [
            (rng.getrandbits(KEY_BITS), b"v%d" % rng.getrandbits(30))
            for _ in range(rng.choice([1, 3, 17, 200]))
        ]
        applied, proof = tree.batch_insert(batch)
        new_root = tree.get_root()
        assert verify_consistency(proof, root, new_root, applied), rnd
        recorded.update(dict(applied))
        root = new_root
    for k, v in recorded.items():
        assert verify_inclusion(tree.inclusion_cert(k), root, k, v)
    for _ in range(50):
        k = rng.getrandbits(KEY_BITS)
        if k not in recorded:
            assert verify_non_inclusion(tree.non_inclusion_witness(k), root, k)
    print("valid history: %d keys, all proofs verify" % len(recorded))

    # -- the v3 shadow-insertion attack is rejected -------------------------
    k = next(iter(recorded))
    v_prime = b"equivocation"
    d_star = next(d for d in range(KEY_BITS) if key_bit(k, d) == 1)
    fake_root = hash_node(d_star, prefix(k, d_star), root, hash_leaf(k, v_prime))
    # (a) opaque S under the new junction: no advice -> rejected
    attack_a = ["S", root, "L", "N", d_star]
    assert not verify_consistency(attack_a, root, fake_root, [(k, v_prime)])
    # (b) opened O under the new junction: edge coherence fails, because the
    #     preserved subtree's region is a prefix of k while the junction
    #     places k's leaf on the other side
    rn = tree.root
    attack_b = ["O", rn.depth, rn.region, rn.left.get_hash(), rn.right.get_hash(),
                "L", "N", d_star]
    assert not verify_consistency(attack_b, root, fake_root, [(k, v_prime)])
    print("shadow insertion (v3 attack): rejected")

    # -- re-recording a present key is rejected ------------------------------
    # honest path: dedup skips it
    applied, proof = tree.batch_insert([(k, v_prime)])
    assert applied == [] and proof == []
    # adversarial path: any placement violates edge coherence somewhere
    # (Theorem 1: no accepting run exists); spot-check a few crafted spots
    for d_try in range(0, 12):
        p_try = prefix(k, d_try)
        forged = hash_node(d_try, p_try, hash_leaf(k, v_prime), root)
        stream = ["L", "S", root, "N", d_try]
        assert not verify_consistency(stream, root, forged, [(k, v_prime)])
    print("re-recording a present key: rejected")

    # -- tamper checks --------------------------------------------------------
    t2 = SparseMerkleTree()
    a1, p1 = t2.batch_insert([(rng.getrandbits(KEY_BITS), b"a") for _ in range(64)])
    r1 = t2.get_root()
    a2, p2 = t2.batch_insert([(rng.getrandbits(KEY_BITS), b"b") for _ in range(64)])
    r2 = t2.get_root()
    assert verify_consistency(p2, r1, r2, a2)
    assert not verify_consistency(p2, r1, r2, a2[:-1])            # dropped item
    assert not verify_consistency(p2, r1, r2, a2 + [a2[0]])       # duplicate key
    bad = list(p2)
    for i, op in enumerate(bad):
        if op == "N":
            bad[i + 1] = (bad[i + 1] + 1) % KEY_BITS              # shift a depth
            break
    assert not verify_consistency(bad, r1, r2, a2)
    bad = list(p2)
    changed = False
    for i, op in enumerate(bad):
        if op == "O":
            bad[i + 2] ^= 1                       # flip authenticated old region
            changed = True
            break
    assert changed
    assert not verify_consistency(bad, r1, r2, a2)
    assert not verify_consistency(p2, r1, r2, [(k_, b"x") for k_, _ in a2])  # values
    print("tamper checks: all rejected")

    print("rsmt6a: OK")
