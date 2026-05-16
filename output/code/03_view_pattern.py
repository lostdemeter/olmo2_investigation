"""
03_view_pattern.py — The Chain-of-Custody View pattern from T5j.

Demonstrates:
  - The SequenceTensor S[s, t, i] as the canonical parent
  - Named views with explicit symmetry declarations
  - Runtime symmetry violation detection

This is the methodological contribution from Chapter 4.4.

Run: python3 output/code/03_view_pattern.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Named symmetries ──
class Sym:
    POSITIONS = "Sym(POSITIONS)"     # marginalised over token order within sentence
    SENTENCES = "Sym(SENTENCES)"     # marginalised over sentence identity
    DIRECTION = "Sym(DIRECTION)"     # treated co-occurrence as symmetric


# ── SymmetryViolation ──
class SymmetryViolation(Exception):
    """Raised when a consumer requires a symmetry the view has assumed."""
    pass


# ── View: a derived artifact with back-pointer ──
@dataclass
class View:
    name: str
    tensor: np.ndarray
    symmetries_assumed: tuple[str, ...] = ()
    derivation: str = ""
    parent: Optional["View"] = None

    def describe(self):
        syms = ", ".join(self.symmetries_assumed) if self.symmetries_assumed else "none"
        return (f"View('{self.name}', shape={self.tensor.shape}, "
                f"symmetries=[{syms}], parent={self.parent.name if self.parent else 'None'})")


def require_sensitivity_to(view: View, *required: str):
    """Check that none of the required symmetries have been assumed away."""
    for sym in required:
        if sym in view.symmetries_assumed:
            raise SymmetryViolation(
                f"View '{view.name}' assumes {sym}, but consumer requires sensitivity to it."
            )


# ── The parent artifact: SequenceTensor S[s, t, i] ──
# A tiny corpus: 2 sentences, 4 tokens each, 6 distinct word types
CORPUS = [
    ["the", "cat", "chases", "mice"],     # sentence 0
    ["the", "dog", "chases", "cats"],     # sentence 1
]
TOKENS = sorted(set(w for s in CORPUS for w in s))
TOKEN_IDX = {t: i for i, t in enumerate(TOKENS)}

S = np.zeros((len(CORPUS), max(len(s) for s in CORPUS), len(TOKENS)), dtype=np.float32)
for sid, sentence in enumerate(CORPUS):
    for pos, word in enumerate(sentence):
        S[sid, pos, TOKEN_IDX[word]] = 1.0

parent = View(name="S[s,t,i]", tensor=S, derivation="raw corpus")

# ── View 1: Bag of tokens (co-occurrence matrix) ──
# Marginalizes over positions AND sentences
C = np.zeros((len(TOKENS), len(TOKENS)), dtype=np.float32)
for sid in range(S.shape[0]):
    sent_tokens = S[sid, :, :].sum(axis=0)  # sum over positions → presence vector
    for i in range(len(TOKENS)):
        for j in range(len(TOKENS)):
            if sent_tokens[i] > 0 and sent_tokens[j] > 0 and i != j:
                C[i, j] += 1.0
                C[j, i] += 1.0  # symmetric

view_cooccurrence = View(
    "co_occurrence", C,
    symmetries_assumed=(Sym.POSITIONS, Sym.SENTENCES, Sym.DIRECTION),
    derivation="sum_{s,t} S[s,:,i] ⊗ S[s,:,j] (symmetric)",
    parent=parent,
)

# ── View 2: Bigram transition matrix (direction-aware) ──
# Marginalizes over sentences but preserves position AND direction
T = np.zeros((len(TOKENS), len(TOKENS)), dtype=np.float32)
counts = np.zeros(len(TOKENS), dtype=np.float32)
for sid in range(S.shape[0]):
    for t in range(S.shape[1] - 1):
        i = np.argmax(S[sid, t, :])
        j = np.argmax(S[sid, t+1, :])
        T[i, j] += 1.0
        counts[i] += 1.0

# Row-stochastic
for i in range(len(TOKENS)):
    if counts[i] > 0:
        T[i, :] /= counts[i]

view_bigram = View(
    "bigram_transition", T,
    symmetries_assumed=(Sym.SENTENCES,),  # position AND direction preserved
    derivation="sum_s P(next=j | current=i) within each sentence",
    parent=parent,
)

# ── View 3: Token presence (bag, no sentence info) ──
presence = (S.sum(axis=(0, 1)) > 0).astype(np.float32)

view_presence = View(
    "token_presence", presence,
    symmetries_assumed=(Sym.POSITIONS, Sym.SENTENCES, Sym.DIRECTION),
    derivation="sum_{s,t} S[s,t,i] > 0",
    parent=parent,
)

# ── Demonstrations ──
print("=" * 60)
print("The Chain-of-Custody View Pattern (T5j)")
print("=" * 60)

print("\n── Parent artifact ──")
print(f"  S[s, t, i]  shape={S.shape}")
print(f"  Sentences: {len(CORPUS)}, Positions: {S.shape[1]}, Tokens: {len(TOKENS)}")
print(f"  Words: {TOKENS}")

print("\n── Derived views ──")
for v in [view_cooccurrence, view_bigram, view_presence]:
    print(f"  • {v.describe()}")

print("\n── Symmetry enforcement ──")
# This should PASS: bigram preserves position and direction
try:
    require_sensitivity_to(view_bigram, Sym.POSITIONS, Sym.DIRECTION)
    print("  ✓ require_sensitivity_to(bigram, POSITIONS, DIRECTION) → PASS")
except SymmetryViolation as e:
    print(f"  ✗ {e}")

# This should FAIL: co-occurrence has assumed POSITIONS and DIRECTION
try:
    require_sensitivity_to(view_cooccurrence, Sym.POSITIONS, Sym.DIRECTION)
    print("  ✗ require_sensitivity_to(co_occurrence, POSITIONS, DIRECTION) → should have failed!")
except SymmetryViolation as e:
    print(f"  ✓ require_sensitivity_to(co_occurrence, POSITIONS, DIRECTION) → SymmetryViolation")
    print(f"    Reason: {e}")

# ── Bonus: demonstrate the function-driver duality hint ──
print("\n── The bigram operator's asymmetric structure ──")
T_antisym = T - T.T
print("  Antisymmetric part T - T^T (encodes directionality):")
for i, t_i in enumerate(TOKENS):
    for j, t_j in enumerate(TOKENS):
        if abs(T_antisym[i, j]) > 0.01:
            direction = "→" if T_antisym[i, j] > 0 else "←"
            print(f"    {t_i:>5s} {direction} {t_j:<5s}  (asymmetry={T_antisym[i, j]:+.3f})")

print("\nDone. The View pattern enforces chain of custody at runtime.")
