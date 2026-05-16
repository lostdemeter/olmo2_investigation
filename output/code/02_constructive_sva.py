"""
02_constructive_sva.py — Simplified subject-verb agreement on a constructed transformer.

Demonstrates:
  - A 1-layer, 1-head constructed transformer that generates "to be" forms
  - Attention routes NUMBER from subject to operator position
  - LM-head dot products produce exact integer margins

This is the canonical example from Chapter 2.4 (T4-SVA).

Run: python3 output/code/02_constructive_sva.py
"""

import numpy as np

# ── Vocabulary: 12 subjects + 4 verb forms + 2 operators ─────
# Each token is encoded on 3 axes × 6 dims = 18-dim residual

AXES = ["NUMBER", "LEX_CLASS", "TENSE"]
BLOCK = 6  # dims per axis (sign, mag, any_state, flag1, flag2, flag3)

# Helper: build a block from (sign, mag) + any_state
def axis_block(sign, mag, any_state=1.0):
    v = np.zeros(BLOCK)
    v[0] = sign
    v[1] = mag
    v[2] = any_state
    return v

# Helper: build a full residual by concatenating axis blocks
def residual(axis_vals):
    """axis_vals: dict mapping axis name -> (sign, mag) or None for unused"""
    blocks = []
    for ax in AXES:
        if ax in axis_vals and axis_vals[ax] is not None:
            s, m = axis_vals[ax]
            blocks.append(axis_block(s, m))
        else:
            blocks.append(np.zeros(BLOCK))
    return np.concatenate(blocks)

# ── Subjects (12) ──
SUBJECTS = {
    "boy":    residual({"NUMBER": (+1, +1), "LEX_CLASS": (+1, +1)}),
    "girl":   residual({"NUMBER": (+1, +1), "LEX_CLASS": (+1, +1)}),
    "dog":    residual({"NUMBER": (+1, +1), "LEX_CLASS": (+1, +1)}),
    "cat":    residual({"NUMBER": (+1, +1), "LEX_CLASS": (+1, +1)}),
    "wolf":   residual({"NUMBER": (+1, +1), "LEX_CLASS": (+1, +1)}),
    "bird":   residual({"NUMBER": (+1, +1), "LEX_CLASS": (+1, +1)}),
    "boys":   residual({"NUMBER": (-1, +1), "LEX_CLASS": (+1, +1)}),
    "girls":  residual({"NUMBER": (-1, +1), "LEX_CLASS": (+1, +1)}),
    "dogs":   residual({"NUMBER": (-1, +1), "LEX_CLASS": (+1, +1)}),
    "cats":   residual({"NUMBER": (-1, +1), "LEX_CLASS": (+1, +1)}),
    "wolves": residual({"NUMBER": (-1, +1), "LEX_CLASS": (+1, +1)}),
    "birds":  residual({"NUMBER": (-1, +1), "LEX_CLASS": (+1, +1)}),
}

# ── Verb forms (4) ──
VERBS = {
    "is":   residual({"NUMBER": (+1, +1), "LEX_CLASS": (-1, +1), "TENSE": (+1, +1)}),
    "are":  residual({"NUMBER": (-1, +1), "LEX_CLASS": (-1, +1), "TENSE": (+1, +1)}),
    "was":  residual({"NUMBER": (+1, +1), "LEX_CLASS": (-1, +1), "TENSE": (-1, +1)}),
    "were": residual({"NUMBER": (-1, +1), "LEX_CLASS": (-1, +1), "TENSE": (-1, +1)}),
}

# ── Operators (2): carry LEX_CLASS=-1, TENSE=±1, and an agree-flag ──
# The agree-flag is a non-zero value in the NUMBER block's flag dim
AGREE_PRESENT = residual({"NUMBER": None, "LEX_CLASS": (-1, +1), "TENSE": (+1, +1)})
# Add the agree flag in NUMBER block dim 5 (the 6th dim = flag3)
AGREE_PRESENT[5] = 1.0

AGREE_PAST = residual({"NUMBER": None, "LEX_CLASS": (-1, +1), "TENSE": (-1, +1)})
AGREE_PAST[5] = 1.0

# ── Attention head ──
# Q matches agree-flag (dim 5), K matches NUMBER any_state (dim 2)
# V copies NUMBER block, W_O writes it into current position
# Simplified: we just route NUMBER from subject to op position

def attend(subject_residual, op_residual):
    """Copy the NUMBER block (dims 0-5) from subject to op position."""
    result = op_residual.copy()
    result[0:6] = subject_residual[0:6]  # NUMBER block routed
    return result

# ── LM head: dot product against all verb forms ──
VERB_NAMES = list(VERBS.keys())
VERB_MATRIX = np.stack([VERBS[v] for v in VERB_NAMES], axis=0)  # (4, 18)

def decode(residual_after_attention):
    """Return logits and the decoded verb."""
    logits = VERB_MATRIX @ residual_after_attention
    idx = np.argmax(logits)
    return logits, VERB_NAMES[idx]

# ── Test all 24 cases ──
print("=" * 65)
print("Constructive SVA: 12 subjects × 2 tenses")
print("=" * 65)
TENSES = [("present", AGREE_PRESENT), ("past", AGREE_PAST)]

pass_count = 0
failures = []

for subj_name, subj_res in SUBJECTS.items():
    for tense_name, op_res in TENSES:
        # Forward pass
        after_attn = attend(subj_res, op_res)
        logits, predicted = decode(after_attn)

        # Ground truth
        is_singular = (subj_res[0] == +1)  # sign dim of NUMBER block
        is_present = (tense_name == "present")
        if is_singular and is_present:
            expected = "is"
        elif not is_singular and is_present:
            expected = "are"
        elif is_singular and not is_present:
            expected = "was"
        else:
            expected = "were"

        # Compute margins
        sorted_logits = np.sort(logits)[::-1]
        margin = sorted_logits[0] - sorted_logits[1]

        ok = predicted == expected
        pass_count += ok
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append((subj_name, tense_name, predicted, expected))
        print(f"  {subj_name:>6s} {tense_name:<8s} → {predicted:>4s} "
              f"(expected {expected:>4s}) "
              f"margin={margin:+.0f}  [{status}]")

print(f"\nResults: {pass_count}/24 PASS")
if failures:
    print(f"Failures:")
    for s, t, p, e in failures:
        print(f"  {s} {t}: predicted {p}, expected {e}")

# Verify integer-arithmetic property
all_margins = []
for subj_name, subj_res in SUBJECTS.items():
    for tense_name, op_res in TENSES:
        after_attn = attend(subj_res, op_res)
        logits, _ = decode(after_attn)
        sorted_logits = np.sort(logits)[::-1]
        margin = sorted_logits[0] - sorted_logits[1]
        all_margins.append(margin)

print(f"\n── Integer-arithmetic verification ──")
print(f"  All margins are integers: {all(np.equal(np.mod(all_margins, 1), 0))}")
print(f"  Mean margin: {np.mean(all_margins):+.3f}")
print(f"  Min margin:  {np.min(all_margins):+.0f}")
print(f"  Max margin:  {np.max(all_margins):+.0f}")
