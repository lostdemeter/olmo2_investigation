# Chapter 3: Scaling the Substrate via LLM Collaboration

With the constructive proof established at toy scale, the T5 series addressed the engineering question: can the φ-geometric substrate be scaled automatically? The answer is a layered succession of experiments (T5A through T5D), each adding a new capability, culminating in a fully autonomous self-improving loop.

## 3.1 Multi-Clause Agreement and Positional Axes (T5A)

The first challenge was extending grammatical capabilities beyond single-clause subject-verb agreement. The T5A experiment demonstrated that the geometric model could handle two-clause sentences of the form `[SUBJ_1, OP_1, AND, SUBJ_2, OP_2]`, ensuring each verb agreed with its respective subject without cross-talk.

A total of 576 prompt combinations × 2 operations each = **1,152 agreement tests** were run under three conditions:

| Condition | prev-only | clause-id | Result |
|-----------|-----------|-----------|--------|
| A (baseline) | ✓ | ✗ | 1152/1152 (100%) |
| B (stress) | ✗ | ✗ | 864/1152 (75.0%) |
| C (clause-id fix) | ✗ | ✓ | 1152/1152 (100%) |

The critical finding is in Condition B: the 75.0% pass rate is not an empirical accident — it is the **closed-form prediction** derived from the vocabulary's NUMBER distribution and PyTorch's tie-breaking rule. When the model fails, it fails by the rules of the substrate, not by gradient-descent accidents. This is a strong signature of the constructive theory.

Condition C introduced the **CLAUSE_ID axis** — the first axis in the project that is not lexical or morphological but positional, encoding where each token sits relative to clause boundaries. This is the constructive analogue of a learned position embedding, and it composes with the rest of the substrate by the same rules (sign/magnitude/any-state per block, K-match by sign product).

## 3.2 Multi-Lexeme Conjugation (T5B)

The next step scaled from a single verb ("to be") to multiple verbs (lexemes). This was accomplished through two key architectural insights:

1. **Separation of Morphology and Lexicalisation**: The vocabulary was defined at the level of abstract morphological "cells" (e.g., a unique token for `(LEXEME=BE, PERSON=1st, NUMBER=sg, TENSE=present)`). A separate, simple lookup table mapped these cells to their surface English forms. This kept the geometric space clean and orthogonal.

2. **Content-Based Attention Heads**: Multiple attention heads, each configured to look for a unique flag on a specific input token, gathered the necessary information from the input. One head attended to the SUBJECT token to extract number and person information, while another attended to the VERB_STEM token to identify the lexeme.

This proved that the geometric substrate could be extended with new semantic axes (like LEXEME) and that multiple attention heads could work in parallel to compose complex outputs — all without requiring positional masks.

## 3.3 LLM-Powered Labeling at Scale (T5C)

The bottleneck of manually labeling vocabulary was overcome by establishing a robust pipeline using a larger LLM (`gpt-oss:20b`). The process involved three stages:

1. **LLM-based classification** with self-consistency checks
2. A set of hand-curated **linguistic constraints** (e.g., boy and girl must differ on the gender axis)
3. A **deterministic reconciliation algorithm** that repaired any violations in the LLM's output

The pipeline successfully produced a clean, 56-word labeled vocabulary that passed a full 18/18 analogy battery, proving that the labeling process could be reliably automated.

## 3.4 The Self-Improving Auto-Loop (T5D)

The culmination of the T5 series is a closed automation loop integrating three actors:

- **Builder** (the geometric model): a pure function `state → (E, vocab, tok, heads)` that rebuilds the constructed transformer from the current axes and labels every iteration
- **Labeller** (`gpt-oss:20b`): classifies words on the current axis schema, proposes new axes when collision groups are large
- **Challenger** (`llama3.1:8b`): proposes new vocabulary words near the existing semantic neighborhood

A refinement controller sits in the middle, runs a test battery (self-decode + analogy), picks one action per iteration (expand vocab or propose axis), applies it, re-runs the battery, and commits or rolls back based on a weighted score.

### Phase 1: Demo Run (10 iterations)

The demo validated the loop architecture: 10 iterations with no human input grew the vocabulary from 56 to 76 words, and `gpt-oss` autonomously discovered the new axis `time_scale` (durativity, distinguishing geological objects like mountains from artifacts like books and tables). Self-decode rose from 26 to 43 (+65%), largest collision dropped from 7 to 5, while the analogy battery held at 16–18/18.

### Phase 2: Scaling to Saturation (22,432 iterations)

An overnight run of ~17.5 hours under a watchdog script pushed the loop to saturation:

| Metric | Before (T5C) | After (T5D v2) |
|--------|-------------|----------------|
| Vocabulary | 56 | **1,438** |
| Semantic axes | 6 | **13** |
| Distinguishable cells (4^axes) | 4,096 | **67 million** |
| Self-decode | 26/56 (46%) | **1,204/1,438 (84%)** |
| Analogy battery | 18/18 | **16/18** |
| Largest collision | 7 | **2** |
| Human inputs | seeds + 6 axes | seeds + 6 axes (unchanged) |

Seven semantic axes were discovered autonomously by `gpt-oss`, in order:

| Axis | What it captures |
|------|-----------------|
| `time_scale` | durativity / natural vs. artefactual |
| `social_role_presence` | discourse-deictic vs. role-denoting |
| `emotional_valence` | affective polarity of abstract concepts |
| `personhood` | pronoun/person vs. animal |
| `initiative_authority` | leadership / agency strength |
| `group_affiliation_strength` | individual vs. collective referent |
| `status_origin` | inherited vs. acquired status |

These are conceptually independent of the seed-6 axes (gender, number, animacy, age, royalty, family) and of each other. Every one was proposed by `gpt-oss:20b` with no prompting beyond "propose ONE new axis that distinguishes these words."

### Engineering Lessons from Saturation

Three generalizable findings emerged from the overnight run:

1. **Hallucinated-compound saturation**: Without a dictionary filter, `llama3.1:8b` invents morphologically-plausible-but-nonexistent compounds. A one-line dictionary filter eliminated this failure mode.

2. **Adaptive triggers > fixed thresholds**: The original axis-proposal trigger (fixed collision threshold) stopped firing when the substrate became the bottleneck. An adaptive replacement (`growth==0 AND collision >= 3`) attacked the substrate only when vocabulary growth stalled.

3. **In-loop saturation detection**: Adding a rolling score-window flatness criterion turned an unbounded run into a self-terminating experiment with a clear exit point.

## 3.5 Inductive Bias and Its Limits (F-BL-T1)

In parallel with the constructive experiments, the Bloch-OLMo architecture tested whether geometric structure could emerge purely from architectural inductive bias. The results provide an important counterpoint:

- **Success at the surface**: The model trained well, and BlockRMSNorm successfully enforced its geometric constraint at the embedding layer, with 68% of tested semantic opposites aligning correctly — just shy of the 75% target.
- **Failure at depth**: The geometric structure rapidly dissolved as it propagated through the model's layers.

This demonstrates that while architectural inductive bias can encourage local geometric structure at the input layer, it is insufficient to ensure that geometry is preserved or utilized by deeper computational machinery. The geometry must be more explicitly woven into the computational fabric — as it is in the φ-computer rewrite (Chapter 1) and the constructive auto-loop (this chapter).
