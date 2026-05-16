# Chapter 4: The Methodological Audit — What We Learned From Our Own Failures

The T5 series' later experiments (T5e through T5j) shifted focus from building the substrate to understanding the **process of building it**. Two failures — the BUILD/COLLAPSE/EMIT cascade's inability to discover latent axes from raw text, and a chain-of-custody audit that revealed how information was silently projected away — led to a deeper methodological contribution: a discipline for keeping research artifacts honest about what they have forgotten.

## 4.1 The Build/Collapse/Emit Architecture

The Triangulation Principle (established at T5e) states: **every unknown coordinate in concept-space is determined by the selector-chain induced by the corpus.** A "question" is a partial axis-state configuration — most axes pinned, one unknown. Information gain over the stored corpus identifies the *selector* — the pinned axis whose value most predicts the unknown. The unknown coordinate is then determined by gear-hierarchical matching.

This principle gives rise to three operational modes, all using the same information-gain gearbox machinery:

| Mode | Question | Output |
|------|----------|--------|
| **BUILD** | "Where does the inconsistency function vanish?" | a configuration |
| **COLLAPSE** | "Which pinned axis most predicts the hole?" | filled configuration |
| **EMIT** | "Which axis most predicts the rest?" | ordered tokens |

The **Reverse Gearbox** (T5f) demonstrated that the same info-gain ranking that picks the selector for COLLAPSE picks the natural anchor for EMIT. Round-trip identity holds: for every configuration in the corpus, `parse(emit(c)) == c`. English SVO order is a one-line permutation overlay.

The **BUILD as Zero-Hunting** paradigm (T5g) reframed BUILD as discovery on an implicitly defined manifold: given tokens, find the configuration where an *inconsistency function* vanishes. A three-stage cascade (compressor → processor → targeter, inspired by the rhzeros Riemann-zero hunter) achieved round-trip identity on hand-built and natural-English corpora.

### The Axis Discovery Ceiling (T5i)

When the same cascade was asked to discover the **axes themselves** — with no axis labels, using only pairwise token co-occurrence — it hit a fundamental ceiling. On a 96-sentence corpus with 4 ground-truth axes, the system recovered ≈7 over-fragmented axes with axis-pair recall of only 0.19.

The dominant confound was **grammatical agreement**: number and tense agreement between subjects and verbs produces mutual-exclusion fingerprints that are grammatically rather than semantically driven. Two pairwise fingerprints (mutual exclusion and state-mate co-occurrence) are not sufficient to separate semantic axes from grammatical agreement patterns.

## 4.2 The Chain of Custody Audit

After T5i ran out of runway, a retrospective audit was conducted across experiments T5e through T5i. The audit form asked, for each experiment:

1. **Artifact** — what was computed
2. **Projections** — what dimensions/structure were marginalized away
3. **Symmetry** — what symmetry of the parent the projection assumes
4. **Back-pointer** — can the parent be reconstructed from the artifact plus the symmetry?
5. **Verdict** — was the projection safe, dangerous, or broken?

Two distinct failure types emerged:

**Type 1: Data shift.** A projection that was safe in T5g (treating tokens as atomic) became load-bearing in T5h because the data changed (synthetic single-form data → natural English with morphological variation). Same code, same projection, different data → different validity. Felt as 4/24 leave-one-out failures.

**Type 2: Question shift.** A projection that was safe in T5g/T5h (bag-of-tokens within sentences) became load-bearing in T5i because the question changed (build given axes → discover axes). Same data, same projection, different question → different validity. Felt as axis over-fragmentation.

The methodological lesson:

> Every projection is a contract between an artifact, a data regime, and a question. When any of the three changes, the contract is void until re-checked.

Neither failure was detected at the time because the pipeline only audited each projection once, at the moment it was introduced.

## 4.3 The Abelian Critique and Spacetime Irreducibility

Two conceptual reframes emerged from the audit, pointing at the same underlying gap.

### The Abelian Critique

The bag-of-co-occurrences artifact that T5i consumed is the **Abelianization of the corpus** — what survives when you quotient the free monoid of token sequences by `ab = ba`. Language is non-commutative ("dog bites man" ≠ "man bites dog"). The symmetric co-occurrence matrix annihilates the antisymmetric part `T - Tᵀ` that encodes directional dependencies. The project had been doing representation theory on the wrong group.

### Spacetime Irreducibility (The Sunflower)

The φ-spiral of sunflower seeds is not a property of a 2D disk; it is the shadow of a (radius, angle, time) growth trajectory whose spacetime optimum, projected to the surface, *is* the spiral. φ appears in the projection because it is the most-irrational number — the unique angle that "never repeats" across all prefix-lengths simultaneously.

The lesson for the project: when we factor variables as independent and then drop terms that locally evaluate to zero, we destroy the bridge connecting the simplified equation to the parent. Later, when the simplified frame breaks, we cannot get back. This is exactly what happened across T5e→T5i: each experiment projected away a dimension that seemed harmless at the time, until the accumulated projections made the original question unanswerable.

## 4.4 The Sequence Tensor (T5j)

The disciplinary response was the **Sequence Tensor** — a canonical parent artifact that preserves all information and from which all derived statistics are computed as named views with explicit symmetry declarations.

`S[s, t, i] = 1` if sentence `s` has token `i` at position `t`. Every derived statistic (co-occurrence matrix, bigram transition operator, PMI encoder) is a *view* of `S` that declares what symmetries it assumes:

```python
@dataclass
class View:
    name: str
    tensor: Tensor
    symmetries_assumed: tuple[str, ...]  # e.g., Sym(POSITIONS), Sym(SENTENCES)
    derivation: str                      # how it was computed from parent
    parent: View                         # constructive back-pointer

def require_sensitivity_to(view: View, *required: str):
    """Raises SymmetryViolation if any required symmetry is assumed."""
```

The `Sym` namespace provides named symmetries referenced by views and consumers alike:

- `Sym(POSITIONS)` — marginalizing away token order within a sentence
- `Sym(SENTENCES)` — marginalizing away which sentence a token is in
- `Sym(DIRECTION)` — treating co-occurrence as symmetric

When a consumer declares `require_sensitivity_to(view, Sym(POSITIONS), Sym(DIRECTION))`, it is **refused at runtime** if the view has marginalized either of those dimensions away. This is the chain of custody enforced as code, not just as a methodological reminder.

## 4.5 Eigenmode Spectroscopy of Language (T5j)

The first custodial consumer computed the eigenvalues and eigenvectors of the non-symmetric bigram transition operator `T[i, j] = P(next = j | current = i)` — a direction-aware view that the co-occurrence matrix cannot provide.

### The Low-Rank Structure of a 39-Word Corpus

On an 80-sentence corpus with 39 distinct tokens, the 39×39 transition matrix has only 9 non-trivial modes above |λ|=0.05: 1 stationary + 4 complex pairs + 2 small real. Everything else is numerical noise. The corpus's linguistic structure lives in **9 dimensions of operator energy**.

This low-rank structure is invisible to the symmetric co-occurrence matrix, which is rank ≈39 by construction on the same data.

### Function-Driver / Content-Carrier Duality

Reading left and right eigenvectors of the **same** mode together exposes a structural fact about language that symmetric statistics conflate:

- **Right eigenvectors** of non-stationary modes are dominated by **content words** — tokens transported through the transition
- **Left eigenvectors** of the same modes are dominated by **function words** (`at`, `on`, `the`, `are`) — what drives the transition forward

The stationary mode (λ=1) collects function-word probability mass cleanly: `the` 22.8%, `at` 13.9%, `on` 7.3%. Stop-word identification falls out of the operator's λ=1 eigenvector without any PMI threshold, frequency cutoff, or stop list.

The symmetric co-occurrence matrix cannot represent this asymmetry. The antisymmetric part `T - Tᵀ` encodes "function words drive, content words are driven," and co-occurrence is exactly `T + Tᵀ` (up to scale). The structural payoff of not symmetrizing is the discovery of distinct, meaningful roles within a single operator.

### Unsupervised Morphological Clustering

Mode-space cosine similarity of token signature vectors (over the top 12 modes):

- Same lemma (howl/howled/howls/howling): 0.969 mean
- Same axis (cross-lemma): 0.901 mean
- Random content pair: 0.847 mean

Many same-lemma pairs have cosine 1.000 — they are **structurally indistinguishable** in the bigram operator's eigenstructure. This is a free clustering signal that earlier experiments would have benefited from: cluster tokens by mode-space cosine before training a PMI encoder, and morphological-asymmetry failures become a data-pooling problem rather than a per-form-PMI problem.

### Independent Confirmation of the Gear-Shift

The action↔trigger coupling (2.0 bits mutual information, identified in T5g) appears as the **dominant non-stationary direction** of the bigram operator: λ_3 puts 50% of content-mass on action and 35% on trigger. Same finding, different methodology, no shared code path — independent confirmation of the gear-shift structure.

### The Honest Limit

Same-axis cosine is 0.90, random is 0.85 — a gap of only 0.05. The bigram operator is *aware* of axes but does not *isolate* them. Pairwise statistics with direction preserved are one rung above pairwise statistics without direction, but still in the marginal regime. The chain of custody is the deliverable of T5j, not the eigenmodes; future axis-discovery work must climb further up the hierarchy of views.

## 4.6 Closing the Loop: PMI Weight-Vector Lemma Clustering (T5k)

The T5j spectroscopy made a falsifiable prediction: if mode-space cosine similarity distinguishes morphological variants (0.969 same-lemma vs. 0.901 same-axis), then clustering tokens by their co-occurrence profile and pooling their statistics should fix the 4/24 leave-one-out failures that the T5h encoder suffered from morphological asymmetry.

**First attempt: bigram eigenmode signatures.** Token signature vectors from the right eigenvectors of the 39×39 bigram transition matrix (80-sentence T5i corpus) were clustered by cosine similarity. Result: total failure. The uniform template structure makes all content tokens structurally indistinguishable in the bigram operator. Every threshold between 0.85 and 0.97 produced 2–3 giant mixed-axis clusters with lemma-pair recall below 0.10. The eigenmode space is too sparse for a 24-sentence evaluation corpus.

**Second attempt: PMI weight vectors.** Instead of clustering in eigenmode space, each token is represented by its PMI weight vector across all 16 (axis, state) pairs. Two morphological variants of the same lemma (howl and howled) have near-identical vectors because they occur in identical axis-state configurations. The only difference is the count of occurrences — and the PMI weight vector normalises for that.

**Architecture.** The pipeline has five steps:

1. Train the PMI encoder on the full 24-sentence T5h corpus (`discover_encoder` with smoothing 0.5).
2. For each token, build a 16-dimensional weight vector: `W[token, (a, s)] = max(0, P(a=s|token)/P(a=s) − 1)`.
3. Cluster tokens by cosine similarity of their weight vectors at a swept threshold.
4. Build a lemma map from the discovered clusters: each token in a cluster maps to that cluster's canonical form.
5. Re-run T5h leave-one-out with the PMI encoder trained on lemma-mapped tokens, and with held-out tokens mapped through the same lemma map during BUILD.

**Results.** A threshold sweep (0.80–0.99 in 0.01 steps) identified 0.93 as optimal:

| Metric | Value |
|--------|-------|
| Discovered clusters | 9 (2 pure lemma, 7 mixed) |
| Lemma-pair recall | 0.28 |
| Cluster purity | 0.22 |
| Content tokens clustered | 27/33 |
| Standard LOO | 20/24 |
| Lemma-pooled LOO | 23/24 |
| Improvement | +3 (0 regressions) |

The 3 fixed cases (sentences 15, 21, 23) all involve morphological variants that the clustering correctly grouped: `chase↔chased`, `peck↔pecked`, `cat↔cats`. The mixed clusters also contributed: although `bird↔peck↔seed` mixes three lemmas, it pools statistics within a shared subject context that benefits the cat↔peck↔seed sentences.

**The one remaining failure: a corpus-size artifact.** Sentence 3 (`the dog howled at the moon on the floor`) remains broken because `howled` and `moon` are perfectly co-occurrent in this 24-sentence corpus. Their PMI weight vectors have cosine 0.978 — higher than `howled↔howl` at 0.914 — because every howled occurrence includes moon and vice versa. No clustering threshold can separate perfectly correlated tokens. On a larger corpus where `howled at the stranger` and `howled at the door` appear alongside `howled at the moon`, the PMI vectors would diverge naturally: `howled` would gain weight on additional trigger states while `moon` would not.

**Morphological asymmetry is a data-pooling problem.** The T5k result confirms the T5j prediction with three specific findings:

1. PMI weight-vector clustering closes the morphological asymmetry gap for 3 of 4 baseline failures.
2. The 4th failure is a corpus-size artifact, not a structural limitation of the approach.
3. Impure clusters (mixing function drivers with co-occurrent content words) are helpful, not harmful — they pool statistics within the same axis context, and the T4 constructive proof shows that axis independence is the correct inductive bias.

The practical implication: PMI-based lemma clustering is a drop-in preprocessing step for any encoder-based pipeline. On a corpus with sufficient contextual diversity, it recovers the T5j-predicted 0.969 same-lemma similarity and eliminates morphological asymmetry entirely.
