# TruthSpace Volume 2: The Geometric Interpretation of OLMo2

**From Architectural Inductive Bias to φ-Computer Equivalence**

This repository contains the manuscript, experimental code, and figures for the second TruthSpace paper — a geometric reinterpretation of the OLMo2 transformer architecture through the lens of **φ-computation**. The central claim: a standard transformer is not merely *analogous* to a φ-computer — it IS one, up to architectural constraints. Training discovers the geometry; it does not create it.

[**Read the full paper (PDF)**](output/paper.pdf)

## Abstract

This volume presents a geometric interpretation of the OLMo2 transformer through the lens of φ-computation — a framework in which all neural operations are expressed as exact φ-geometric primitives rather than floating-point approximations. The work is organized as a series of constructive experiments (T1–T6) that progressively build and validate this interpretation.

**Part I (Chapters 1–2): Reverse-engineering and reconstruction.** We show that OLMo2's attention, MLP, and normalization layers admit exact φ-algebraic rewrites with byte-level fidelity (max error ≤ 5×10⁻¹⁶). We then construct a transformer from scratch with hand-placed weights — no gradient descent — that achieves byte-exact accuracy on concept arithmetic and subject-verb agreement.

**Part II (Chapters 3–4): Scaling and methodology.** LLM collaboration is used to scale the constructive substrate (Chapter 3). A methodological audit (Chapter 4) identifies the key failure modes encountered and the corrections that converged on the T5 architecture. The T5k experiment achieves 23/24 leave-one-out accuracy on a 24-sentence grammatical corpus using PMI weight-vector clustering — a +3 improvement over the baseline.

**Part III (Chapters 5–6): The Axial State Machine.** The ASM is introduced as a named hidden state with carrier channels evolving by unitary rotation. Three experiments validate it: (E1) RECT-pair rules discovered from synthetic data at 100% accuracy; (E2) the same algorithm achieves 96/96 (100%) on a self-similar φ-quantised corpus when axes are unconfounded; (E3) causal perturbation confirms that BUILD-populated carriers are load-bearing at the readout level (100% specificity).

## Key Claims

1. **φ-Computer Equivalence**: OLMo2's attention, MLP, and normalization layers admit exact φ-algebraic rewrites — the φ-geometric reformulation is not an approximation but an exact algebraic rewrite (max error ≤ 5×10⁻¹⁶).

2. **Constructive Proof**: A transformer can be built from scratch with hand-placed weights, no gradient descent, and byte-exact accuracy on concept arithmetic and grammatical agreement. The T4 series achieves 288/288, 376/376, and 24/24 test cases with every logit margin an exact integer.

3. **Self-Improving Auto-Loop**: The T5D experiment grew a vocabulary from 56 to 1,438 words and discovered 7 new semantic axes autonomously, with no human input beyond the initial seed — validating the "constructive AI" paradigm.

4. **Axial State Machine**: A differentiable architecture with named carrier channels evolving by unitary rotation, where gradient flow learns which rotation to apply as a function of input and context. RECT-pair discovery achieves 100% accuracy on synthetic data and 96/96 on unconfounded φ-quantised naturalistic data.

5. **Chain of Custody**: Every statistical projection destroys information. The Sequence Tensor enforces this discipline at runtime — views declare what symmetries they assume, and consumers are refused if they ask questions of views that have marginalized away required dimensions.

## Experiment Series

| Series | What | Key Result |
|--------|------|------------|
| **T1–T3** | (Volume 1) | φ-computer equivalence framework |
| **T4-NEG0** | 4-state alphabet construction | 288/288, integer margins |
| **T4-MLP** | SwiGLU MLP compatibility | 376/376, replicates holographic-gate phenom. |
| **T4-MLP-CROSS** | Attention/MLP boundary | 0/32 → 32/32 with 6-channel MLP |
| **T4-SVA** | Subject-verb agreement generation | 24/24, hand-placed weights, no training |
| **T5A** | Multi-clause agreement | 1,152/1,152 with CLAUSE_ID axis |
| **T5B** | Multi-lexeme conjugation | Content-based attention heads |
| **T5C** | LLM-powered labeling | 56-word vocabulary, 18/18 analogy battery |
| **T5D** | Self-improving auto-loop | 56→1,438 words, 6→13 axes, 84% self-decode |
| **T5e–T5i** | METHODOLOGY — failures & corrections | Chain of custody audit |
| **T5j** | Eigenmode spectroscopy | 9 linguistic dimensions from 39×39 bigram op. |
| **T5k** | PMI weight-vector lemma clustering | 23/24 LOO (+3 over baseline) |
| **T6 (E1)** | RECT-pair discovery | 100% train, 100% validation |
| **T6 (E2)** | Channel-meaning probe | 96/96 on unconfounded φ-quantised corpus |
| **T6 (E3)** | Causal perturbation | 100% readout specificity (C1) |
| **T6 (E4)** | φ-Quantisation | Mean error 0.015 from arccos(1/φⁿ) |

## Repository Structure

```
output/
├── paper.md              # Master document (frontmatter + all chapters)
├── paper.pdf             # Compiled PDF
├── paper.tex             # LaTeX output from pandoc
├── chapters/             # Individual chapter markdown files
├── code/                 # Runnable experimental code
│   ├── 01_4state_alphabet.py
│   ├── 02_constructive_sva.py
│   ├── 03_view_pattern.py
│   ├── 04_asm_forward.py
│   ├── 05_rect_pair_discovery.py
│   └── 06_causal_perturbation.py
└── figures/              # Generated figures (5 PNGs)
scripts/
├── build_paper.sh        # Build PDF/HTML/DOCX from markdown
└── preamble.tex          # Pandoc LaTeX preamble
LICENSE                   # GPLv3
README.md                 # This file
```

## Building

Requires pandoc with xelatex for PDF output:

```bash
./scripts/build_paper.sh        # PDF
./scripts/build_paper.sh html   # HTML
./scripts/build_paper.sh docx   # Word
```

### Figure Regeneration

Figures are generated by `output/figures/generate_figures.py`. The build script regenerates them automatically if matplotlib + numpy are available. Pass `--skip-figures` to skip:

```bash
./scripts/build_paper.sh --skip-figures
```

## Methodological Contribution: Chain of Custody

A core deliverable of this volume is the **chain of custody** discipline — a runtime-enforced framework for keeping research artifacts honest about what they have forgotten.

Every projection is encoded as a `View` that declares its assumed symmetries:

```python
@dataclass
class View:
    name: str
    tensor: Tensor
    symmetries_assumed: tuple[str, ...]
    derivation: str
    parent: View

def require_sensitivity_to(view: View, *required: str):
    """Raises SymmetryViolation if any required symmetry is assumed."""
```

When a consumer calls `require_sensitivity_to(view, Sym(POSITIONS), Sym(DIRECTION))`, it is **refused at runtime** if the view has marginalized either dimension away. This is the key finding of Chapter 4: every projection is a contract between an artifact, a data regime, and a question. When any of the three changes, the contract is void until re-checked.

## Dependencies

- **Build**: pandoc, xelatex (TeX Live), Python 3
- **Figures**: Python 3 with numpy, matplotlib
- **Experimental code**: Python 3 with numpy (standalone, no deep learning framework required)

## License

Copyright (C) 2026 TruthSpace Geometric LCM Project

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

## Citation

```bibtex
@techreport{truthspace2026geometric,
  title     = {TruthSpace Volume 2: The Geometric Interpretation of OLMo2},
  subtitle  = {From Architectural Inductive Bias to φ-Computer Equivalence},
  author    = {{TruthSpace Geometric LCM Project}},
  year      = {2026},
  month     = may,
  note      = {Constructive experiments T1–T6 demonstrating φ-computer equivalence
               of transformer architectures}
}
```
