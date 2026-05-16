# Chapter 1: The Geometric OLMo2 Experiment

## 1.1 The OLMo2 φ-Computer Equivalence

The first phase of the research program sought to validate the core thesis of the original TruthSpace paper: that a standard transformer, without modification to its pre-trained weights, is computationally equivalent to a φ-computer. The OLMo2 model was chosen as the testbed for this experiment.

The work, encapsulated in the `phi_olmo2.py` implementation, involved a systematic, layer-by-layer replacement of the standard OLMo2 architecture with components built from φ-geometric primitives. This was not an approximation, but an assertion of algebraic identity. Key components were rewritten as follows:

- **`PhiRMSNorm`**: Standard RMS Normalization was re-interpreted as a projection onto the unit φ-sphere.
- **`PhiOlmo2Attention`**: The attention mechanism, including Q/K/V projections, Rotary Position Embeddings (RoPE), and softmax, was reformulated as a process of spatial navigation on the φ-lattice.
- **`PhiOlmo2MLP`**: The SwiGLU activation in the MLP was expressed as a `phi_swiglu` operation, treating the gate as a φ-level selector.

The resulting `PhiOlmo2ForCausalLM` model was designed to load weights directly from a standard OLMo2 checkpoint. This approach serves as a powerful validation of the geometric theory: if the φ-rewritten model can produce numerically identical outputs to the original, it demonstrates that the underlying computation was already geometric in nature.

The identity verification suite (`verify.py`) confirmed that all φ-identities pass with maximum differences ≤ 5e-16: phi_exp, phi_sigmoid, phi_softmax, phi_silu, phi_rmsnorm, and phi_rope. The φ-geometric reformulation is not an approximation — it is an exact algebraic rewrite.

## 1.2 Architectural Inductive Bias: The Bloch-OLMo Experiment

Parallel to the reverse-engineering effort, a second line of inquiry explored whether the φ-geometric structure could be encouraged to form organically through architectural design. This led to the creation of `BlochOlmoForCausalLM`, a smaller, custom-built model designed to test a specific geometric hypothesis from scratch.

The central innovation in `bloch_olmo.py` is the `BlockRMSNorm` layer. Unlike a standard RMSNorm which normalizes the entire hidden state, this layer independently normalizes discrete 4-dimensional blocks of the activations. The hypothesis was that by enforcing a constant geometric structure — a "Bloch-sphere isotropy" — at each block, the model would be architecturally biased to develop the quaternion-like structures that the TruthSpace theory predicts are fundamental to semantic representation.

The results (Finding F-BL-T1) were a "clean failure with a strong signal." The model trained well and the BlockRMSNorm layer successfully enforced its geometric constraint at the embedding layer, with 68% of tested semantic opposites aligning correctly. However, this geometric structure rapidly dissolved as it propagated through the model's deeper layers. The antipodality signal was significantly weaker in the middle and final layers of the network.

This finding suggests that while architectural inductive bias can encourage the formation of local geometric structures at the input layer, it is not sufficient on its own to ensure that this geometry is preserved or utilized by the deeper computational machinery of the transformer. The geometry must be explicitly woven into the computational fabric — a theme that will recur throughout the following chapters.
