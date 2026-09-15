# ICLR 2027 writing blueprint

## Central question

**When can internal-state adaptation replace weight adaptation?**

Answer: when the target correction is both directionally transferable from a
small support set and locally expressible at the chosen activation site.

## Claim hierarchy

1. **Optimization geometry.** The top eigenspace of the support correctness-
   gradient second moment maximizes retained support gradient energy among all
   rank-constrained subspaces (Ky Fan).
2. **Energy is not enough.** Query energy overlap `rho` does not by itself rank
   adaptation success.
3. **Signed transfer matters.** `kappa` distinguishes shared energy from an
   aligned correction, but is still not sufficient.
4. **The actuator must be sufficient.** Query-basis and site oracles separate
   subspace-estimation error from functional-authority limits.
5. **Empirical regimes.** BRIGHT is a positive fixed-semantics evidence shift;
   RoboFail is a class-mixture/site mismatch; ManipBench is a task-remapping
   boundary; Camelyon separates available actuator capacity from finite-
   support realization.

## Main-paper evidence

- Table 1: locked BRIGHT Qwen2.5 fixed-configuration comparison.
- Table 2: Qwen3-VL and InternVL BRIGHT architecture transfer.
- Table 3: corrected InternVL matched three-seed Camelyon/Guardian/ManipBench.
- Table 4: existing query-oracle alpha capacity diagnostic (clearly marked).
- Table 5: existing paired significance and mechanism controls.
- Table 6: Geometry--Actuator regime synthesis.
- Efficiency: unchanged query latency; 100/20 compact controller scalars or
  1,024/64 rank-16 scalars versus 10.1M LoRA parameters.

## Figures to produce after content freeze

1. **Method schematic:** support gradients -> second moment -> top eigenspace
   -> bounded visual KV residual.  Visually separate frozen weights from the
   learned event state.
2. **Geometry x Actuator map:** x-axis directional transfer, y-axis oracle
   actuator sufficiency; place BRIGHT, Camelyon, RoboFail, and ManipBench with
   protocol/oracle caveats.
3. **RoboFail mixture diagnostic:** original versus query-prior-reweighted
   kappa for three seeds, with the oracle label prominent.

## Reviewer-facing caveats

- Do not call the method unsupervised TTA; it uses a small labeled support.
- Do not pool BRIGHT fixed-support and three-seed estimands.
- Do not treat archived support12 BRIGHT geometry as a matched correlate of
  support24 performance.
- Do not claim kappa is sufficient or a leaderboard predictor.
- Do not promote Q-only, QKVO, query-basis, query-prior, or class-conditional
  variants to the method.
- Do not use the superseded InternVL ManipBench raw-label results.
- State that Random-KV is missing from the Guardian suite.

## Stop rule

No new GPU experiment is required unless a numerical source-of-truth audit
reveals a missing artifact for a claim already present in the draft.  New
benchmarks, actuator sweeps, and fully matched BRIGHT LoRA support-seed runs are
out of scope for this submission draft.
