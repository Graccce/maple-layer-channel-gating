# Validation boundary

On 2026-10-19, the pre-publication implementation passed 12 CPU unit tests in the existing AutoDL environment (torch 2.1.2+cu118). This used real prompt learner/Dassl optimizer code with a tiny CLIP fixture. Syntax, shell parsing, diff whitespace and no-GPU refusal checks also passed.

The suite covers modes, depth 1/2/9, shapes, finite values, identity initialization, gradients, optimizer updates, CPU bfloat16, config/CLI wiring, encoder freezing, checkpoint roundtrips and mismatches, real Dassl save/resume plus MaPLe evaluation loading, experiment argument/overwrite guards, and synthetic HM calculation. Script-command simulations and synthetic logs are not real training results.

Actual historical baseline checkpoint prompt parameters were loaded and compared exactly; a legacy gate family was explicitly rejected. Source/output hashes were checked for preservation. Historical checkpoint, source and log provenance were not complete enough to claim a new-method gain, so no historical accuracy is promoted to a project result.

The publication changes fix the baseline test's reference: a checksummed upstream fixture replaces the moving Git HEAD. Automatic CI checks syntax, repository invariants and shell parsing. It does not claim GPU or full CLIP validation.

The revised publication test suite was actually rerun in the same AutoDL CPU environment: 12 tests passed in 1.908 seconds. The new dependency-free repository checks and Python syntax compilation also passed locally. Whitespace diagnostics on the initial full source import include existing upstream formatting; they are not runtime failures.

## 2026-10-20 GPU and formal experiments

13 CPU tests passed in 1.708 seconds, including the new exact-count analyzer and scalar/channel projection-folding test. Real CLIP CUDA fp16 smoke passed in all three modes with exact identity-initialization logits, finite backward/updates and frozen encoders. All three OxfordPets one-epoch smoke runs and nine five-epoch paired formal runs completed; independent Base/Novel evaluation loaded epoch 5. Finite checkpoints, identical frozen backbones, source hashes and 15 preserved original files were verified. See [report](RESULTS_2026-10-20.md) and its machine-readable evidence.

Not yet verified: CUDA AMP/full-fp32 equivalence, full GPU interruption/resume, DTD experiments, fixed-one ablation and full-model folded-checkpoint evaluation. The float64 fold check covers prompt projections only. Static CI does not execute research CPU/GPU tests.

## 2026-10-21 follow-up

14 CPU tests passed in 1.423 seconds. Fixed-one scalar/channel seed-1 checkpoints exactly match the none checkpoint on 328 shared state tensors, with equal Base/Novel accuracy. Full-CLIP native-projection/fp16 folding ran on every Base/Novel test example; logits were finite, but scalar Novel and channel Base/Novel each lost one correct prediction. This is not bitwise equivalence or a lossless checkpoint conversion. See [ablation evidence](ABLATIONS_2026-10-21.md). DTD download and fixed split acquisition failed/ran too slowly; no DTD dataset was adopted or trained. AMP/full-fp32 and GPU interruption/resume remain unverified.
