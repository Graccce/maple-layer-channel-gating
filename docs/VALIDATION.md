# Validation boundary

On 2026-10-19, the pre-publication implementation passed 12 CPU unit tests in the existing AutoDL environment (torch 2.1.2+cu118). This used real prompt learner/Dassl optimizer code with a tiny CLIP fixture. Syntax, shell parsing, diff whitespace and no-GPU refusal checks also passed.

The suite covers modes, depth 1/2/9, shapes, finite values, identity initialization, gradients, optimizer updates, CPU bfloat16, config/CLI wiring, encoder freezing, checkpoint roundtrips and mismatches, real Dassl save/resume plus MaPLe evaluation loading, experiment argument/overwrite guards, and synthetic HM calculation. Script-command simulations and synthetic logs are not real training results.

Actual historical baseline checkpoint prompt parameters were loaded and compared exactly; a legacy gate family was explicitly rejected. Source/output hashes were checked for preservation. Historical checkpoint, source and log provenance were not complete enough to claim a new-method gain, so no historical accuracy is promoted to a project result.

The publication changes fix the baseline test's reference: a checksummed upstream fixture replaces the moving Git HEAD. Automatic CI checks syntax, repository invariants and shell parsing. It does not claim GPU or full CLIP validation.

The revised publication test suite was actually rerun in the same AutoDL CPU environment: 12 tests passed in 1.908 seconds. The new dependency-free repository checks and Python syntax compilation also passed locally. Whitespace diagnostics on the initial full source import include existing upstream formatting; they are not runtime failures.

Not yet verified: full CLIP CUDA inference/backward, CUDA fp16/AMP equivalence, actual dataset training/evaluation of the new gates, or full GPU interruption/resume. See Issues and EXPERIMENTS.md.
