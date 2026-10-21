# Project tasks

- [x] none/scalar/channel with identity initialization
- [x] Strict checkpoint-family checks
- [x] Lightweight CPU tests and isolated experiment runner
- [x] Pin upstream baseline fixture independently of HEAD
- [x] Real CLIP CUDA smoke test for all three modes
- [x] OxfordPets one-epoch seed-1 validation
- [x] OxfordPets five-epoch paired seeds 1/2/3
- [x] DTD official archive/fixed-split validation, three-mode smoke, and paired seeds 1/2/3
- [x] Trained prompt-projection folding check (CPU float64)
- [x] Fixed-one scalar/channel training control (seed 1)
- [x] Full-CLIP fp16 folding comparison on Base/Novel (seed 1; small prediction changes observed)
- [x] Initial primary-source comparison: MaPLe, CoCoOp, SSF, DCP, PromptSRC
- [ ] Further layer/channel-gating novelty checks and unresolved full-text candidate

Track actual work in repository Issues. Check an item only after preserving its test or experiment evidence. No automatic GPU training is scheduled.

Evidence: [OxfordPets report](RESULTS_2026-10-20.md). AutoDL was verified shut down on 2026-10-21.

Follow-up evidence: [2026-10-21 ablations](ABLATIONS_2026-10-21.md). DTD download/split network failures are documented there.

Latest evidence: [DTD results](RESULTS_DTD_2026-10-22.md), [data preparation](DTD_DATA_PREPARATION.md), [related work](RELATED_WORK_2026-10-22.md).
