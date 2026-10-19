# Experiments

1. Run syntax/static checks and the CPU unit suite.
2. With an available GPU, run `gpu_smoke.py` for none/scalar/channel: real CLIP logits, pinned-baseline comparison, backward and optimizer step.
3. OxfordPets, seed 1, one epoch (`smoke`): verify data paths, losses, checkpoint reload, base/new results, timing and memory fields.
4. OxfordPets, five epochs (`formal`), paired seeds 1/2/3, all three modes.
5. Prepare DTD and repeat without changing the main comparison's optimizer, shots or epoch settings.

Use a new RUN_ID for each rerun. The runner refuses existing output directories and never automatically resumes. Keep dataset splits and few-shot caches fixed. Do not mix smoke with formal results.

Report Base Accuracy, Novel Accuracy, per-seed HM, trainable parameters, time, and allocated/reserved GPU peak memory. Compute HM per seed, then report its mean and sample standard deviation. Timing currently includes the trainer's final evaluation and is named accordingly; distinguish it from pure training-kernel time.

```bash
python scripts/maple/summarize_gating.py \
  output/gating_v1/formal/pets_formal01/oxford_pets/channel/seed{1,2,3}
```

The same-epoch checkpoint is evaluated on both base and novel test subsets. Validation accuracy must not be substituted for base test accuracy.

Hypotheses may fail. A useful next ablation is gate fixed at 1; frozen projection, alternative gate LR, and gate regularization are separate diagnostic experiments and must not be silently mixed into the primary comparison. Fold a trained gate into W,b to test the affine equivalence. Record layer-wise gate distributions and saturation, without treating additional parameters as additional expressiveness.

No formal results are available in this repository yet.
