# Project rules

- Keep `TRAINER.MAPLE.GATE_TYPE` defaulting to `none`.
- Preserve upstream MaPLe trainable parameters and CLIP freezing.
- Gates are static learned parameters shared over samples, not input-conditioned gates.
- Never silently cross-load none/scalar/channel or legacy gate checkpoints.
- Keep the upstream fixture pinned; never replace baseline references with current HEAD.
- Run `python scripts/check_repository.py` and `python -m unittest discover -s tests -v` after relevant changes.
- Do not launch GPU training, rent resources, or download datasets without authorization.
- Do not commit datasets, logs, checkpoints, account details, or access tokens.
- Keep each experiment in a new output directory. Report failed and unrun checks honestly.
- Never describe this affine-redundant gate as an increase in function expressiveness or claim novelty/accuracy gains without evidence.
- Use focused feature branches and review changes before merging. Keep experiment results separate from implementation status.
