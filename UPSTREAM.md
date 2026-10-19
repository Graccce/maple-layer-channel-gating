# Upstream provenance

- Repository: https://github.com/muzairkhattak/multimodal-prompt-learning
- Snapshot: `69bce21ae8eda80ad6187534b2dce09cf6c59e17`
- Paper: MaPLe: Multi-modal Prompt Learning, CVPR 2023, https://arxiv.org/abs/2210.03117
- License: upstream MIT, retained verbatim in LICENSE.
- This repository imports a source snapshot; its initial Git history is not the upstream commit history.

`tests/fixtures/maple_upstream.py` contains the unmodified upstream trainer at this snapshot. SHA256:

`e067414c25e87a7ea40b9ec399fd13f655fe38a5f205e3bc5a5ce74a78a761b0`

The baseline-forward tests parse this fixture and verify its hash. They do not use the moving repository HEAD. The fixture is reference material and is not imported as a second trainer.

Local additions include the gate/checkpoint implementation, gate configs, tests, experiment/profile scripts, and project documentation/CI. Original MaPLe paper metrics reproduced in UPSTREAM_README.md belong to the authors and are not results of the new gates.
