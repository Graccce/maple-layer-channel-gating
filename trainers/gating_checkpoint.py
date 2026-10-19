"""Strict gate-family checks shared by evaluation, init weights, and resume."""
import torch


def validate_gate_state(learner, state, prefix='prompt_learner.'):
    legacy = prefix + 'gate_alphas'
    if legacy in state:
        raise RuntimeError('Legacy gate checkpoint detected; recover its matching original source/config. '
                           'Its formula cannot be inferred from the key and it must not be silently converted.')
    schema_key = prefix + '_gate_schema'
    incoming = state.get(schema_key)
    expected = learner.state_dict()
    expected_schema = expected.get('_gate_schema')
    if (incoming is None) != (expected_schema is None):
        raise RuntimeError('Gate checkpoint/config mismatch: original none vs scalar/channel. '
                           'Select the matching GATE_TYPE; baseline-to-gated transfer is not enabled.')
    if incoming is not None:
        if incoming.dtype != torch.int64 or not torch.equal(incoming.cpu(), expected_schema.cpu()):
            raise RuntimeError('Gate schema/version mismatch: select the matching scalar/channel configuration.')
    actual = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix + 'gate_alphas.')}
    wanted = {k: v for k, v in expected.items() if k.startswith('gate_alphas.')}
    if set(actual) != set(wanted) or any(actual[k].shape != wanted[k].shape for k in wanted):
        raise RuntimeError('Gate parameter count/shape mismatch; checkpoint may be incomplete or PROMPT_DEPTH differs.')


def load_prompt_checkpoint(model, state):
    """Only class-dependent fixed tokens are replaced; all learned keys are strict."""
    state = state.copy()
    # Base-to-novel evaluation rebuilds these buffers for the target class names.
    for key, value in model.state_dict().items():
        if key in ('prompt_learner.token_prefix', 'prompt_learner.token_suffix'):
            state[key] = value
    model.load_state_dict(state, strict=True)
