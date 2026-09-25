"""Reject exception-saved partial attempts as resume boundaries.
The production integration must keep the last successful save unchanged on a
failed forward/backward, and write failure metadata separately. Never rewind
only the sampler cursor: the failed forward can already have mutated BN.
"""
def complete_boundary(payload):
    c=payload['counters'];cfg=payload['config']
    return (c['microbatches']==c['attempts']*cfg['accumulation']
            and c['attempts']==c['updates']+c['skips']
            and c['exposures']==c['microbatches']*cfg['physical_batch']
            and payload['sampler']['next_microbatch']==c['microbatches']
            and payload['scheduler']['completed']==payload['ema_updates']==c['updates'])

def require_complete_boundary(payload):
    if not complete_boundary(payload):
        raise RuntimeError('Checkpoint contains an incomplete optimizer attempt; use the previous complete boundary. Do not skip the failed batch.')

def save_complete_checkpoint(path, payload, atomic_save):
    require_complete_boundary(payload)
    atomic_save(path, payload)
