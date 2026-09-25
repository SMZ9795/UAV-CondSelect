"""Apply Bernoulli thinning to a shuffled repeating image stream.
Random draws are reproducible functions of the seed and epoch.
At the dataset's natural generated fraction, every entry is accepted.
"""
from fractions import Fraction
import hashlib
import random
from bisect import bisect_right

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('\0'.join(map(str, parts)).encode()).digest()[:8], 'big')

class RatioBatchSampler:
    def __init__(self, entries, batch_size, seed, p_gen, start_batch=0):
        self.entries = entries
        self.batch_size, self.seed, self.start_batch = batch_size, seed, start_batch
        self.p_gen = p_gen
        counts = {}
        for m in entries:
            k = m['task'] if m['domain'] == 'generated' else 'clean'
            counts[k] = counts.get(k, 0) + 1
        conditions = sorted(set(counts) - {'clean'})
        assert 0 < p_gen < 1 and conditions
        # Represent the matched sampling fractions exactly.
        if p_gen in (4/9, 2/7):
            p_gen = Fraction(4,9) if p_gen == 4/9 else Fraction(2,7)
        weights = {k: ((1-p_gen)/n if k == 'clean' else p_gen/(len(conditions)*n)) for k,n in counts.items()}
        maximum = max(weights.values())
        self.accept = {k: float(w/maximum) for k,w in weights.items()}
        self.epochs, self.ends = [], [0]

    def extend(self):
        epoch = len(self.epochs)
        order = list(range(len(self.entries)))
        random.Random(stable_seed('calibration-uniform-order', self.seed, epoch)).shuffle(order)
        rng = random.Random(stable_seed('ratio-thinning-v1', self.seed, epoch))
        accepted = []
        for i in order:
            m = self.entries[i]
            key = m['task'] if m['domain'] == 'generated' else 'clean'
            if rng.random() < self.accept[key]:
                accepted.append(i)
        self.epochs.append(accepted)
        self.ends.append(self.ends[-1] + len(accepted))

    def batch(self, number):
        pos = number * self.batch_size
        while self.ends[-1] < pos + self.batch_size:
            self.extend()
        result = []
        while len(result) < self.batch_size:
            epoch = bisect_right(self.ends, pos) - 1
            part = self.epochs[epoch][pos-self.ends[epoch]:pos-self.ends[epoch]+self.batch_size-len(result)]
            result.extend(part)
            pos += len(part)
        return result

    def state_dict(self, consumed_batches):
        return {'algorithm': 'permutation-bernoulli-thinning', 'seed': self.seed,
                'p_gen': self.p_gen, 'consumed_batches': consumed_batches,
                'next_indices': self.batch(consumed_batches),
                'sampling_generator': 'stateless: Python Random SHA256(seed, original epoch); reconstruct from origin; no global RNG consumption'}

    def __iter__(self):
        n = self.start_batch
        while True:
            yield self.batch(n)
            n += 1

    def __len__(self):
        return 2**31-1
