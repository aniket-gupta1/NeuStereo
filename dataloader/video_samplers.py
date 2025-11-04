import random
from typing import Dict, List, Tuple
from torch.utils.data import Sampler


class LengthAugmentedDataset:
    """A thin wrapper that exposes (start_index, length) as dataset items.

    It builds an index map of tuples (start_idx, length) where start_idx refers
    to the underlying dataset.valid_starts_all position. This lets samplers
    target specific lengths and the underlying dataset provides the clip
    extraction via get_clip(seq_idx, frame_idx, length).
    """

    def __init__(self, base_dataset, allowed_lengths: List[int]):
        self.base = base_dataset
        if not hasattr(self.base, 'valid_starts_all') or not hasattr(self.base, 'max_len_per_start'):
            raise ValueError("base_dataset must have valid_starts_all and max_len_per_start populated; call build_all_starts() first")

        self.allowed_lengths = sorted(int(x) for x in allowed_lengths)
        self.index_map: List[Tuple[int, int]] = []
        # parallel list that stores the originating dataset/source for each
        # entry in index_map. This lets samplers ensure batches contain
        # samples from a single dataset/source at a time.
        self.index_sources: List[str] = []

        # Build index_map: list of (start_idx, length)
        for start_idx, max_len in enumerate(self.base.max_len_per_start):
            for L in self.allowed_lengths:
                if max_len >= L:
                    self.index_map.append((start_idx, L))
                    # determine source label for this start
                    try:
                        seq_idx, _ = self.base.valid_starts_all[start_idx]
                        src = self.base.seq_source[seq_idx] if hasattr(self.base, 'seq_source') else 'default'
                    except Exception:
                        src = 'default'
                    self.index_sources.append(src)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        start_idx, L = self.index_map[idx]
        seq_idx, frame_idx = self.base.valid_starts_all[start_idx]
        return self.base.get_clip(seq_idx, frame_idx, L)

    def get_indices_by_length(self) -> Dict[int, List[int]]:
        d: Dict[int, List[int]] = {L: [] for L in self.allowed_lengths}
        for idx, (start_idx, L) in enumerate(self.index_map):
            d[L].append(idx)
        return d

    def get_indices_by_length_and_source(self) -> Dict[int, Dict[str, List[int]]]:
        """Return nested mapping length -> source -> list of augmented indices.

        This is used by samplers that want to ensure each batch contains
        samples from a single dataset source as well as a single length.
        """
        d: Dict[int, Dict[str, List[int]]] = {L: {} for L in self.allowed_lengths}
        for idx, (start_idx, L) in enumerate(self.index_map):
            src = self.index_sources[idx] if idx < len(self.index_sources) else 'default'
            if src not in d[L]:
                d[L][src] = []
            d[L][src].append(idx)
        return d


class LengthGroupedBatchSampler(Sampler):
    """BatchSampler that yields batches where all samples have the same sequence length.

    It takes a LengthAugmentedDataset (or anything with get_indices_by_length())
    and yields lists of indices (into that augmented dataset) where each batch
    contains indices corresponding to the same length L. The length for each
    batch is chosen according to `length_probs` (uniform by default).

    NOTE: This sampler yields batches (lists of indices) and is intended to be
    used as DataLoader(batch_sampler=...). It does not implement distributed
    support; you'd need to wrap/partition pools per-process for multi-GPU.
    """

    def __init__(self, length_augmented_dataset, batch_size: int, drop_last: bool = False, shuffle: bool = True, length_probs: Dict[int, float] = None):
        self.dataset = length_augmented_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Prefer using length+source pools if available on the dataset
        if hasattr(self.dataset, 'get_indices_by_length_and_source'):
            self.indices_by_length_and_source = self.dataset.get_indices_by_length_and_source()
            # build a flat list of lengths (keys)
            self.lengths = list(self.indices_by_length_and_source.keys())
        else:
            self.indices_by_length = self.dataset.get_indices_by_length()
            self.lengths = list(self.indices_by_length.keys())

        if length_probs is None:
            # uniform over lengths
            prob = 1.0 / len(self.lengths)
            self.length_probs = {L: prob for L in self.lengths}
        else:
            self.length_probs = dict(length_probs)

        # prepare internal pools structure (will be reshuffled in __iter__)
        if hasattr(self, 'indices_by_length_and_source'):
            # nested mapping L -> src -> list[idx]
            self._pools = {L: {src: list(idxs) for src, idxs in self.indices_by_length_and_source[L].items()} for L in self.lengths}
        else:
            self._pools = {L: list(self.indices_by_length[L]) for L in self.lengths}

    def __iter__(self):
        # prepare a working copy of pools
        # Build working pools and cursors for iteration
        if hasattr(self, 'indices_by_length_and_source'):
            # pools: L -> src -> list
            pools = {L: {src: list(idxs) for src, idxs in self._pools[L].items()} for L in self.lengths}
            if self.shuffle:
                for L in pools:
                    for src in pools[L]:
                        random.shuffle(pools[L][src])

            cursors = {L: {src: 0 for src in pools[L]} for L in pools}

            total_available = sum(len(idxs) for L in pools for idxs in pools[L].values())

            while True:
                if total_available < self.batch_size:
                    break

                # pick a length
                L = random.choices(self.lengths, weights=[self.length_probs[l] for l in self.lengths], k=1)[0]

                # pick a source for this length that still has remaining items
                src_pools = pools.get(L, {})
                available_srcs = [s for s in src_pools.keys() if cursors[L][s] < len(src_pools[s])]
                if not available_srcs:
                    # no source left for this length, check if any pool left globally
                    remaining_global = sum(len(idxs) - cursors[L2].get(s2, 0) for L2 in pools for s2, idxs in pools[L2].items())
                    if remaining_global < self.batch_size:
                        break
                    else:
                        continue

                # choose source weighted by remaining items
                weights = [len(src_pools[s]) - cursors[L][s] for s in available_srcs]
                src = random.choices(available_srcs, weights=weights, k=1)[0]

                pool = src_pools[src]
                cursor = cursors[L][src]

                end = cursor + self.batch_size
                batch_indices = pool[cursor:end]
                cursors[L][src] = end

                if len(batch_indices) < self.batch_size:
                    if self.drop_last:
                        # skip incomplete batch
                        continue
                    else:
                        # yield smaller batch if allowed
                        pass

                total_available -= len(batch_indices)
                yield batch_indices

        else:
            pools = {L: list(v) for L, v in self._pools.items()}
            if self.shuffle:
                for L in pools:
                    random.shuffle(pools[L])

            cursors = {L: 0 for L in self.lengths}

            total_available = sum(len(v) for v in pools.values())

            while True:
                if total_available < self.batch_size:
                    break

                # pick a length
                L = random.choices(self.lengths, weights=[self.length_probs[l] for l in self.lengths], k=1)[0]

                pool = pools[L]
                cursor = cursors[L]

                if cursor >= len(pool):
                    # pool exhausted
                    if all(cursors[l] >= len(pools[l]) for l in self.lengths):
                        break
                    else:
                        continue

                end = cursor + self.batch_size
                batch_indices = pool[cursor:end]
                cursors[L] = end

                if len(batch_indices) < self.batch_size:
                    if self.drop_last:
                        continue
                    else:
                        pass

                total_available -= len(batch_indices)
                yield batch_indices

    def __len__(self):
        # compute total number of augmented indices available
        if hasattr(self, 'indices_by_length_and_source'):
            total = 0
            for L, src_map in self.indices_by_length_and_source.items():
                for src, idxs in src_map.items():
                    total += len(idxs)
        elif hasattr(self, 'indices_by_length'):
            total = sum(len(v) for v in self.indices_by_length.values())
        else:
            # fallback: compute from _pools (whatever structure it has)
            total = 0
            for v in self._pools.values():
                if isinstance(v, dict):
                    for idxs in v.values():
                        total += len(idxs)
                else:
                    total += len(v)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size
