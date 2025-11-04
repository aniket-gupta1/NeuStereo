Video dataloader notes
======================

This folder contains utilities to provide a unified "video" dataset API for
both native video datasets (e.g., Spring) and traditional single-frame
stereo-image datasets (e.g., KITTI, FoundationStereo).

Key components
- `video_datasets.py`: VideoStereoDataset base class and `SpringDataset` plus
  the `build_dataset(cfg)` factory which constructs video-compatible datasets
  from the `cfg.stage` config entries.
- `video_wrappers.py`: `VideoFromStereo` — wraps an existing single-frame
  dataset instance (which exposes `samples`) into a video-style dataset.
- `video_samplers.py`: `LengthAugmentedDataset` and
  `LengthGroupedBatchSampler` — let the training loop request batches where
  all samples share the same sequence length while allowing that length to
  vary across batches.
- `video_collate.py`: `video_collate_fn` — collates a batch of equal-length
  clips into tensors of shape `[B, T, ...]`.

Config options
- `general.allowed_sequence_lengths` (list of ints)
  - Which sequence lengths the length-grouped sampler may produce (e.g. [1,3,5]).
  - If omitted, sensible defaults [1,3,5] are used and clipped by dataset max.

- `general.length_probs` (mapping length -> weight)
  - Optional probabilities for selecting a sequence length per-batch. Keys must
    match entries in `allowed_sequence_lengths`. Values do not need to sum to
    1; they'll be normalized internally. If omitted, lengths are chosen
    uniformly.

- `general.sequence_length` (int, optional)
  - A global fixed sequence length that the builder may use when appropriate.
  - For Spring the builder creates random-length clips by default. To force a
    fixed length set this value and Spring will be constructed with
    `random_sequence=False`.

Notes and tips
- Mixing datasets: The factory wraps single-frame datasets as `sequence_length=1`.
  Spring is treated as a variable-length video dataset. The length-grouped
  sampler allows each batch to have consistent T while letting T vary
  between batches.

- Batching: If you prefer a simpler setup, create separate dataloaders for
  each fixed T you care about (e.g., spring T=3, kitti T=1) and alternate
  between them in the training loop. The current implementation provides a
  single DataLoader with a BatchSampler that groups by T.

- Distributed training: The current `LengthGroupedBatchSampler` is not
  distributed-aware. If you plan to run multi-GPU training, we'll need to
  implement per-process partitioning of the indices. Ask me to add that if
  needed.

Examples
- Train on Spring with variable lengths (1,3,5) and KITTI single-frame:
  - Set `general.stage: ['spring', 'kitti15']` and
    `general.allowed_sequence_lengths: [1,3,5]`.
  - The sampler will pick lengths per-batch (with probabilities set by
    `general.length_probs`).

- Force Spring to produce fixed 5-frame clips and keep KITTI at 1:
  - Set `general.stage: ['spring', 'kitti15']` and `general.sequence_length: 5`.
  - Alternatively, create two separate dataloaders: one for Spring(T=5) and
    one for KITTI(T=1) and alternate between them.
