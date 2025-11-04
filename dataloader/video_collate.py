import torch
import numpy as np


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        # assume HWC -> CHW
        if t.ndim == 3:
            t = t.permute(2, 0, 1)
        return t
    # fallback
    return torch.tensor(x)


def video_collate_fn(batch):
    """Collate function for video samples where each sample is a dict with
    keys: 'left', 'right', 'disp', 'pose', 'intrinsics'. Each of left/right/disp
    are lists of length T (per-sample) of tensors or arrays. This collate
    assumes the sampler ensures all samples in this batch have the same T.
    Returns a dict with batched tensors: left [B, T, C, H, W], right [B, T, C, H, W],
    disp [B, T, H, W] (or None), pose [B, T, 4,4], intrinsics [B, T, ...], and
    frame_mask [B, T] (all ones here because all sequences share same length).
    """

    assert isinstance(batch, list)
    B = len(batch)
    if B == 0:
        return {}

    # Determine T from first sample
    first = batch[0]
    T = len(first['left'])

    # allocate lists
    left_list = []
    right_list = []
    disp_list = []
    pose_list = []
    intrinsics_list = []

    for sample in batch:
        assert len(sample['left']) == T, "All samples in batch must have same T"

        # convert each frame and stack per sample
        left_frames = [_to_tensor(f) for f in sample['left']]
        # ensure shape [T, C, H, W]
        left = torch.stack(left_frames, dim=0)
        left_list.append(left)

        right_frames = [_to_tensor(f) for f in sample['right']]
        right = torch.stack(right_frames, dim=0)
        right_list.append(right)

        # --- DISP (may contain None entries per-frame) ---
        if 'disp' in sample and sample['disp'] is not None and len(sample['disp']) > 0:
            # find a reference frame to infer shape/dtype
            ref_disp = next((d for d in sample['disp'] if d is not None), None)
            ref_t = _to_tensor(ref_disp) if ref_disp is not None else None
            disp_frames = []
            for d in sample['disp']:
                if d is None:
                    if ref_t is not None:
                        disp_frames.append(torch.zeros_like(ref_t))
                    else:
                        # fallback to a zero scalar
                        disp_frames.append(torch.tensor(0.))
                else:
                    disp_frames.append(_to_tensor(d))
            # disp might be [H,W] -> make [T,H,W]
            try:
                disp = torch.stack(disp_frames, dim=0)
            except Exception:
                # if shapes mismatch, coerce with zeros_like using ref_t
                disp = torch.stack([df if isinstance(df, torch.Tensor) else torch.zeros_like(ref_t) for df in disp_frames], dim=0)
            disp_list.append(disp)
        else:
            disp_list.append(None)

        # --- POSE (may contain None entries per-frame) ---
        if 'pose' in sample and sample['pose'] is not None and len(sample['pose']) > 0:
            ref_pose = next((p for p in sample['pose'] if p is not None), None)
            if ref_pose is None:
                pose_list.append(None)
            else:
                ref_t = torch.tensor(ref_pose) if not isinstance(ref_pose, torch.Tensor) else ref_pose
                pose_frames = []
                for p in sample['pose']:
                    if p is None:
                        pose_frames.append(torch.zeros_like(ref_t))
                    else:
                        pose_frames.append(torch.tensor(p) if not isinstance(p, torch.Tensor) else p)
                pose = torch.stack(pose_frames, dim=0)
                pose_list.append(pose)
        else:
            pose_list.append(None)

        # --- INTRINSICS (may contain None entries per-frame) ---
        if 'intrinsics' in sample and sample['intrinsics'] is not None and len(sample['intrinsics']) > 0:
            intr_frames_raw = [i for i in sample['intrinsics']]
            ref_intr = next((i for i in intr_frames_raw if i is not None), None)
            if ref_intr is None:
                intrinsics_list.append(None)
            else:
                ref_t = torch.tensor(ref_intr) if not isinstance(ref_intr, torch.Tensor) else ref_intr
                intr_frames = []
                for i in intr_frames_raw:
                    if i is None:
                        intr_frames.append(torch.zeros_like(ref_t))
                    else:
                        intr_frames.append(torch.tensor(i) if not isinstance(i, torch.Tensor) else i)
                intr = torch.stack(intr_frames, dim=0)
                intrinsics_list.append(intr)
        else:
            intrinsics_list.append(None)

    # Stack across batch -> [B, T, ...]
    left_b = torch.stack(left_list, dim=0)
    right_b = torch.stack(right_list, dim=0)

    if any(d is not None for d in disp_list):
        # Normalize all disp tensors to shape [T, H, W] matching image spatial size.
        # Use left images as reference spatial dims.
        _, _, _, H_ref, W_ref = left_b.shape

        def _normalize_disp(d):
            # d may be a tensor of various shapes or a numpy array
            if not isinstance(d, torch.Tensor):
                d = _to_tensor(d)

            # If channel-first image-like (C,H,W) for a single frame -> convert to (1,H,W)
            if d.ndim == 3 and d.shape[0] == 3 and d.shape[1] != T:
                # treat as single-frame RGB disparity-like -> reduce channels
                d = d.mean(dim=0, keepdim=False).unsqueeze(0)

            # If we have (T, C, H, W) -> reduce channel
            if d.ndim == 4:
                # possible layouts: (T,C,H,W) or (T,H,W,C)
                if d.shape[1] == 3:
                    # (T,C,H,W)
                    d = d.mean(dim=1)
                elif d.shape[-1] == 3:
                    # (T,H,W,C) -> permute to (T,C,H,W) then reduce
                    d = d.permute(0, 3, 1, 2).mean(dim=1)

            # If we have (C,H,W) where C==3 -> single frame
            if d.ndim == 3 and d.shape[0] == 3 and d.shape[1] == H_ref:
                d = d.mean(dim=0, keepdim=False).unsqueeze(0)

            # At this point prefer d to be 3D: (T, H, W) or 2D (H,W)
            if d.ndim == 2:
                d = d.unsqueeze(0)

            # If time dim doesn't match, try to broadcast/repeat single-frame
            if d.shape[0] == 1 and T > 1:
                d = d.repeat(T, 1, 1)

            # Resize spatially if needed using bilinear interpolation
            if d.shape[1] != H_ref or d.shape[2] != W_ref:
                # interpolate expects N,C,H,W
                d4 = d.unsqueeze(1).float()
                d4 = torch.nn.functional.interpolate(d4, size=(H_ref, W_ref), mode='bilinear', align_corners=False)
                d = d4.squeeze(1)

            return d

        disp_list2 = []
        for d in disp_list:
            if d is None:
                # create zeros [T,H_ref,W_ref]
                disp_list2.append(torch.zeros((T, H_ref, W_ref), dtype=left_b.dtype))
            else:
                try:
                    dn = _normalize_disp(d)
                except Exception:
                    # fallback to zeros if normalization fails
                    dn = torch.zeros((T, H_ref, W_ref), dtype=left_b.dtype)
                disp_list2.append(dn)

        disp_b = torch.stack(disp_list2, dim=0)
    else:
        disp_b = None

    if any(p is not None for p in pose_list):
        pose_list2 = [p if p is not None else torch.zeros((T, 4, 4)) for p in pose_list]
        pose_b = torch.stack(pose_list2, dim=0)
    else:
        pose_b = None

    if any(i is not None for i in intrinsics_list):
        intr_list2 = [i if i is not None else torch.zeros((T, *batch[0]['intrinsics'][0].shape)) for i in intrinsics_list]
        intr_b = torch.stack(intr_list2, dim=0)
    else:
        intr_b = None

    frame_mask = torch.ones((B, T), dtype=torch.bool)

    out = {
        'left': left_b,
        'right': right_b,
        'disp': disp_b,
        'pose': pose_b,
        'intrinsics': intr_b,
        'frame_mask': frame_mask
    }

    return out
