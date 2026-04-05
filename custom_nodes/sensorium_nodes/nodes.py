"""
Sensorium custom nodes for Revealing Rendering.

SensoriumLayerInfo   — inspect an AVM_LAYER_SET: list labels, count, confidence
SensoriumSelectLayer — pick one or more layers by name or index, output MASK
SensoriumMaskViz     — colored mask overlay WITHOUT the legend, with per-object ID selection
"""

import torch
import numpy as np
import json


# ── shared color palette (matches SAM3VideoOutput) ───────────────────────────

COLORS = [
    [0.0, 0.5, 1.0],   # Blue
    [1.0, 0.3, 0.3],   # Red
    [0.3, 1.0, 0.3],   # Green
    [1.0, 1.0, 0.0],   # Yellow
    [1.0, 0.0, 1.0],   # Magenta
    [0.0, 1.0, 1.0],   # Cyan
    [1.0, 0.5, 0.0],   # Orange
    [0.5, 0.0, 1.0],   # Purple
    [0.0, 0.8, 0.4],   # Teal
    [0.9, 0.6, 0.1],   # Gold
    [0.6, 0.2, 0.8],   # Violet
    [0.2, 0.8, 0.9],   # Sky
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _layer_keys(layer_set: dict) -> list:
    return list(layer_set.keys())


def _mask_from_layer(layer_value):
    if not isinstance(layer_value, dict):
        return None
    if all(isinstance(k, int) for k in layer_value.keys()):
        frames = []
        for frame_idx in sorted(layer_value.keys()):
            m = layer_value[frame_idx].get("mask")
            if m is not None:
                if m.dim() == 4:
                    m = m[0]
                if m.dim() == 3:
                    m = m[0]
                frames.append(m.float())
        if frames:
            return torch.stack(frames, dim=0)
    return None


def _parse_confidence_from_string(raw_response: str) -> dict:
    import re
    scores = {}
    try:
        for block in re.finditer(r'\{.*?\}', raw_response, re.DOTALL):
            try:
                obj = json.loads(block.group())
                if "layers" in obj and isinstance(obj["layers"], list):
                    for entry in obj["layers"]:
                        label = entry.get("label", "")
                        conf = entry.get("confidence")
                        if label and conf is not None:
                            scores[label] = float(conf)
            except Exception:
                continue
    except Exception:
        pass
    return scores


# ── Node 1: SensoriumLayerInfo ────────────────────────────────────────────────

class SensoriumLayerInfo:
    """
    Inspect an AVM_LAYER_SET.
    Outputs labels, count, index map, and whether propagation has run.
    """

    CATEGORY = "Sensorium"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_set": ("AVM_LAYER_SET",),
            },
            "optional": {
                "raw_response": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("labels", "count", "index_map", "has_masks")
    FUNCTION = "run"

    def run(self, layer_set: dict, raw_response: str = ""):
        keys = _layer_keys(layer_set)
        labels_str = ", ".join(keys)
        count = len(keys)
        index_map = json.dumps({str(i): k for i, k in enumerate(keys)}, indent=2)

        has_masks = "no"
        for v in layer_set.values():
            if isinstance(v, dict) and all(isinstance(k, int) for k in v.keys()):
                has_masks = "yes"
                break

        if raw_response:
            scores = _parse_confidence_from_string(raw_response)
            if scores:
                annotated = []
                for k in keys:
                    s = scores.get(k)
                    annotated.append(f"{k}:{s:.2f}" if s is not None else k)
                labels_str = ", ".join(annotated)

        return (labels_str, count, index_map, has_masks)


# ── Node 2: SensoriumSelectLayer ─────────────────────────────────────────────

class SensoriumSelectLayer:
    """
    Select one or more layers from an AVM_LAYER_SET by name or index.
    """

    CATEGORY = "Sensorium"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_set": ("AVM_LAYER_SET",),
                "selection": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "tooltip": "Layer name(s) or 0-based index(es), comma-separated. e.g. 'person' or '0,2'"
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("mask", "selected_labels", "count")
    FUNCTION = "run"

    def run(self, layer_set: dict, selection: str):
        keys = _layer_keys(layer_set)
        tokens = [t.strip() for t in selection.split(",") if t.strip()]

        selected_keys = []
        for token in tokens:
            try:
                idx = int(token)
                if 0 <= idx < len(keys):
                    k = keys[idx]
                    if k not in selected_keys:
                        selected_keys.append(k)
                continue
            except ValueError:
                pass
            token_lower = token.lower()
            for k in keys:
                if token_lower == k.lower() or token_lower in k.lower():
                    if k not in selected_keys:
                        selected_keys.append(k)

        masks = []
        for k in selected_keys:
            m = _mask_from_layer(layer_set[k])
            if m is not None:
                masks.append(m)

        if masks:
            combined = torch.stack(masks, dim=0).max(dim=0).values
        else:
            combined = torch.zeros(1, 64, 64)

        labels_out = ", ".join(selected_keys) if selected_keys else "(none matched)"
        return (combined, labels_out, len(selected_keys))


# ── Node 3: SensoriumMaskViz ──────────────────────────────────────────────────

class SensoriumMaskViz:
    """
    Colored mask overlay on video frames — NO legend, NO UI clutter.

    obj_id   -1  → show all objects with different colors (no legend drawn)
    obj_id   N   → show only object N with its color
    alpha        → overlay transparency (0=invisible, 1=opaque mask)

    Outputs:
      visualization  IMAGE  — colored overlay, clean (no legend)
      mask           MASK   — selected object mask [F, H, W]
      obj_ids        STRING — "0, 1, 2, ..." available IDs with scores
    """

    CATEGORY = "Sensorium"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks":       ("SAM3_VIDEO_MASKS",),
                "video_state": ("SAM3_VIDEO_STATE",),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES",),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "-1 = all objects, 0..N = specific object"
                }),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask overlay opacity"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("visualization", "mask", "obj_ids")
    FUNCTION = "run"

    def run(self, masks, video_state, scores=None, obj_id=-1, alpha=0.5):
        import os
        from PIL import Image as PILImage

        # video_state may be a dataclass or a plain dict depending on which node produced it
        if isinstance(video_state, dict):
            h          = video_state["height"]
            w          = video_state["width"]
            num_frames = video_state["num_frames"]
            temp_dir   = video_state["temp_dir"]
        else:
            h          = video_state.height
            w          = video_state.width
            num_frames = video_state.num_frames
            temp_dir   = video_state.temp_dir

        if not masks:
            empty = torch.zeros(num_frames, h, w, 3)
            empty_mask = torch.zeros(num_frames, h, w)
            return (empty, empty_mask, "no masks")

        vis_frames = []
        out_masks = []
        num_objects = 0

        for frame_idx in range(num_frames):
            # Load original frame
            frame_path_jpg = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
            if os.path.exists(frame_path_jpg):
                img = PILImage.open(frame_path_jpg).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
            else:
                img_np = np.zeros((h, w, 3), dtype=np.float32)

            vis = torch.from_numpy(img_np.copy())
            out_mask = torch.zeros(h, w)

            if frame_idx in masks:
                frame_mask = masks[frame_idx]
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)

                if frame_mask.dim() == 3 and frame_mask.shape[0] >= 1:
                    num_objects = max(num_objects, frame_mask.shape[0])

                    if obj_id == -1:
                        # All objects, each with its own color — NO legend
                        for oid in range(frame_mask.shape[0]):
                            om = frame_mask[oid].float()
                            if om.max() > 1.0:
                                om = om / 255.0
                            color = torch.tensor(COLORS[oid % len(COLORS)])
                            mask_rgb = om.unsqueeze(-1) * color.view(1, 1, 3)
                            vis = vis * (1 - alpha * om.unsqueeze(-1)) + alpha * mask_rgb
                            out_mask = torch.max(out_mask, om)
                    else:
                        # Single selected object
                        oid = min(obj_id, frame_mask.shape[0] - 1)
                        om = frame_mask[oid].float()
                        if om.max() > 1.0:
                            om = om / 255.0
                        color = torch.tensor(COLORS[oid % len(COLORS)])
                        mask_rgb = om.unsqueeze(-1) * color.view(1, 1, 3)
                        vis = vis * (1 - alpha * om.unsqueeze(-1)) + alpha * mask_rgb
                        out_mask = om

                        # Extract single obj mask for output
                        if obj_id < frame_mask.shape[0]:
                            out_mask = frame_mask[obj_id].float()
                            if out_mask.max() > 1.0:
                                out_mask = out_mask / 255.0

            vis_frames.append(vis.clamp(0, 1))
            out_masks.append(out_mask)

        # Build obj_ids string from scores
        if scores is not None and num_objects > 0:
            # Grab scores from first frame that has them
            first_frame = next(iter(scores.keys()), None)
            if first_frame is not None:
                fs = scores[first_frame]
                if hasattr(fs, 'tolist'):
                    fs = fs.tolist()
                    if fs and isinstance(fs[0], list):
                        fs = fs[0]
                ids_str = ", ".join(
                    f"{oid}:{fs[oid]:.2f}" if oid < len(fs) else str(oid)
                    for oid in range(num_objects)
                )
            else:
                ids_str = ", ".join(str(i) for i in range(num_objects))
        else:
            ids_str = ", ".join(str(i) for i in range(num_objects))

        visualization = torch.stack(vis_frames, dim=0)   # [F, H, W, 3]
        out_mask_tensor = torch.stack(out_masks, dim=0)  # [F, H, W]

        return (visualization, out_mask_tensor, ids_str)


# ── Registry ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SensoriumLayerInfo":   SensoriumLayerInfo,
    "SensoriumSelectLayer": SensoriumSelectLayer,
    "SensoriumMaskViz":     SensoriumMaskViz,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SensoriumLayerInfo":   "Sensorium — Layer Info",
    "SensoriumSelectLayer": "Sensorium — Select Layer",
    "SensoriumMaskViz":     "Sensorium — Mask Viz (no legend)",
}
