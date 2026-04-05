"""
Sensorium custom nodes for Revealing Rendering.

SensoriumLayerInfo   — inspect an AVM_LAYER_SET: list labels, count, confidence
SensoriumSelectLayer — pick one or more layers by name or index, output MASK
"""

import torch
import json


# ── helpers ──────────────────────────────────────────────────────────────────

def _layer_keys(layer_set: dict) -> list[str]:
    return list(layer_set.keys())


def _mask_from_layer(layer_value) -> torch.Tensor | None:
    """
    An AVM_LAYER_SET value is either:
      - a boxes dict  {"boxes": [...], "labels": [...]}           (pre-propagation)
      - a SAM3_VIDEO_MASKS dict  {frame_idx: {"mask": tensor}}    (post-propagation)

    Returns a MASK tensor [F, H, W] if video masks exist, else None.
    """
    if not isinstance(layer_value, dict):
        return None

    # Post-propagation: keys are frame indices (ints)
    if all(isinstance(k, int) for k in layer_value.keys()):
        frames = []
        for frame_idx in sorted(layer_value.keys()):
            m = layer_value[frame_idx].get("mask")
            if m is not None:
                if m.dim() == 4:
                    m = m[0]       # [1,H,W] → [H,W] ... wait, keep [C,H,W]? take first
                if m.dim() == 3:
                    m = m[0]       # [C,H,W] → [H,W]
                frames.append(m.float())
        if frames:
            return torch.stack(frames, dim=0)  # [F, H, W]

    return None


def _confidence_from_raw(layer_set: dict) -> dict[str, float]:
    """
    Confidence is not stored in AVM_LAYER_SET itself.
    Returns empty dict — confidence comes from the raw_response STRING.
    """
    return {}


def _parse_confidence_from_string(raw_response: str) -> dict[str, float]:
    """
    Parse confidence values out of the Gemini raw_response STRING output.
    Expected format somewhere in the string:
      {"layers": [{"label": "person", "confidence": 0.82, ...}, ...]}
    """
    scores = {}
    try:
        # Find JSON blocks in the raw response
        import re
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

    Outputs:
      labels      STRING  — comma-separated layer names  e.g. "person, hat, bag"
      count       INT     — number of layers
      index_map   STRING  — JSON  {"0": "person", "1": "hat", ...}
      has_masks   STRING  — "yes" or "no" (whether propagation has run)
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

        # Check if any layer has propagated masks
        has_masks = "no"
        for v in layer_set.values():
            if isinstance(v, dict) and all(isinstance(k, int) for k in v.keys()):
                has_masks = "yes"
                break

        # Optionally parse confidence from raw_response and append to labels
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

    selection   — comma-separated names or 0-based indices, e.g. "person" or "0,2"
                  mixing is fine: "person, 2"

    Outputs:
      mask        MASK    — [F,H,W] combined mask of all selected layers (post-propagation)
      labels      STRING  — which labels were actually selected
      count       INT     — number of selected layers
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
                    "tooltip": "Layer name(s) or index(es), comma-separated. e.g. 'person' or '0,2'"
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
            # Try as integer index
            try:
                idx = int(token)
                if 0 <= idx < len(keys):
                    k = keys[idx]
                    if k not in selected_keys:
                        selected_keys.append(k)
                continue
            except ValueError:
                pass

            # Try as exact label match (case-insensitive)
            token_lower = token.lower()
            for k in keys:
                if token_lower == k.lower() or token_lower in k.lower():
                    if k not in selected_keys:
                        selected_keys.append(k)

        # Collect masks for selected layers
        masks = []
        for k in selected_keys:
            m = _mask_from_layer(layer_set[k])
            if m is not None:
                masks.append(m)

        if masks:
            # Union all selected masks: max across layers
            combined = torch.stack(masks, dim=0).max(dim=0).values  # [F, H, W]
        else:
            # No propagated masks yet — return empty single-frame mask
            combined = torch.zeros(1, 64, 64)

        labels_out = ", ".join(selected_keys) if selected_keys else "(none matched)"
        return (combined, labels_out, len(selected_keys))


# ── Registry ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SensoriumLayerInfo":   SensoriumLayerInfo,
    "SensoriumSelectLayer": SensoriumSelectLayer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SensoriumLayerInfo":   "Sensorium — Layer Info",
    "SensoriumSelectLayer": "Sensorium — Select Layer",
}
