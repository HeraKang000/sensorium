"""
Sensorium custom nodes for Revealing Rendering.

SensoriumLayerInfo   — inspect an AVM_LAYER_SET: list labels, count, confidence
SensoriumSelectLayer — pick one or more layers by name or index, output MASK
SSSam3VideoOutput    — exact clone of SAM3VideoOutput with show_legend toggle
"""

import gc
import os
import json
import torch
import numpy as np


# ── shared color palette (matches SAM3VideoOutput) ───────────────────────────

COLORS = [
    [0.0, 0.5, 1.0],
    [1.0, 0.3, 0.3],
    [0.3, 1.0, 0.3],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 0.0],
    [0.5, 0.0, 1.0],
]


# ── helpers (AVM layer nodes) ─────────────────────────────────────────────────

def _layer_keys(layer_set):
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


def _parse_confidence_from_string(raw_response):
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
    CATEGORY = "Sensorium"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"layer_set": ("AVM_LAYER_SET",)},
            "optional": {"raw_response": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("labels", "count", "index_map", "has_masks")
    FUNCTION = "run"

    def run(self, layer_set, raw_response=""):
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
                annotated = [f"{k}:{scores[k]:.2f}" if k in scores else k for k in keys]
                labels_str = ", ".join(annotated)

        return (labels_str, count, index_map, has_masks)


# ── Node 2: SensoriumSelectLayer ─────────────────────────────────────────────

class SensoriumSelectLayer:
    CATEGORY = "Sensorium"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_set": ("AVM_LAYER_SET",),
                "selection": ("STRING", {"default": "0", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("mask", "selected_labels", "count")
    FUNCTION = "run"

    def run(self, layer_set, selection):
        keys = _layer_keys(layer_set)
        tokens = [t.strip() for t in selection.split(",") if t.strip()]

        selected_keys = []
        for token in tokens:
            try:
                idx = int(token)
                if 0 <= idx < len(keys) and keys[idx] not in selected_keys:
                    selected_keys.append(keys[idx])
                continue
            except ValueError:
                pass
            token_lower = token.lower()
            for k in keys:
                if (token_lower == k.lower() or token_lower in k.lower()) and k not in selected_keys:
                    selected_keys.append(k)

        masks = [m for k in selected_keys for m in [_mask_from_layer(layer_set[k])] if m is not None]
        combined = torch.stack(masks, dim=0).max(dim=0).values if masks else torch.zeros(1, 64, 64)
        labels_out = ", ".join(selected_keys) if selected_keys else "(none matched)"
        return (combined, labels_out, len(selected_keys))


# ── Node 3: SSSam3VideoOutput ─────────────────────────────────────────────────

class SSSam3VideoOutput:
    """
    Drop-in replacement for SAM3VideoOutput.
    Identical behaviour + show_legend toggle to hide the ID/score overlay.

    show_legend = True  → same as original SAM3VideoOutput
    show_legend = False → clean colored masks, no left-side legend
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS", {"tooltip": "Masks from SAM3Propagate"}),
                "video_state": ("SAM3_VIDEO_STATE", {"tooltip": "Video state for dimensions"}),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES", {"tooltip": "Confidence scores from SAM3Propagate"}),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Specific object ID (-1 for all combined)"
                }),
                "plot_all_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show all object masks (True) or only selected obj_id (False)"
                }),
                "show_legend": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show ID/score legend overlay (True) or hide it (False)"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "frames", "visualization")
    FUNCTION = "extract"
    CATEGORY = "Sensorium"

    # ── legend helpers (copied from SAM3VideoOutput) ──────────────────────────

    def _draw_legend(self, vis_frame, num_objects, colors, obj_id=-1, frame_scores=None):
        h, w = vis_frame.shape[:2]
        box_size = max(16, min(32, h // 20))
        padding = max(4, box_size // 4)
        text_width = box_size * 6
        legend_item_height = box_size + padding

        if obj_id >= 0:
            items = [(obj_id, frame_scores[obj_id] if frame_scores and obj_id < len(frame_scores) else None)]
        else:
            items = [(oid, frame_scores[oid] if frame_scores and oid < len(frame_scores) else None)
                     for oid in range(num_objects)]
            items.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0)))

        legend_height = len(items) * legend_item_height + padding
        legend_width = box_size + text_width + padding * 2
        start_x, start_y = padding, padding
        bg_alpha = 0.7

        for y in range(start_y, min(start_y + legend_height, h)):
            for x in range(start_x, min(start_x + legend_width, w)):
                vis_frame[y, x] = vis_frame[y, x] * (1 - bg_alpha) + torch.tensor([0.1, 0.1, 0.1]) * bg_alpha

        for idx, (oid, score) in enumerate(items):
            item_y = start_y + padding + idx * legend_item_height
            color = torch.tensor(colors[oid % len(colors)])
            for y in range(item_y, min(item_y + box_size, h)):
                for x in range(start_x + padding, min(start_x + padding + box_size, w)):
                    vis_frame[y, x] = color
            text_x = start_x + padding + box_size + padding
            score_str = f"{oid}:{score:.2f}" if score is not None else str(oid)
            self._draw_text(vis_frame, score_str, text_x, item_y, box_size)

        return vis_frame

    def _draw_text(self, img, text, x, y, size):
        chars = {
            '0': [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
            '1': [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
            '2': [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],
            '3': [[1,1,1],[0,0,1],[1,1,1],[0,0,1],[1,1,1]],
            '4': [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],
            '5': [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],
            '6': [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],
            '7': [[1,1,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
            '8': [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],
            '9': [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],
            ':': [[0,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,0]],
            '.': [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,1,0]],
        }
        h, w = img.shape[:2]
        scale = max(1, size // 6)
        char_width = 4 * scale
        curr_x = x
        for char in text:
            if char in chars:
                for row_idx, row in enumerate(chars[char]):
                    for col_idx, pixel in enumerate(row):
                        if pixel:
                            for sy in range(scale):
                                for sx in range(scale):
                                    px = curr_x + col_idx * scale + sx
                                    py = y + row_idx * scale + sy
                                    if 0 <= px < w and 0 <= py < h:
                                        img[py, px] = torch.tensor([1.0, 1.0, 1.0])
                curr_x += char_width
            elif char == ' ':
                curr_x += char_width

    # ── main extract (identical to SAM3VideoOutput.extract + show_legend) ─────

    def extract(self, masks, video_state, scores=None, obj_id=-1, plot_all_masks=True, show_legend=False):
        from PIL import Image as PILImage

        # Resolve video_state (dataclass or dict)
        if isinstance(video_state, dict):
            h          = video_state["height"]
            w          = video_state["width"]
            num_frames = video_state["num_frames"]
            temp_dir   = video_state["temp_dir"]
            session_id = video_state.get("session_uuid", "unknown")
        else:
            h          = video_state.height
            w          = video_state.width
            num_frames = video_state.num_frames
            temp_dir   = video_state.temp_dir
            session_id = video_state.session_uuid

        print(f"[SSSam3VideoOutput] masks type={type(masks)}, len={len(masks) if masks else 0}")
        if masks:
            first_key = next(iter(masks))
            first_val = masks[first_key]
            print(f"[SSSam3VideoOutput] first frame_idx={first_key}, mask shape={first_val.shape if hasattr(first_val, 'shape') else type(first_val)}")

        if not masks:
            print("[SSSam3VideoOutput] masks is empty — returning blank frames")
            empty = torch.zeros(num_frames, h, w)
            empty_img = torch.zeros(num_frames, h, w, 3)
            return (empty, empty_img, empty_img)

        mmap_dir  = os.path.join(temp_dir, "ss_mmap_output")
        os.makedirs(mmap_dir, exist_ok=True)

        mask_mmap  = np.memmap(os.path.join(mmap_dir, "masks.mmap"),  dtype='float32', mode='w+', shape=(num_frames, h, w))
        frame_mmap = np.memmap(os.path.join(mmap_dir, "frames.mmap"), dtype='float32', mode='w+', shape=(num_frames, h, w, 3))
        vis_mmap   = np.memmap(os.path.join(mmap_dir, "vis.mmap"),    dtype='float32', mode='w+', shape=(num_frames, h, w, 3))

        num_objects = 0

        for frame_idx in range(num_frames):
            jpg = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
            if os.path.exists(jpg):
                img_np = np.array(PILImage.open(jpg).convert("RGB")).astype(np.float32) / 255.0
            else:
                img_np = np.zeros((h, w, 3), dtype=np.float32)

            img_tensor = torch.from_numpy(img_np)
            frame_mmap[frame_idx] = img_np

            if frame_idx in masks:
                frame_mask = masks[frame_idx]
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)

                vis_frame = img_tensor.clone()
                combined_mask = torch.zeros(h, w)

                if frame_mask.numel() == 0 or (frame_mask.dim() == 3 and frame_mask.shape[0] == 0):
                    frame_mask = torch.zeros(h, w)

                elif frame_mask.dim() == 3 and frame_mask.shape[0] >= 1:
                    num_objects = max(num_objects, frame_mask.shape[0])

                    if plot_all_masks:
                        for oid in range(frame_mask.shape[0]):
                            om = frame_mask[oid].float()
                            if om.max() > 1.0:
                                om = om / 255.0
                            color = torch.tensor(COLORS[oid % len(COLORS)])
                            mask_rgb = om.unsqueeze(-1) * color.view(1, 1, 3)
                            vis_frame = vis_frame * (1 - 0.5 * om.unsqueeze(-1)) + 0.5 * mask_rgb
                            combined_mask = torch.max(combined_mask, om)
                    else:
                        vis_oid = obj_id if 0 <= obj_id < frame_mask.shape[0] else 0
                        om = frame_mask[vis_oid].float()
                        if om.max() > 1.0:
                            om = om / 255.0
                        color = torch.tensor(COLORS[vis_oid % len(COLORS)])
                        mask_rgb = om.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - 0.5 * om.unsqueeze(-1)) + 0.5 * mask_rgb
                        for oid in range(frame_mask.shape[0]):
                            om2 = frame_mask[oid].float()
                            if om2.max() > 1.0:
                                om2 = om2 / 255.0
                            combined_mask = torch.max(combined_mask, om2)

                    if 0 <= obj_id < frame_mask.shape[0]:
                        output_mask = frame_mask[obj_id].float()
                        if output_mask.max() > 1.0:
                            output_mask = output_mask / 255.0
                    else:
                        output_mask = combined_mask
                    frame_mask = output_mask

                else:
                    if frame_mask.dim() == 3:
                        frame_mask = frame_mask.squeeze(0)
                    frame_mask = frame_mask.float()
                    if frame_mask.max() > 1.0:
                        frame_mask = frame_mask / 255.0
                    num_objects = max(num_objects, 1)
                    color = torch.tensor(COLORS[0])
                    mask_rgb = frame_mask.unsqueeze(-1) * color.view(1, 1, 3)
                    vis_frame = vis_frame * (1 - 0.5 * frame_mask.unsqueeze(-1)) + 0.5 * mask_rgb

                if frame_mask.numel() == 0:
                    frame_mask = torch.zeros(h, w)

                # ── legend: only draw if show_legend=True ──────────────────
                if show_legend and num_objects > 0:
                    legend_obj_id = -1 if plot_all_masks else obj_id
                    frame_scores = None
                    if scores is not None and frame_idx in scores:
                        fs = scores[frame_idx]
                        if hasattr(fs, 'tolist'):
                            frame_scores = fs.tolist()
                            if frame_scores and isinstance(frame_scores[0], list):
                                frame_scores = frame_scores[0]
                        elif hasattr(fs, '__iter__'):
                            frame_scores = list(fs)
                    vis_frame = self._draw_legend(vis_frame, num_objects, COLORS,
                                                  obj_id=legend_obj_id, frame_scores=frame_scores)

                vis_mmap[frame_idx]  = np.clip(vis_frame.numpy(), 0, 1)
                mask_mmap[frame_idx] = frame_mask.cpu().numpy()

            else:
                mask_mmap[frame_idx] = np.zeros((h, w), dtype=np.float32)
                vis_mmap[frame_idx]  = img_np

            if frame_idx % 50 == 0 and frame_idx > 0:
                mask_mmap.flush(); frame_mmap.flush(); vis_mmap.flush()
                gc.collect()

        mask_mmap.flush(); frame_mmap.flush(); vis_mmap.flush()

        all_masks  = torch.from_numpy(mask_mmap)
        all_frames = torch.from_numpy(frame_mmap)
        all_vis    = torch.from_numpy(vis_mmap)

        print(f"[SSSam3VideoOutput] done — num_objects={num_objects}, frames={num_frames}")
        return (all_masks, all_frames, all_vis)


# ── Registry ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SensoriumLayerInfo":   SensoriumLayerInfo,
    "SensoriumSelectLayer": SensoriumSelectLayer,
    "SSSam3VideoOutput":    SSSam3VideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SensoriumLayerInfo":   "Sensorium — Layer Info",
    "SensoriumSelectLayer": "Sensorium — Select Layer",
    "SSSam3VideoOutput":    "SS — SAM3 Video Output",
}
