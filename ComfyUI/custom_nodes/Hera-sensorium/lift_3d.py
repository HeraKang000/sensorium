"""
Lift3D — lifts OpenPose 2D keypoints to 3D using a Depth Anything v2 depth map.

Exports a Unity-importable JSON file and returns:
  - keypoints_json  : STRING  (full JSON as string)
  - people_count    : INT     (number of detected people)
  - preview_image   : IMAGE   (depth map with keypoints overlaid)
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

# ── COCO-18 keypoint index → name mapping ─────────────────────────────────────
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    # index 17 is "background" / neck in some variants — keep as unnamed but
    # still process so index arithmetic stays correct
    "neck",
]

# Colour used to draw each keypoint on the preview (R, G, B float 0-1)
_KP_COLOUR = (0.0, 1.0, 0.0)   # bright green
_DOT_RADIUS = 4                 # pixels


# ── helpers ────────────────────────────────────────────────────────────────────

def _draw_dot(canvas: np.ndarray, cx: int, cy: int, radius: int, colour) -> None:
    """Draw a filled circle on an (H, W, 3) float32 canvas in-place."""
    h, w = canvas.shape[:2]
    r, g, b = colour
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                canvas[y, x] = (r, g, b)


def _depth_map_to_hw(depth_map: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Accept depth_map in any of the shapes ComfyUI may deliver:
      (H, W), (1, H, W), (B, H, W), (B, H, W, C)
    Returns a (target_h, target_w) float32 tensor, resized if necessary.
    """
    t = depth_map.float()

    # Normalise to (1, 1, H, W) for F.interpolate
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    elif t.dim() == 3:
        t = t[0].unsqueeze(0).unsqueeze(0)       # take first batch/channel
    elif t.dim() == 4:
        # Could be (B,H,W,C) — ComfyUI IMAGE convention
        if t.shape[-1] <= 4:                     # last dim is channel
            t = t[0, :, :, 0].unsqueeze(0).unsqueeze(0)
        else:
            t = t[0, 0].unsqueeze(0).unsqueeze(0)

    h, w = t.shape[2], t.shape[3]
    if h != target_h or w != target_w:
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)

    return t[0, 0]   # (H, W)


# ── node ───────────────────────────────────────────────────────────────────────

class Lift3DNode:
    """ComfyUI node: Lift 3D (Pose + Depth → Unity)"""

    CATEGORY = "Sensorium/Lift"
    RETURN_TYPES = ("STRING", "INT", "IMAGE")
    RETURN_NAMES = ("keypoints_json", "people_count", "preview_image")
    FUNCTION = "lift"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "depth_map":      ("IMAGE",),
                "image_width":    ("INT",    {"default": 2048, "min": 1,   "max": 16384}),
                "image_height":   ("INT",    {"default": 1024, "min": 1,   "max": 8192}),
                "confidence_threshold": (
                    "FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "output_path":    ("STRING", {"default": "lift3d_output.json"}),
                "frame_index":    ("INT",    {"default": 0,    "min": 0,   "max": 999999}),
            }
        }

    # ------------------------------------------------------------------

    def lift(
        self,
        pose_keypoints,
        depth_map: torch.Tensor,
        image_width: int,
        image_height: int,
        confidence_threshold: float,
        output_path: str,
        frame_index: int,
    ):
        W, H = image_width, image_height

        # Camera intrinsics (pinhole estimate)
        fx = fy = float(W)
        cx = W / 2.0
        cy = H / 2.0

        # ── prepare depth map ─────────────────────────────────────────
        depth_hw = _depth_map_to_hw(depth_map, H, W)          # (H, W) float32
        depth_np = depth_hw.cpu().numpy()

        # ── build preview canvas from depth map ───────────────────────
        # depth_np is 0-1; convert to RGB (H, W, 3)
        preview = np.stack([depth_np, depth_np, depth_np], axis=-1)

        # ── lift each person ──────────────────────────────────────────
        people_out = []

        # pose_keypoints may be a list or a single dict depending on the
        # upstream node. Normalise to a list-of-people list.
        if isinstance(pose_keypoints, dict):
            # Some nodes wrap as {"people": [...]}
            raw_people = pose_keypoints.get("people", [])
        elif isinstance(pose_keypoints, list):
            # Could be [[kp…], [kp…]] or [{"pose_keypoints_2d": [...]}]
            raw_people = pose_keypoints
        else:
            raw_people = []

        for person_id, person in enumerate(raw_people):
            # Accept either a flat list of 54 values or the OpenPose dict
            if isinstance(person, dict):
                flat = person.get("pose_keypoints_2d", [])
            elif isinstance(person, (list, tuple)):
                flat = list(person)
            else:
                continue

            kp_3d = {}
            for kp_idx, name in enumerate(KEYPOINT_NAMES):
                base = kp_idx * 3
                if base + 2 >= len(flat):
                    kp_3d[name] = {"x": None, "y": None, "z": None, "confidence": 0.0}
                    continue

                u_norm = float(flat[base])
                v_norm = float(flat[base + 1])
                conf   = float(flat[base + 2])

                if conf <= confidence_threshold or conf == 0.0:
                    kp_3d[name] = {"x": None, "y": None, "z": None, "confidence": conf}
                    continue

                # Pixel coordinates
                u = u_norm * W
                v = v_norm * H

                # Clamp to valid range
                ui = int(np.clip(round(u), 0, W - 1))
                vi = int(np.clip(round(v), 0, H - 1))

                depth_val = float(depth_np[vi, ui])

                x3d = (u - cx) * depth_val / fx
                y3d = (v - cy) * depth_val / fy
                z3d = depth_val

                kp_3d[name] = {
                    "x": round(x3d, 6),
                    "y": round(y3d, 6),
                    "z": round(z3d, 6),
                    "confidence": round(conf, 4),
                }

                # Draw dot on preview
                _draw_dot(preview, ui, vi, _DOT_RADIUS, _KP_COLOUR)

            people_out.append({"person_id": person_id, "keypoints_3d": kp_3d})

        # ── assemble output dict ──────────────────────────────────────
        result = {"frame": frame_index, "people": people_out}
        json_str = json.dumps(result, indent=2)

        # ── write JSON file ───────────────────────────────────────────
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(json_str)

        # ── convert preview to ComfyUI IMAGE tensor (B,H,W,3) ────────
        preview_clipped = np.clip(preview, 0.0, 1.0).astype(np.float32)
        preview_tensor = torch.from_numpy(preview_clipped).unsqueeze(0)  # (1,H,W,3)

        return (json_str, len(people_out), preview_tensor)
