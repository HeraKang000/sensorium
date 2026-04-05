#!/bin/bash
# ============================================================
# Sensorium — Provisioning Script
# github.com/HeraKang000/sensorium
#
# Pipeline: 360 video → Depth (DepthAnything v2) + Canny + Segmentation (SAM3)
# ============================================================

set -euo pipefail

COMFY_ROOT="/workspace/ComfyUI"
CUSTOM_NODES="$COMFY_ROOT/custom_nodes"
MODELS="$COMFY_ROOT/models"
LOG="/workspace/provisioning.log"

# ── HuggingFace: disable xet backend (avoids incomplete download failures)
#    and point cache to a known location we clean up after
export HF_HUB_DISABLE_XET=1
export HF_HOME="/workspace/.hf_cache"

exec > >(tee -a "$LOG") 2>&1
echo ""
echo "======================================================"
echo " Sensorium Provisioning — $(date)"
echo "======================================================"

# ── helpers ──────────────────────────────────────────────────
green()  { echo -e "\033[32m[OK]\033[0m  $*"; }
yellow() { echo -e "\033[33m[--]\033[0m  $*"; }
red()    { echo -e "\033[31m[ERR]\033[0m $*"; }

pip_quiet() { pip install -q --no-warn-script-location "$@"; }

clone_or_update() {
    local NAME=$1 URL=$2
    local DIR="$CUSTOM_NODES/$NAME"
    if [ ! -d "$DIR/.git" ]; then
        echo "  Cloning $NAME..."
        if git clone --depth 1 "$URL" "$DIR"; then
            git -C "$DIR" checkout -f HEAD 2>/dev/null || true
            [ -f "$DIR/requirements.txt" ] && pip_quiet -r "$DIR/requirements.txt"
            green "$NAME installed"
        else
            red "$NAME clone failed — skipping"
        fi
    else
        yellow "$NAME exists — pulling"
        git -C "$DIR" pull --ff-only 2>/dev/null || true
    fi
}

dl_hf() {
    local REPO=$1 FILE=$2 DIR=$3
    local TARGET="$DIR/$(basename "$FILE")"
    if [ -f "$TARGET" ]; then
        yellow "EXISTS  $(basename "$FILE")"
        return
    fi
    echo "  Downloading $(basename "$FILE") from $REPO ..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$REPO', filename='$FILE', local_dir='$DIR')
"
    green "$(basename "$FILE")"
}

# ── 1. System packages ────────────────────────────────────────
echo ""
echo "── 1. System packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget curl > /dev/null
green "System packages ready"

# ── 2. Python packages ────────────────────────────────────────
echo ""
echo "── 2. Python packages"
pip_quiet huggingface_hub hf_transfer
pip_quiet decord opencv-python-headless "imageio[ffmpeg]"
pip_quiet einops

# ComfyUI-AutoVideoMasking deps
pip_quiet groundingdino-py supervision
pip_quiet google-genai

# ONNX runtime for YOLO detection
pip_quiet onnxruntime-gpu

green "Python packages ready"

# ── 3. ComfyUI ───────────────────────────────────────────────
echo ""
echo "── 3. ComfyUI"
if [ ! -d "$COMFY_ROOT/.git" ]; then
    echo "  Cloning ComfyUI..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY_ROOT"
    pip_quiet -r "$COMFY_ROOT/requirements.txt"
    green "ComfyUI installed"
else
    yellow "ComfyUI exists — pulling"
    git -C "$COMFY_ROOT" pull --ff-only 2>/dev/null || true
fi

mkdir -p \
    "$MODELS/sam3" \
    "$MODELS/detection"

# ── 4. Custom nodes ──────────────────────────────────────────
echo ""
echo "── 4. Custom nodes"

clone_or_update "ComfyUI-VideoHelperSuite" \
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"

clone_or_update "ComfyUI-KJNodes" \
    "https://github.com/kijai/ComfyUI-KJNodes"

# Depth + Canny preprocessors (DepthAnything v2, Canny, etc.)
clone_or_update "comfyui_controlnet_aux" \
    "https://github.com/Fannovel16/comfyui_controlnet_aux"

# SAM3 base nodes: LoadSAM3Model, SAM3Segmentation
clone_or_update "ComfyUI-SAM3" \
    "https://github.com/PozzettiAndrea/ComfyUI-SAM3"

# SAMhera extended nodes: SAM3Grounding, SAM3VideoInitialize, SAM3VideoPropagate
clone_or_update "ComfyUI-AutoVideoMasking" \
    "https://github.com/ZeroSpaceStudios/ComfyUI-AutoVideoMasking"

AVM_DIR="$CUSTOM_NODES/ComfyUI-AutoVideoMasking"
if [ -f "$AVM_DIR/requirements.txt" ]; then
    echo "  Ensuring ComfyUI-AutoVideoMasking requirements..."
    pip_quiet -r "$AVM_DIR/requirements.txt"
    green "ComfyUI-AutoVideoMasking requirements OK"
fi

clone_or_update "ComfyUI-Manager" \
    "https://github.com/Comfy-Org/ComfyUI-Manager"

echo "security_level = weak" > "$CUSTOM_NODES/ComfyUI-Manager/config.ini"

# Sensorium custom nodes — symlink from cloned repo into custom_nodes
SENSORIUM_DIR="/workspace/sensorium"
if [ -d "$SENSORIUM_DIR/custom_nodes/sensorium_nodes" ]; then
    ln -sf "$SENSORIUM_DIR/custom_nodes/sensorium_nodes" "$CUSTOM_NODES/sensorium_nodes"
    green "Sensorium nodes linked"
fi

green "All custom nodes ready"

# ── 5. Models ────────────────────────────────────────────────
echo ""
echo "── 5. Models"

echo "  [SAM3 checkpoint — apozz/sam3-safetensors]"
dl_hf "apozz/sam3-safetensors" \
    "sam3.safetensors" \
    "$MODELS/sam3"

echo "  [ONNX: YOLOv10m — for SAM3Grounding text-prompted detection]"
dl_hf "onnx-community/yolov10m" \
    "onnx/model.onnx" \
    "$MODELS/detection"
python3 -c "
import shutil, os
src = '/workspace/ComfyUI/models/detection/onnx/model.onnx'
dst = '/workspace/ComfyUI/models/detection/yolov10m.onnx'
if os.path.exists(src) and not os.path.exists(dst):
    shutil.move(src, dst)
"

# DepthAnything v2 weights are auto-downloaded by comfyui_controlnet_aux on first use

green "All models downloaded"

# ── 6. Clean up HF cache ─────────────────────────────────────
echo ""
echo "── 6. Cleaning HF cache"
rm -rf "$HF_HOME"
green "HF cache cleared"

# ── 7. Launch ComfyUI ────────────────────────────────────────
echo ""
echo "── 7. Launching ComfyUI"
pkill -f "python.*main.py" 2>/dev/null || true
sleep 1

nohup python3 "$COMFY_ROOT/main.py" \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    >> /workspace/comfyui.log 2>&1 &

echo "  PID: $!"
echo "  Log: /workspace/comfyui.log"

echo ""
echo "======================================================"
green "Sensorium provisioning complete — $(date)"
echo "======================================================"
echo ""
echo "  Pipeline: 360 video → Depth (DepthAnything v2) + Canny + Segmentation (SAM3)"
echo ""
echo "  Custom nodes:"
echo "    comfyui_controlnet_aux   — DepthAnythingV2Preprocessor, CannyEdgePreprocessor"
echo "    ComfyUI-SAM3             — LoadSAM3Model, SAM3Segmentation"
echo "    ComfyUI-AutoVideoMasking — SAM3Grounding, SAM3VideoInitialize, SAM3VideoPropagate"
echo "    ComfyUI-VideoHelperSuite — VHS_LoadVideo, VHS_VideoCombine"
echo ""
