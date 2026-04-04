# Sensorium — Project Conversation Log

## Session 1 — 2026-04-04

### Initial Vision

**User:** Building a groundbreaking spatial computing / filmmaking tool called **Sensorium**.

- Preferred build tools: **ComfyUI**, **Blender**, etc.
- Core concept: a new tool for **video making with sensor and perceptual input**
- Goal: bridge the physical/sensory world with generative media pipelines

---

### Clarifying Questions (from Claude)

Before architecting, the following questions were raised:

**1. Sensor inputs?**
- IMU/motion capture (body movement, head tracking)?
- Biometric (heart rate, EEG, breath)?
- LiDAR / depth cameras (spatial mapping)?
- Microphone / audio analysis?
- Eye tracking?
- All of the above as modular inputs?

**2. Primary output modality?**
- Real-time generative video (ComfyUI pipelines triggered/guided by sensors)?
- 3D scene composition in Blender driven by sensor data?
- A timeline/edit layer that maps sensor events to film cuts/effects?
- Live performance → rendered film?

**3. Core interaction model?**
- An artist performs/moves and the system generates/responds?
- Sensor data as a "score" that plays back through generative models?
- Node-based patching (like Max/MSP or TouchDesigner) connecting sensors to generators?

**4. Target hardware?**
- Apple Vision Pro / visionOS?
- Standard webcam + pose estimation?
- Custom sensors?
- Mobile (iPhone LiDAR)?

**5. First demo / v0 goal?**
- What is the one thing you want to be able to *do* with it first?

---

---

### Answers

#### 1. Sensor / Media Inputs

Tiered input system — simple inputs elevated by ML, advanced inputs supported natively:

| Basic Input | ML Elevation Layer | Advanced Input |
|---|---|---|
| Standard video | Monocular depth estimation | ZED cam / depth camera |
| Flat video | NeRF / Gaussian Splatting | Volumetric data |
| 360 video | Spatial audio inference | Apple Immersive Video |
| Still image / photo | Outpainting / 3D lift | 360 Image |
| — | LiDAR reconstruction | LiDAR point cloud data |

**Design principle:** ML as a bridge — someone with just a phone enters the same pipeline as someone with professional volumetric gear. Start simple, expand through models or hardware.

Confirmed input types:
- Basic video
- 360 video
- Depth camera (ZED cam)
- LiDAR data
- Apple Immersive Video
- Basic photo / image
- 360 image
- Volumetric data

---

#### 2. Primary Output Modality

**Mixed Reality — interactive, extended reality, universal delivery.**

Two parallel output streams:

| Human-Consumable | Machine-Consumable |
|---|---|
| Spatial video (MV-HEVC, EAC 360) | USD / glTF scenes |
| Projection-mapped video | OpenXR / WebXR streams |
| Real-time XR (Meta Quest 3, AVP) | Unity / Unreal packages |
| Web (WebGL / WebGPU / WebXR) | OSC / data streams |
| Live performance stream | Platform templates |
| Large-scale / multi-channel projection | — |
| In-vehicle displays | — |

**Universal format bridge:** USD (Universal Scene Description) — supported by Apple, NVIDIA, Pixar, Blender, Unity, Unreal.

**Design principle:** Author once, deliver everywhere. Platform templates per target (Quest, AVP, Web, Unity, Unreal, projection).

#### 3. Core Interaction Model

**Semantic filmmaking — "Who Framed Roger Rabbit", automated.**

Not rotoscoping or manual masking. The tool understands the scene semantically (depth, segmentation, who/what is in frame), and characters/VFX are placed with rules that respond automatically to that understanding. Artistic and cinematic value — the director/editor works at the level of narrative and presence, not pixel labor.

Key distinction from Niantic/Pokemon Go: this is about **cinematic and artistic authorship**, not geospatial information overlay.

As sensors advance (basic cam → ZED → LiDAR → volumetric), the tool's understanding deepens — same authoring model, richer spatial data.

**Core Architecture:**

```
INPUT LAYER
  standard video / 360 / depth cam (ZED) / LiDAR / Apple Immersive / image
       ↓
UNDERSTANDING LAYER  ← the heart of Sensorium
  depth estimation (DepthAnything / ZED SDK / Apple Depth API)
  segmentation (SAM2 — Segment Anything Model, video-aware)
  person / object detection (YOLO / Grounding DINO)
  scene semantics → spatial map
       ↓
AUTHORING LAYER
  director places characters / VFX / generative elements
  ComfyUI pipelines (style, generation, transformation)
  Blender (geometry, rendering, Gaussian Splatting)
  interaction rules (character reacts to person, depth, light, etc.)
       ↓
INTERACTION ENGINE
  real-time compositing respecting scene depth + segmentation
  character/VFX responds to scene events
       ↓
OUTPUT LAYER
  Web (WebXR / WebGPU)
  Meta Quest 3 / AVP
  USD export → Unity / Unreal templates
  Multi-channel projection
  Flat video / 360 video / Apple Immersive Video
  In-vehicle displays / Live performance
```

**Anchoring project:** Mixed Reality film "Revealing Rendering"
- Explores how humanity has outsourced vision to rendering machines
- Examines the etymology of "rendering" — to give back AND to surrender
- Mirrors → screens, cameras → sensors, images becoming operational
- Avant-garde / feminist film traditions applied to immersive media
- Sensorium is both the tool AND an embodiment of the film's thesis

**Funding ambition:** Doris Duke Foundation, Guggenheim — artistic-technical tool making

#### 4. Hardware & Tool Philosophy

**Development machine:** M1 MacBook Air (primary)
**Remote GPU:** RTX 3080 Super (Seoul) / RTX 4090 (LA, remote accessible from Korea)
**GPU rental:** Vast.ai (familiar workflow)

**Tool philosophy:** Light and fast. Web-based first. App possible. No heavy desktop software unless funded. The browser app does zero inference — all heavy work routes to remote GPU.

**Existing work — SAMhera (ComfyUI workflow):**
- Gemini VLM → enumerate/localize objects in video (2-stage: label then bounding box)
- SAM3 → propagate segmentation masks across video timeline
- Outputs per-object mask sequences for compositing
- Already eliminates manual bounding box / point placement
- **This IS the Understanding Layer of Sensorium**

**Full Proposed Architecture:**

```
SENSORIUM WEB APP  (browser — light, fast, runs anywhere)
  drag-drop media input / timeline view / XR preview (Three.js / WebXR)
       ↕  WebSocket + REST
PYTHON ORCHESTRATOR  (FastAPI — thin local or remote process)
  ├── ComfyUI API  ← SAMHERA WORKFLOW (existing foundation)
  │     ├── Gemini VLM → scene understanding
  │     ├── SAM3 → per-object video segmentation
  │     ├── DepthAnything → depth map (to add)
  │     └── Generative pipelines (img2img, video, style)
  ├── Blender Python API  (scene composition, USD export)
  └── Output formatters
        ├── WebXR / Three.js scene (Quest 3, AVP, browser)
        ├── 360 / EAC video
        ├── USD → Unity / Unreal templates
        └── Flat video (ffmpeg)

REMOTE GPU  (Seoul RTX 3080 / LA RTX 4090 / Vast.ai)
  ComfyUI server — SAMhera + extended Sensorium workflows
  accessible via SSH tunnel or ngrok
```

#### 5. First Demo / v0 Goal

To be defined after structure is confirmed.

---

### Architecture Notes

**Build order:**
1. Python FastAPI orchestrator wrapping ComfyUI API
2. Extend SAMhera workflow with DepthAnything node
3. Lightweight web frontend (media upload → job status → preview)
4. Three.js spatial preview + WebXR for Quest 3
5. Character/VFX placement authoring layer
6. USD export → Unity/Unreal templates

**Key dependencies:**
- ComfyUI (existing, with SAMhera)
- Gemini API (existing)
- SAM3 (existing in workflow)
- DepthAnything v2 (to add)
- FastAPI + WebSocket
- Three.js / WebXR
- Blender Python API
- ffmpeg

---

*Log continues as conversation progresses.*
