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

### Next Steps

Awaiting answers to the above questions to determine the right starting point:
- Python orchestration layer
- Web-based node editor
- Blender addon
- ComfyUI custom node pack

---

*Log continues as conversation progresses.*
