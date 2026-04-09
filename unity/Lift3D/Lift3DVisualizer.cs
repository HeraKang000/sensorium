using System.Collections.Generic;
using UnityEngine;
using Sensorium.Lift3D;

namespace Sensorium.Lift3D
{
    /// <summary>
    /// Spawns sphere joints and LineRenderer bones for each detected person,
    /// and plays back all frames at a configurable FPS.
    /// </summary>
    public class Lift3DVisualizer : MonoBehaviour
    {
        [Header("References")]
        public Lift3DLoader loader;

        [Header("Playback")]
        public float playbackFPS   = 24f;
        public bool  loop          = true;
        public bool  autoPlay      = true;

        [Header("Visuals")]
        public float jointRadius   = 0.05f;
        public float boneWidth     = 0.02f;
        public float depthScale    = 100f;
        public Color jointColor    = new Color(0.2f, 1f, 0.4f);
        public Color boneColor     = new Color(1f, 1f, 1f, 0.7f);
        public Color lowConfColor  = new Color(1f, 0.3f, 0.3f, 0.5f);
        public float confidenceCutoff = 0.3f;

        // ── state ─────────────────────────────────────────────────────────────
        int   _currentFrame;
        float _timer;
        bool  _playing;

        // ── internal pools ────────────────────────────────────────────────────
        readonly List<GameObject>   _jointObjects  = new();
        readonly List<LineRenderer> _boneRenderers = new();

        Material _jointMat;
        Material _boneMat;

        // ── Unity lifecycle ───────────────────────────────────────────────────

        void Awake()
        {
            _jointMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            _boneMat  = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        }

        void Start()
        {
            if (loader == null) return;
            if (loader.Frames.Count > 0 && autoPlay)
                StartPlayback();
        }

        void Update()
        {
            if (!_playing || loader == null || loader.Frames.Count == 0) return;

            _timer += Time.deltaTime;
            float frameDuration = 1f / Mathf.Max(playbackFPS, 0.1f);

            if (_timer >= frameDuration)
            {
                _timer -= frameDuration;
                AdvanceFrame();
            }
        }

        void OnDestroy()
        {
            if (_jointMat != null) Destroy(_jointMat);
            if (_boneMat  != null) Destroy(_boneMat);
        }

        // ── public API ────────────────────────────────────────────────────────

        public void StartPlayback()
        {
            _currentFrame = 0;
            _timer        = 0f;
            _playing      = true;
            ShowFrame(_currentFrame);
        }

        public void StopPlayback() => _playing = false;

        public void Visualize(Lift3DFrame frame)
        {
            ClearAll();
            if (frame?.people == null) return;
            foreach (var person in frame.people)
                SpawnPerson(person);
        }

        // ── private helpers ───────────────────────────────────────────────────

        void AdvanceFrame()
        {
            _currentFrame++;
            if (_currentFrame >= loader.Frames.Count)
            {
                if (loop) _currentFrame = 0;
                else { _playing = false; return; }
            }
            ShowFrame(_currentFrame);
        }

        void ShowFrame(int index)
        {
            if (loader.Frames == null || loader.Frames.Count == 0) return;
            var frame = loader.Frames[Mathf.Clamp(index, 0, loader.Frames.Count - 1)];
            Visualize(frame);
        }

        void SpawnPerson(Person3D person)
        {
            if (person.keypoints_3d == null) return;

            var kpPositions = new Dictionary<string, Vector3>();

            foreach (var (name, kp) in person.keypoints_3d)
            {
                if (!kp.IsValid || kp.confidence < confidenceCutoff) continue;

                Vector3 pos = kp.ToVector3() * depthScale;
                kpPositions[name] = pos;

                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = $"P{person.person_id}_{name}";
                go.transform.SetParent(transform, worldPositionStays: false);
                go.transform.localPosition = pos;
                go.transform.localScale    = Vector3.one * jointRadius * 2f;
                Destroy(go.GetComponent<Collider>());

                var r = go.GetComponent<Renderer>();
                r.material       = _jointMat;
                r.material.color = kp.confidence >= 0.6f ? jointColor : lowConfColor;
                _jointObjects.Add(go);
            }

            foreach (var (a, b) in SkeletonDef.Bones)
            {
                if (!kpPositions.TryGetValue(a, out var posA)) continue;
                if (!kpPositions.TryGetValue(b, out var posB)) continue;

                var go = new GameObject($"Bone_{a}_{b}");
                go.transform.SetParent(transform, worldPositionStays: false);

                var lr = go.AddComponent<LineRenderer>();
                lr.useWorldSpace = false;
                lr.positionCount = 2;
                lr.startWidth    = boneWidth;
                lr.endWidth      = boneWidth;
                lr.material      = _boneMat;
                lr.material.color = boneColor;
                lr.SetPosition(0, posA);
                lr.SetPosition(1, posB);

                _boneRenderers.Add(lr);
                _jointObjects.Add(go);
            }
        }

        void ClearAll()
        {
            foreach (var go in _jointObjects)
                if (go != null) Destroy(go);
            _jointObjects.Clear();
            _boneRenderers.Clear();
        }
    }
}
