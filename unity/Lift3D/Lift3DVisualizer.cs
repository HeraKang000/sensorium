using System.Collections.Generic;
using UnityEngine;
using Sensorium.Lift3D;

namespace Sensorium.Lift3D
{
    /// <summary>
    /// Spawns sphere joints and LineRenderer bones for each detected person.
    ///
    /// Wire up: add Lift3DLoader to a GameObject, then drag this component's
    /// GameObject into Lift3DLoader → onFrameLoaded. Or call Visualize() manually.
    ///
    /// Requires URP. Joints use URP/Lit, bones use URP/Unlit.
    /// Scale: the depth values from Depth Anything are 0-1 (relative).
    /// Use depthScale to map to world units that make sense in your scene.
    ///</summary>
    public class Lift3DVisualizer : MonoBehaviour
    {
        [Header("References")]
        public Lift3DLoader loader;

        [Header("Visuals")]
        public float jointRadius   = 0.02f;
        public float boneWidth     = 0.008f;
        public float depthScale    = 2f;     // multiply all z values
        public Color jointColor    = new Color(0.2f, 1f, 0.4f);
        public Color boneColor     = new Color(1f, 1f, 1f, 0.7f);
        public Color lowConfColor  = new Color(1f, 0.3f, 0.3f, 0.5f);
        public float confidenceCutoff = 0.3f;

        // ── internal pools ────────────────────────────────────────────────────
        readonly List<GameObject>     _jointObjects = new();
        readonly List<LineRenderer>   _boneRenderers = new();

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
            if (loader != null)
            {
                loader.onFrameLoaded.AddListener(Visualize);
                if (loader.CurrentFrame != null)
                    Visualize(loader.CurrentFrame);
            }
        }

        void OnDestroy()
        {
            if (_jointMat != null) Destroy(_jointMat);
            if (_boneMat  != null) Destroy(_boneMat);
        }

        // ── public API ────────────────────────────────────────────────────────

        public void Visualize(Lift3DFrame frame)
        {
            ClearAll();

            if (frame?.people == null) return;

            foreach (var person in frame.people)
                SpawnPerson(person);
        }

        // ── private helpers ───────────────────────────────────────────────────

        void SpawnPerson(Person3D person)
        {
            if (person.keypoints_3d == null) return;

            var kpPositions = new Dictionary<string, Vector3>();

            // ── joints ────────────────────────────────────────────────────────
            foreach (var (name, kp) in person.keypoints_3d)
            {
                if (!kp.IsValid || kp.confidence < confidenceCutoff) continue;

                Vector3 pos = kp.ToVector3() * depthScale;
                kpPositions[name] = pos;

                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = $"Person{person.person_id}_{name}";
                go.transform.SetParent(transform, worldPositionStays: false);
                go.transform.localPosition = pos;
                go.transform.localScale    = Vector3.one * jointRadius * 2f;

                Destroy(go.GetComponent<Collider>());  // no physics needed

                var r = go.GetComponent<Renderer>();
                r.material = _jointMat;
                r.material.color = kp.confidence >= 0.6f ? jointColor : lowConfColor;

                _jointObjects.Add(go);
            }

            // ── bones ─────────────────────────────────────────────────────────
            foreach (var (a, b) in SkeletonDef.Bones)
            {
                if (!kpPositions.TryGetValue(a, out var posA)) continue;
                if (!kpPositions.TryGetValue(b, out var posB)) continue;

                var go = new GameObject($"Bone_{a}_{b}");
                go.transform.SetParent(transform, worldPositionStays: false);

                var lr = go.AddComponent<LineRenderer>();
                lr.useWorldSpace    = false;
                lr.positionCount    = 2;
                lr.startWidth       = boneWidth;
                lr.endWidth         = boneWidth;
                lr.material         = _boneMat;
                lr.material.color   = boneColor;
                lr.SetPosition(0, posA);
                lr.SetPosition(1, posB);

                _boneRenderers.Add(lr);
                _jointObjects.Add(go);   // track for cleanup
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
