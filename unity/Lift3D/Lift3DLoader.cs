using System.IO;
using UnityEngine;
using Newtonsoft.Json;          // install via Package Manager: com.unity.nuget.newtonsoft-json
using Sensorium.Lift3D;

namespace Sensorium.Lift3D
{
    /// <summary>
    /// Reads a lift3d_output.json file (single frame or updated every frame)
    /// and exposes the parsed data to other components.
    ///
    /// Attach to any GameObject. Assign jsonFilePath in the Inspector,
    /// or call LoadFile(path) from another script.
    /// </summary>
    public class Lift3DLoader : MonoBehaviour
    {
        [Header("Source")]
        [Tooltip("Absolute path to the JSON written by the ComfyUI Lift3D node.")]
        public string jsonFilePath = @"C:\ComfyUI\output\lift3d_output.json";

        [Tooltip("If true, re-reads the file every frame (live ComfyUI feed).")]
        public bool liveReload = false;

        [Header("Events")]
        public UnityEngine.Events.UnityEvent<Lift3DFrame> onFrameLoaded;

        public Lift3DFrame CurrentFrame { get; private set; }

        // ── Unity lifecycle ───────────────────────────────────────────────────

        void Start() => LoadFile(jsonFilePath);

        void Update()
        {
            if (liveReload)
                LoadFile(jsonFilePath);
        }

        // ── public API ────────────────────────────────────────────────────────

        public void LoadFile(string path)
        {
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[Lift3DLoader] File not found: {path}");
                return;
            }

            try
            {
                string json = File.ReadAllText(path);
                var frame = JsonConvert.DeserializeObject<Lift3DFrame>(json);
                CurrentFrame = frame;
                Debug.Log($"[Lift3DLoader] Loaded frame {frame?.frame}, {frame?.people?.Count ?? 0} people.");
                onFrameLoaded?.Invoke(frame);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[Lift3DLoader] Parse error: {e.Message}");
            }
        }
    }
}
