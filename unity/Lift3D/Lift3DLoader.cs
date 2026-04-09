using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;
using Sensorium.Lift3D;

namespace Sensorium.Lift3D
{
    /// <summary>
    /// Loads a lift3d .jsonl file (one JSON frame per line) and exposes
    /// the parsed frame list to other components.
    /// </summary>
    public class Lift3DLoader : MonoBehaviour
    {
        [Header("Source")]
        [Tooltip("Absolute path to the .jsonl written by the ComfyUI Lift3D node.")]
        public string jsonFilePath = @"C:\ComfyUI\output\lift3d_output.jsonl";

        [Header("Events")]
        public UnityEngine.Events.UnityEvent<Lift3DFrame> onFrameLoaded;

        public List<Lift3DFrame> Frames  { get; private set; } = new();
        public Lift3DFrame CurrentFrame  { get; private set; }

        // ── Unity lifecycle ───────────────────────────────────────────────────

        void Start() => LoadFile(jsonFilePath);

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
                var lines = File.ReadAllLines(path);
                Frames.Clear();

                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var frame = JsonConvert.DeserializeObject<Lift3DFrame>(line);
                    if (frame != null) Frames.Add(frame);
                }

                Debug.Log($"[Lift3DLoader] Loaded {Frames.Count} frames from {path}");

                if (Frames.Count > 0)
                {
                    CurrentFrame = Frames[0];
                    onFrameLoaded?.Invoke(CurrentFrame);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[Lift3DLoader] Parse error: {e.Message}");
            }
        }

        public void SetFrame(int index)
        {
            if (Frames == null || Frames.Count == 0) return;
            index = Mathf.Clamp(index, 0, Frames.Count - 1);
            CurrentFrame = Frames[index];
            onFrameLoaded?.Invoke(CurrentFrame);
        }
    }
}
