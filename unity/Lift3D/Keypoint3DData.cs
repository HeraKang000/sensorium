using System;
using System.Collections.Generic;

namespace Sensorium.Lift3D
{
    // ── JSON schema matches lift_3d.py output exactly ─────────────────────────

    [Serializable]
    public class Keypoint3D
    {
        public float? x;
        public float? y;
        public float? z;
        public float confidence;

        public bool IsValid => x.HasValue && y.HasValue && z.HasValue && confidence > 0f;

        public UnityEngine.Vector3 ToVector3() =>
            IsValid ? new UnityEngine.Vector3(x.Value, -y.Value, z.Value) // flip Y: image→Unity
                    : UnityEngine.Vector3.zero;
    }

    [Serializable]
    public class Person3D
    {
        public int person_id;
        public Dictionary<string, Keypoint3D> keypoints_3d;
    }

    [Serializable]
    public class Lift3DFrame
    {
        public int frame;
        public List<Person3D> people;
    }

    // ── COCO-18 skeleton bone pairs ───────────────────────────────────────────
    public static class SkeletonDef
    {
        public static readonly (string, string)[] Bones =
        {
            ("nose",          "left_eye"),
            ("nose",          "right_eye"),
            ("left_eye",      "left_ear"),
            ("right_eye",     "right_ear"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow",    "left_wrist"),
            ("right_shoulder","right_elbow"),
            ("right_elbow",   "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder","right_hip"),
            ("left_hip",      "right_hip"),
            ("left_hip",      "left_knee"),
            ("left_knee",     "left_ankle"),
            ("right_hip",     "right_knee"),
            ("right_knee",    "right_ankle"),
        };
    }
}
