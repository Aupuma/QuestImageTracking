using UnityEngine;

[CreateAssetMenu(fileName = "ARTrackedImageData", menuName = "AR/Tracked Image Data")]
public class ARTrackedImageData : ScriptableObject
{
    [Tooltip("Unique identifier for this marker")]
    public string id;

    [Tooltip("The image that will be tracked")]
    public Texture2D markerTexture;

    [Tooltip("Physical size of the marker in meters")]
    public float physicalSize = 0.1f;
}
