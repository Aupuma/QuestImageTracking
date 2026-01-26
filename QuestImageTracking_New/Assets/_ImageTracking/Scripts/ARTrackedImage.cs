using UnityEngine;
using OpenCVMarkerLessAR;
using OpenCVForUnity.UnityUtils;
using System.Collections.Generic;
using OpenCVForUnity.UnityIntegration;

[System.Serializable]
public class ARTrackedImage
{
    [Tooltip("Data for this tracked image")]
    public ARTrackedImageData data;

    [Tooltip("The GameObject to be updated with the tracked pose")]
    public GameObject virtualObject;

    // Runtime data - not serialized
    [System.NonSerialized]
    public Pattern pattern;

    [System.NonSerialized]
    public PatternTrackingInfo trackingInfo;

    [System.NonSerialized]
    public bool isDetected;
    
    [System.NonSerialized]
    public OpenCVARUtils.PoseData prevPose;
    
    // Static mode stabilization data
    [HideInInspector] public bool isStabilized = false;
    [HideInInspector] public List<OpenCVARUtils.PoseData> recentPoses = new List<OpenCVARUtils.PoseData>();
    [HideInInspector] public int similarPoseCount = 0;

    public ARTrackedImage()
    {
        trackingInfo = new PatternTrackingInfo();
        prevPose = new OpenCVARUtils.PoseData();
        pattern = new Pattern();
    }
}