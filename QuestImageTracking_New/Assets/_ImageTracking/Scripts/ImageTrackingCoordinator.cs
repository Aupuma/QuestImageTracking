using System.Collections;
using UnityEngine;
using Meta.XR;
using TMPro;

/// <summary>
/// A coordinator that sets up the Quest passthrough camera (using PassthroughCameraAccess),
/// initializes the ImageTracking component with the proper camera calibration values,
/// and runs the image tracking detection in the Update loop.
/// </summary>
public class ImageTrackingCoordinator : MonoBehaviour
{
    [Header("Passthrough Camera")]
    [Tooltip("Component that manages the camera feed.")]
    [SerializeField] private PassthroughCameraAccess _passthroughCameraAccess;

    [Header("Image Tracking")]
    [Tooltip("The ImageTracking component that handles pattern detection.")]
    [SerializeField] private ImageTracking _imageTracking;
    [SerializeField] private TextMeshPro _modeText;
    
    private Transform _cameraAnchor;
    private Texture2D _resultTexture;
    private bool _isDebugMode = false;
    GameObject[] _arHelpers;

    private IEnumerator Start()
    {
        _arHelpers = GameObject.FindGameObjectsWithTag("ARHelper");
        SetDebugMode(_isDebugMode);

        // Validate required components
        if (_passthroughCameraAccess == null)
        {
            Debug.LogError("PassthroughCameraAccess reference is missing.");
            enabled = false;
            yield break;
        }

        // Create camera anchor dynamically
        CreateCameraAnchor();

        // Initialize camera
        yield return InitializeCamera();

        // Initialize the image tracking with proper camera parameters
        InitializeImageTracking();
    }

    /// <summary>
    /// Creates a camera anchor GameObject dynamically at runtime.
    /// </summary>
    private void CreateCameraAnchor()
    {
        GameObject anchorObject = new GameObject("CameraAnchor");
        _cameraAnchor = anchorObject.transform;
    }

    /// <summary>
    /// Initializes the camera with appropriate resolution and waits until ready.
    /// </summary>
    private IEnumerator InitializeCamera()
    {
        while (!_passthroughCameraAccess.IsPlaying)
        {
            yield return null;
        }
        yield return null; // Wait one frame to ensure camera is fully initialized
    }

    /// <summary>
    /// Initializes the image tracking system with accurate camera parameters.
    /// </summary>
    private void InitializeImageTracking()
    {
        // Get camera intrinsic parameters
        var intrinsics = _passthroughCameraAccess.Intrinsics;
        var cx = intrinsics.PrincipalPoint.x;
        var cy = intrinsics.PrincipalPoint.y;
        var fx = intrinsics.FocalLength.x;
        var fy = intrinsics.FocalLength.y;
        var width = intrinsics.SensorResolution.x;
        var height = intrinsics.SensorResolution.y;

        // Scale parameters to match current camera resolution
        var currentResolution = _passthroughCameraAccess.CurrentResolution;
        Debug.Log($"Camera Intrinsics - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, width: {width}, height: {height}");
        Debug.Log($"Current Camera Resolution - width: {currentResolution.x}, height: {currentResolution.y}");

        if (currentResolution.x != width || currentResolution.y != height)
        {
            float scaleX = (float)currentResolution.x / width;
            float scaleY = (float)currentResolution.y / height;
            fx *= scaleX;
            fy *= scaleY;
            cx *= scaleX;
            cy *= scaleY;
            width = currentResolution.x;
            height = currentResolution.y;
        }

        // Configure the ImageTracking component
        _imageTracking.Initialize(width, height, cx, cy, fx, fy);

        _modeText.text = _imageTracking.GetTrackingMode() == ImageTrackingMode.Dynamic ? "Dynamic Mode" : "Static Mode";
    }

    void Update()
    {
        // Skip if camera isn't ready
        if (_passthroughCameraAccess == null || !_passthroughCameraAccess.IsPlaying || !_imageTracking.IsReady)
            return;

        // Toggle between camera view and AR visualization on button press
        HandleDebugVisualizationToggle();

        // Toggle between tracking modes on button press
        HandleModeToggle();

        // Update camera positions
        UpdateCameraPoses();

        // Process the current frame for image tracking
        ProcessImageTracking();
    }

    /// <summary>
    /// Handles button input to toggle between camera view and AR visualization.
    /// </summary>
    private void HandleDebugVisualizationToggle()
    {
        if (OVRInput.GetDown(OVRInput.Button.One))
        {
            SetDebugMode(!_isDebugMode);
        }
    }

    private void SetDebugMode(bool enabled)
    {
        _isDebugMode = enabled;
        foreach(var gameObject in _arHelpers)
        {
            gameObject.SetActive(enabled);
        }
    }

    private void HandleModeToggle()
    {
        if (OVRInput.GetDown(OVRInput.Button.Two))
        {
            // Toggle between modes
            ImageTrackingMode newMode = _imageTracking.GetTrackingMode() == ImageTrackingMode.Dynamic ? 
                ImageTrackingMode.Static : 
                ImageTrackingMode.Dynamic;
                
            _imageTracking.SetTrackingMode(newMode, true);
            _modeText.text = newMode == ImageTrackingMode.Dynamic ? "Dynamic Mode" : "Static Mode";
        }
    }

    /// <summary>
    /// Processes the current camera frame for image tracking.
    /// </summary>
    private void ProcessImageTracking()
    {
        // If we are in static mode and all the images have been stabilized, we can skip the detection step
        if (_imageTracking.GetTrackingMode() == ImageTrackingMode.Static && _imageTracking.AreAllImagesStabilized())
        {
            return;
        }

        // Detect all markers in the current frame
        _imageTracking.DetectImages(_passthroughCameraAccess.GetTexture(), _resultTexture);
        
        // Update poses for all detected markers
        _imageTracking.EstimateImagePoses(_cameraAnchor);
    }

    /// <summary>
    /// Sets the visibility of all AR GameObjects
    /// </summary>
    private void SetARObjectsVisibility(bool isVisible)
    {
        foreach (var marker in _imageTracking.TrackedImages)
        {
            if (marker.virtualObject != null)
            {
                var rendererList = marker.virtualObject.GetComponentsInChildren<Renderer>(true);
                foreach (var renderer in rendererList)
                {
                    renderer.enabled = isVisible && marker.isDetected;
                }
            }
        }
    }

    /// <summary>
    /// Updates the camera anchor position and rotation based on the camera pose.
    /// </summary>
    private void UpdateCameraPoses()
    {
        var cameraPose = _passthroughCameraAccess.GetCameraPose();
        _cameraAnchor.position = cameraPose.position;
        _cameraAnchor.rotation = cameraPose.rotation;
    }
    
    /// <summary>
    /// Add a new marker at runtime
    /// </summary>
    public void AddTrackedImage(ARTrackedImage marker)
    {
        if (_imageTracking != null && _imageTracking.IsReady)
        {
            _imageTracking.AddImage(marker);
        }
    }
}