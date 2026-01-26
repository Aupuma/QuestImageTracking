using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Features2dModule;
using OpenCVForUnity.ImgprocModule;
using System.Collections.Generic;
using UnityEngine;

namespace OpenCVMarkerLessAR
{
    /// <summary>
    /// Detector that can track multiple patterns simultaneously without retraining.
    /// </summary>
    public class MultiplePatternDetector
    {
        /// <summary>
        /// Configuration options for pattern detection
        /// </summary>
        public bool enableRatioTest = true;
        public bool enableHomographyRefinement = true;
        public float homographyReprojectionThreshold = 3.0f;

        // Feature detection and extraction
        private ORB m_detector;
        private ORB m_extractor;
        
        // Reusable buffers for processing the input image once
        private Mat m_grayImg;
        private MatOfKeyPoint m_queryKeypoints;
        private Mat m_queryDescriptors;
        
        // Individual matcher and processing elements for each pattern
        private Dictionary<string, PatternMatchingResources> m_patternResources = new Dictionary<string, PatternMatchingResources>();
        
        /// <summary>
        /// Class that holds matching resources for a single pattern
        /// </summary>
        private class PatternMatchingResources
        {
            public DescriptorMatcher matcher;
            public Pattern pattern;
            public MatOfDMatch matches;
            public List<MatOfDMatch> knnMatches;
            public Mat warpedImg;
            public Mat roughHomography;
            public Mat refinedHomography;
            
            public PatternMatchingResources(Pattern pattern)
            {
                this.pattern = pattern;
                matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
                matches = new MatOfDMatch();
                knnMatches = new List<MatOfDMatch>();
                warpedImg = new Mat();
                roughHomography = new Mat();
                refinedHomography = new Mat();
                
                // Train the matcher with this pattern's descriptors
                List<Mat> descriptors = new List<Mat>(1);
                descriptors.Add(pattern.descriptors.clone());
                matcher.add(descriptors);
                matcher.train();
            }
            
            public void Release()
            {
                if (matches != null) matches.release();
                foreach (var match in knnMatches)
                {
                    match.release();
                }
                if (warpedImg != null) warpedImg.release();
                if (roughHomography != null) roughHomography.release();
                if (refinedHomography != null) refinedHomography.release();
                matcher.clear();
            }
        }

        /// <summary>
        /// Initializes a new instance of the MultiplePatternDetector
        /// </summary>
        public MultiplePatternDetector()
        {
            // Create feature detector
            m_detector = ORB.create();
            m_detector.setMaxFeatures(1000);
            
            // Create feature extractor
            m_extractor = ORB.create();
            m_extractor.setMaxFeatures(1000);
            
            // Initialize processing buffers
            m_grayImg = new Mat();
            m_queryKeypoints = new MatOfKeyPoint();
            m_queryDescriptors = new Mat();
        }

        /// <summary>
        /// Builds a pattern from an image and registers it with the detector
        /// </summary>
        /// <param name="image">The image containing the pattern</param>
        /// <param name="pattern">The pattern object to populate</param>
        /// <param name="patternId">A unique identifier for this pattern</param>
        /// <returns>True if pattern was successfully built and registered</returns>
        public bool BuildAndRegisterPattern(Mat image, Pattern pattern, string patternId)
        {
            // First build the pattern
            bool success = BuildPatternFromImage(image, pattern);
            if (!success)
            {
                Debug.LogError($"Failed to build pattern {patternId}");
                return false;
            }
            
            // Register the pattern with its own dedicated resources
            RegisterPattern(pattern, patternId);
            return true;
        }
        
        /// <summary>
        /// Registers an already-built pattern with the detector
        /// </summary>
        /// <param name="pattern">The pattern to register</param>
        /// <param name="patternId">A unique identifier for this pattern</param>
        public void RegisterPattern(Pattern pattern, string patternId)
        {
            // Remove any existing resources for this pattern ID
            if (m_patternResources.ContainsKey(patternId))
            {
                m_patternResources[patternId].Release();
                m_patternResources.Remove(patternId);
            }
            
            // Create new resources for this pattern
            m_patternResources[patternId] = new PatternMatchingResources(pattern);
        }
        
        /// <summary>
        /// Unregisters a pattern from the detector
        /// </summary>
        /// <param name="patternId">The unique identifier of the pattern to remove</param>
        /// <returns>True if the pattern was found and removed</returns>
        public bool UnregisterPattern(string patternId)
        {
            if (m_patternResources.ContainsKey(patternId))
            {
                m_patternResources[patternId].Release();
                m_patternResources.Remove(patternId);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Builds a pattern from an image
        /// </summary>
        /// <param name="image">Source image</param>
        /// <param name="pattern">Pattern output</param>
        /// <returns>True if pattern was built successfully</returns>
        public bool BuildPatternFromImage(Mat image, Pattern pattern)
        {
            // Store original image in pattern structure
            pattern.size = new Size(image.cols(), image.rows());
            pattern.frame = image.clone();
            GetGray(image, pattern.grayImg);

            // Build 2d and 3d contours (3d contour lie in XY plane since it's planar)
            List<Point> points2dList = new List<Point>(4);
            List<Point3> points3dList = new List<Point3>(4);

            // Image dimensions
            float w = image.cols();
            float h = image.rows();

            points2dList.Add(new Point(0, 0));
            points2dList.Add(new Point(w, 0));
            points2dList.Add(new Point(w, h));
            points2dList.Add(new Point(0, h));

            pattern.points2d.fromList(points2dList);

            // Using normalized 3D points (-0.5 to 0.5)
            points3dList.Add(new Point3(-0.5f, -0.5f, 0));
            points3dList.Add(new Point3(+0.5f, -0.5f, 0));
            points3dList.Add(new Point3(+0.5f, +0.5f, 0));
            points3dList.Add(new Point3(-0.5f, +0.5f, 0));

            pattern.points3d.fromList(points3dList);

            // Extract features from the pattern image
            return ExtractFeatures(pattern.grayImg, pattern.keypoints, pattern.descriptors);
        }

        /// <summary>
        /// Process an input image once and try to detect all registered patterns
        /// </summary>
        /// <param name="image">Input image to search for patterns</param>
        /// <param name="detectedPatternIds">Output list of detected pattern IDs</param>
        /// <param name="trackingInfos">Output dictionary of tracking info for each detected pattern</param>
        /// <returns>True if any patterns were detected</returns>
        public bool DetectPatterns(Mat image, List<string> detectedPatternIds, Dictionary<string, PatternTrackingInfo> trackingInfos)
        {
            // Clear output collections
            detectedPatternIds.Clear();
            
            // Early exit if no patterns are registered
            if (m_patternResources.Count == 0)
                return false;
                
            // Process input image once
            GetGray(image, m_grayImg);
            bool featuresExtracted = ExtractFeatures(m_grayImg, m_queryKeypoints, m_queryDescriptors);
            if (!featuresExtracted)
                return false;
                
            // Try to match with each registered pattern
            foreach (var entry in m_patternResources)
            {
                string patternId = entry.Key;
                PatternMatchingResources resources = entry.Value;
                
                // Get matches between query image and this pattern
                GetMatches(resources.matcher, m_queryDescriptors, resources.matches, resources.knnMatches);
                
                // Find homography transformation to detect good matches
                bool homographyFound = RefineMatchesWithHomography(
                    m_queryKeypoints,
                    resources.pattern.keypoints,
                    homographyReprojectionThreshold,
                    resources.matches,
                    resources.roughHomography);
                    
                if (homographyFound)
                {
                    // Get or create tracking info for this pattern
                    PatternTrackingInfo trackingInfo;
                    if (!trackingInfos.TryGetValue(patternId, out trackingInfo))
                    {
                        trackingInfo = new PatternTrackingInfo();
                        trackingInfos[patternId] = trackingInfo;
                    }
                    
                    // If refinement is enabled, improve the transformation
                    if (enableHomographyRefinement)
                    {
                        // Warp image using found homography
                        Imgproc.warpPerspective(
                            m_grayImg, 
                            resources.warpedImg, 
                            resources.roughHomography, 
                            resources.pattern.size, 
                            Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_CUBIC);
                        
                        // Use a local scope to release temporary resources
                        using (MatOfKeyPoint warpedKeypoints = new MatOfKeyPoint())
                        using (Mat warpedDescriptors = new Mat())
                        using (MatOfDMatch refinedMatches = new MatOfDMatch())
                        {
                            // Detect features on warped image
                            bool result = ExtractFeatures(resources.warpedImg, warpedKeypoints, warpedDescriptors);
                            if (result)
                            {
                                // Match warped image with pattern
                                GetMatches(resources.matcher, warpedDescriptors, refinedMatches, resources.knnMatches);
                                
                                // Refine homography
                                homographyFound = RefineMatchesWithHomography(
                                    warpedKeypoints,
                                    resources.pattern.keypoints,
                                    homographyReprojectionThreshold,
                                    refinedMatches,
                                    resources.refinedHomography);
                                    
                                if (homographyFound)
                                {
                                    // Combined homography (rough * refined)
                                    Core.gemm(
                                        resources.roughHomography, 
                                        resources.refinedHomography, 
                                        1, 
                                        new Mat(), 
                                        0, 
                                        trackingInfo.homography);
                                        
                                    // Transform contour with precise homography
                                    Core.perspectiveTransform(
                                        resources.pattern.points2d, 
                                        trackingInfo.points2d, 
                                        trackingInfo.homography);
                                }
                            }
                        }
                    }
                    else
                    {
                        // Use the rough homography directly
                        resources.roughHomography.copyTo(trackingInfo.homography);
                        
                        // Transform contour with rough homography
                        Core.perspectiveTransform(
                            resources.pattern.points2d, 
                            trackingInfo.points2d, 
                            resources.roughHomography);
                    }
                    
                    // If we made it here, we've detected this pattern
                    detectedPatternIds.Add(patternId);
                }
            }
            
            return detectedPatternIds.Count > 0;
        }
        
        /// <summary>
        /// Extracts features from a grayscale image
        /// </summary>
        private bool ExtractFeatures(Mat image, MatOfKeyPoint keypoints, Mat descriptors)
        {
            if (image.total() == 0 || image.channels() != 1)
                return false;
                
            m_detector.detect(image, keypoints);
            if (keypoints.total() == 0)
                return false;
                
            m_extractor.compute(image, keypoints, descriptors);
            if (keypoints.total() == 0)
                return false;
                
            return true;
        }
        
        /// <summary>
        /// Converts an image to grayscale if needed
        /// </summary>
        private static void GetGray(Mat image, Mat gray)
        {
            if (image.channels() == 3)
                Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY);
            else if (image.channels() == 4)
                Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGBA2GRAY);
            else if (image.channels() == 1)
                image.copyTo(gray);
        }
        
        /// <summary>
        /// Matches descriptors between query and trained pattern
        /// </summary>
        private void GetMatches(DescriptorMatcher matcher, Mat queryDescriptors, MatOfDMatch matches, List<MatOfDMatch> knnMatches)
        {
            // Clear existing matches from previous calls
            matches.release();
            
            // Clear existing knnMatches from previous calls
            foreach (var match in knnMatches)
            {
                match.release();
            }
            knnMatches.Clear();
            
            List<DMatch> matchesList = new List<DMatch>();
            
            if (enableRatioTest)
            {
                // To avoid NaN's when best match has zero distance we will use inversed ratio
                float minRatio = 1.0f / 1.5f;
                
                // KNN match will return 2 nearest matches for each query descriptor
                matcher.knnMatch(queryDescriptors, knnMatches, 2);
                
                for (int i = 0; i < knnMatches.Count; i++)
                {
                    List<DMatch> knnMatchesList = knnMatches[i].toList();
                    
                    DMatch bestMatch = knnMatchesList[0];
                    DMatch betterMatch = knnMatchesList[1];
                    
                    float distanceRatio = bestMatch.distance / betterMatch.distance;
                    
                    // Pass only matches where distance ratio between nearest matches is greater than 1.5
                    if (distanceRatio < minRatio)
                    {
                        matchesList.Add(bestMatch);
                    }
                }
                
                matches.fromList(matchesList);
            }
            else
            {
                // Perform regular match
                matcher.match(queryDescriptors, matches);
            }
        }
        
        /// <summary>
        /// Refines matches using homography
        /// </summary>
        private static bool RefineMatchesWithHomography(
            MatOfKeyPoint queryKeypoints,
            MatOfKeyPoint trainKeypoints,
            float reprojectionThreshold,
            MatOfDMatch matches,
            Mat homography)
        {
            int minNumberMatchesAllowed = 8;
            
            List<KeyPoint> queryKeypointsList = queryKeypoints.toList();
            List<KeyPoint> trainKeypointsList = trainKeypoints.toList();
            List<DMatch> matchesList = matches.toList();
            
            if (matchesList.Count < minNumberMatchesAllowed)
                return false;
                
            // Prepare data for findHomography
            List<Point> srcPointsList = new List<Point>(matchesList.Count);
            List<Point> dstPointsList = new List<Point>(matchesList.Count);
            
            for (int i = 0; i < matchesList.Count; i++)
            {
                srcPointsList.Add(trainKeypointsList[matchesList[i].trainIdx].pt);
                dstPointsList.Add(queryKeypointsList[matchesList[i].queryIdx].pt);
            }
            
            // Find homography matrix and get inliers mask
            using (MatOfPoint2f srcPoints = new MatOfPoint2f())
            using (MatOfPoint2f dstPoints = new MatOfPoint2f())
            using (MatOfByte inliersMask = new MatOfByte(new byte[srcPointsList.Count]))
            {
                srcPoints.fromList(srcPointsList);
                dstPoints.fromList(dstPointsList);
                
                Calib3d.findHomography(
                    srcPoints,
                    dstPoints,
                    Calib3d.FM_RANSAC,
                    reprojectionThreshold,
                    inliersMask, 
                    2000, 
                    0.955).copyTo(homography);
                    
                if (homography.rows() != 3 || homography.cols() != 3)
                    return false;
                    
                // Keep only inlier matches
                List<byte> inliersMaskList = inliersMask.toList();
                List<DMatch> inliers = new List<DMatch>();
                
                for (int i = 0; i < inliersMaskList.Count; i++)
                {
                    if (inliersMaskList[i] == 1)
                        inliers.Add(matchesList[i]);
                }
                
                matches.fromList(inliers);
                
                // Check if we have enough inliers, not just original matches
                return inliers.Count >= minNumberMatchesAllowed;
            }
        }
        
        /// <summary>
        /// Releases resources used by the detector
        /// </summary>
        public void Release()
        {
            // Release all pattern resources
            foreach (var resources in m_patternResources.Values)
            {
                resources.Release();
            }
            m_patternResources.Clear();
            
            // Release common resources
            if (m_grayImg != null) m_grayImg.release();
            if (m_queryKeypoints != null) m_queryKeypoints.release();
            if (m_queryDescriptors != null) m_queryDescriptors.release();
        }
    }
}