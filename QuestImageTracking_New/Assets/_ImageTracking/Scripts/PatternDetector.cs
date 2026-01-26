using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Features2dModule;
using OpenCVForUnity.ImgprocModule;
using System.Collections.Generic;
using UnityEngine;

namespace OpenCVMarkerLessAR
{
    /// <summary>
    /// Pattern detector.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter3_MarkerlessAR using "OpenCV for Unity".
    /// </summary>
    public class PatternDetector
    {
        /// <summary>
        /// The enable ratio test.
        /// </summary>
        public bool enableRatioTest;

        /// <summary>
        /// The enable homography refinement.
        /// </summary>
        public bool enableHomographyRefinement;

        /// <summary>
        /// The homography reprojection threshold.
        /// </summary>
        public float homographyReprojectionThreshold;

        /// <summary>
        /// The m_query keypoints.
        /// </summary>
        MatOfKeyPoint m_queryKeypoints;

        /// <summary>
        /// The m_query descriptors.
        /// </summary>
        Mat m_queryDescriptors;

        /// <summary>
        /// The m_matches.
        /// </summary>
        MatOfDMatch m_matches;

        /// <summary>
        /// The m_knn matches.
        /// </summary>
        List<MatOfDMatch> m_knnMatches;

        /// <summary>
        /// The m_gray image.
        /// </summary>
        Mat m_grayImg;

        /// <summary>
        /// The m_warped image.
        /// </summary>
        Mat m_warpedImg;

        /// <summary>
        /// The m_rough homography.
        /// </summary>
        Mat m_roughHomography;

        /// <summary>
        /// The m_refined homography.
        /// </summary>
        Mat m_refinedHomography;

        /// <summary>
        /// The m_pattern.
        /// </summary>
        Pattern m_pattern;

        /// <summary>
        /// The m_detector.
        /// </summary>
        ORB m_detector;

        /// <summary>
        /// The m_extractor.
        /// </summary>
        ORB m_extractor;

        /// <summary>
        /// The m_matcher.
        /// </summary>
        DescriptorMatcher m_matcher;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternDetector"/> class.
        /// </summary>
        /// <param name="detector">Detector.</param>
        /// <param name="extractor">Extractor.</param>
        /// <param name="matcher">Matcher.</param>
        /// <param name="ratioTest">If set to <c>true</c> ratio test.</param>
        public PatternDetector(ORB detector, ORB extractor, DescriptorMatcher matcher, bool ratioTest = false)
        {
            if (detector == null)
            {
                detector = ORB.create();
                detector.setMaxFeatures(1000);
            }
            if (extractor == null)
            {
                extractor = ORB.create();
                extractor.setMaxFeatures(1000);
            }
            if (matcher == null)
            {
                matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            }

            m_detector = detector;
            m_extractor = extractor;
            m_matcher = matcher;

            enableRatioTest = ratioTest;
            enableHomographyRefinement = true;
            homographyReprojectionThreshold = 3;

            m_queryKeypoints = new MatOfKeyPoint();
            m_queryDescriptors = new Mat();
            m_matches = new MatOfDMatch();
            m_knnMatches = new List<MatOfDMatch>();
            m_grayImg = new Mat();
            m_warpedImg = new Mat();
            m_roughHomography = new Mat();
            m_refinedHomography = new Mat();
        }

        /// <summary>
        /// Train the specified pattern.
        /// </summary>
        /// <param name="pattern">Pattern.</param>
        public void train(Pattern pattern)
        {
            // Store the pattern object
            m_pattern = pattern;

            // API of cv::DescriptorMatcher is somewhat tricky
            // First we clear old train data:
            m_matcher.clear();

            // Then we add vector of descriptors (each descriptors matrix describe one image). 
            // This allows us to perform search across multiple images:
            List<Mat> descriptors = new List<Mat>(1);
            descriptors.Add(pattern.descriptors.clone());
            m_matcher.add(descriptors);

            // After adding train data perform actual train:
            m_matcher.train();
        }

        /// <summary>
        /// Builds the pattern from image.
        /// </summary>
        /// <param name="image">Image.</param>
        /// <param name="pattern">Pattern.</param>
        public bool buildPatternFromImage(Mat image, Pattern pattern)
        {
            //int numImages = 4;
            //float step = Mathf.Sqrt (2.0f);

            // Store original image in pattern structure
            pattern.size = new Size(image.cols(), image.rows());
            pattern.frame = image.clone();
            getGray(image, pattern.grayImg);

            // Build 2d and 3d contours (3d contour lie in XY plane since it's planar)
            List<Point> points2dList = new List<Point>(4);
            List<Point3> points3dList = new List<Point3>(4);

            // Image dimensions
            float w = image.cols();
            float h = image.rows();

            // Normalized dimensions:
            //float maxSize = Mathf.Max (w, h);
            //float unitW = w / maxSize;
            //float unitH = h / maxSize;

            points2dList.Add(new Point(0, 0));
            points2dList.Add(new Point(w, 0));
            points2dList.Add(new Point(w, h));
            points2dList.Add(new Point(0, h));

            pattern.points2d.fromList(points2dList);


            //points3dList.Add (new Point3 (-unitW, -unitH, 0));
            //points3dList.Add (new Point3 (unitW, -unitH, 0));
            //points3dList.Add (new Point3 (unitW, unitH, 0));
            //points3dList.Add (new Point3 (-unitW, unitH, 0));

            points3dList.Add(new Point3(-0.5f, -0.5f, 0));
            points3dList.Add(new Point3(+0.5f, -0.5f, 0));
            points3dList.Add(new Point3(+0.5f, +0.5f, 0));
            points3dList.Add(new Point3(-0.5f, +0.5f, 0));

            pattern.points3d.fromList(points3dList);


            return extractFeatures(pattern.grayImg, pattern.keypoints, pattern.descriptors);
        }

        /// <summary>
        /// Finds the pattern.
        /// </summary>
        /// <returns><c>true</c>, if pattern was found, <c>false</c> otherwise.</returns>
        /// <param name="image">Image.</param>
        /// <param name="info">Info.</param>
        public bool findPattern(Mat image, PatternTrackingInfo info)
        {
            // Convert input image to gray
            getGray(image, m_grayImg);

            // Extract feature points from input gray image
            bool result = extractFeatures(m_grayImg, m_queryKeypoints, m_queryDescriptors);
            if (!result)
                return false;

            // Get matches with current pattern
            getMatches(m_queryDescriptors, m_matches);

            //(GameObject.Find ("DebugHelpers").GetComponent<DebugHelpers> ()).showMat (DebugHelpers.getMatchesImage (m_grayImg, m_pattern.grayImg, m_queryKeypoints, m_pattern.keypoints, m_matches, 100));


            // Find homography transformation and detect good matches
            bool homographyFound = refineMatchesWithHomography(
                                       m_queryKeypoints,
                                       m_pattern.keypoints,
                                       homographyReprojectionThreshold,
                                       m_matches,
                                       m_roughHomography);

            if (homographyFound)
            {

                //(GameObject.Find ("DebugHelpers").GetComponent<DebugHelpers> ()).showMat (DebugHelpers.getMatchesImage (m_grayImg, m_pattern.grayImg, m_queryKeypoints, m_pattern.keypoints, m_matches, 100));

                // If homography refinement enabled improve found transformation
                if (enableHomographyRefinement)
                {
                    // Warp image using found homography
                    Imgproc.warpPerspective(m_grayImg, m_warpedImg, m_roughHomography, m_pattern.size, Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_CUBIC);


                    //(GameObject.Find ("DebugHelpers").GetComponent<DebugHelpers> ()).showMat(m_warpedImg);

                    // Get refined matches:
                    using (MatOfKeyPoint warpedKeypoints = new MatOfKeyPoint())
                    using (MatOfDMatch refinedMatches = new MatOfDMatch())
                    {
                        // Detect features on warped image
                        result = extractFeatures(m_warpedImg, warpedKeypoints, m_queryDescriptors);
                        if (!result)
                            return false;

                        // Match with pattern
                        getMatches(m_queryDescriptors, refinedMatches);

                        // Estimate new refinement homography
                        homographyFound = refineMatchesWithHomography(
                            warpedKeypoints,
                            m_pattern.keypoints,
                            homographyReprojectionThreshold,
                            refinedMatches,
                            m_refinedHomography);
                    }

                    //(GameObject.Find ("DebugHelpers").GetComponent<DebugHelpers> ()).showMat(DebugHelpers.getMatchesImage(m_warpedImg, m_pattern.grayImg, warpedKeypoints, m_pattern.keypoints, refinedMatches, 100));

                    // Get a result homography as result of matrix product of refined and rough homographies:
                    //                              info.homography = m_roughHomography * m_refinedHomography;
                    Core.gemm(m_roughHomography, m_refinedHomography, 1, new Mat(), 0, info.homography);

                    //Debug.Log ("info.homography " + info.homography.ToString ());

                    // Transform contour with rough homography

                    //Core.perspectiveTransform (m_pattern.points2d, info.points2d, m_roughHomography);
                    //info.draw2dContour (image, new Scalar (200, 0, 0, 255));


                    // Transform contour with precise homography

                    Core.perspectiveTransform(m_pattern.points2d, info.points2d, info.homography);

                    //info.draw2dContour (image, new Scalar (0, 200, 0, 255));

                }
                else
                {
                    info.homography = m_roughHomography;

                    //Debug.Log ("m_roughHomography " + m_roughHomography.ToString ());
                    //Debug.Log ("info.homography " + info.homography.ToString ());

                    // Transform contour with rough homography
                    Core.perspectiveTransform(m_pattern.points2d, info.points2d, m_roughHomography);

                    //info.draw2dContour (image, new Scalar (0, 200, 0, 255));
                }
            }

            //(GameObject.Find ("DebugHelpers").GetComponent<DebugHelpers> ()).showMat (DebugHelpers.getMatchesImage (m_grayImg, m_pattern.grayImg, m_queryKeypoints, m_pattern.keypoints, m_matches, 100));
            //Debug.Log ("Features:" + m_queryKeypoints.ToString () + " Matches: " + m_matches.ToString ());

            return homographyFound;
        }

        /// <summary>
        /// Gets the gray.
        /// </summary>
        /// <param name="image">Image.</param>
        /// <param name="gray">Gray.</param>
        static void getGray(Mat image, Mat gray)
        {
            if (image.channels() == 3)
                Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY);
            else if (image.channels() == 4)
                Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGBA2GRAY);
            else if (image.channels() == 1)
                image.copyTo(gray);
        }

        /// <summary>
        /// Extracts the features.
        /// </summary>
        /// <returns><c>true</c>, if features was extracted, <c>false</c> otherwise.</returns>
        /// <param name="image">Image.</param>
        /// <param name="keypoints">Keypoints.</param>
        /// <param name="descriptors">Descriptors.</param>
        bool extractFeatures(Mat image, MatOfKeyPoint keypoints, Mat descriptors)
        {
            if (image.total() == 0)
            {
                return false;
            }
            if (image.channels() != 1)
            {
                return false;
            }

            m_detector.detect(image, keypoints);
            if (keypoints.total() == 0)
                return false;

            m_extractor.compute(image, keypoints, descriptors);
            if (keypoints.total() == 0)
                return false;

            //Debug.Log ("extractFeatures true");

            //Mat tmpImage = new Mat();
            //
            //Features2d.drawKeypoints(image, keypoints, tmpImage);
            //
            //DebugHelpers.showMat(tmpImage);

            return true;
        }

        /// <summary>
        /// Gets the matches.
        /// </summary>
        /// <param name="queryDescriptors">Query descriptors.</param>
        /// <param name="matches">Matches.</param>
        void getMatches(Mat queryDescriptors, MatOfDMatch matches)
        {
            List<DMatch> matchesList = new List<DMatch>();
            //matches.clear();

            if (enableRatioTest)
            {
                // To avoid NaN's when best match has zero distance we will use inversed ratio. 
                float minRatio = 1.0f / 1.5f;

                // KNN match will return 2 nearest matches for each query descriptor
                m_matcher.knnMatch(queryDescriptors, m_knnMatches, 2);

                for (int i = 0; i < m_knnMatches.Count; i++)
                {
                    List<DMatch> m_knnMatchesList = m_knnMatches[i].toList();

                    DMatch bestMatch = m_knnMatchesList[0];
                    DMatch betterMatch = m_knnMatchesList[1];

                    float distanceRatio = bestMatch.distance / betterMatch.distance;

                    // Pass only matches where distance ratio between 
                    // nearest matches is greater than 1.5 (distinct criteria)
                    if (distanceRatio < minRatio)
                    {
                        matchesList.Add(bestMatch);
                    }
                }

                matches.fromList(matchesList);
            }
            else
            {
                matches.fromList(matchesList);

                // Perform regular match
                m_matcher.match(queryDescriptors, matches);
            }

            //Debug.Log ("getMatches " + matches.ToString ());
        }

        /// <summary>
        /// Refines the matches with homography.
        /// </summary>
        /// <returns><c>true</c>, if matches with homography was refined, <c>false</c> otherwise.</returns>
        /// <param name="queryKeypoints">Query keypoints.</param>
        /// <param name="trainKeypoints">Train keypoints.</param>
        /// <param name="reprojectionThreshold">Reprojection threshold.</param>
        /// <param name="matches">Matches.</param>
        /// <param name="homography">Homography.</param>
        static bool refineMatchesWithHomography
        (
            MatOfKeyPoint queryKeypoints,
            MatOfKeyPoint trainKeypoints,
            float reprojectionThreshold,
            MatOfDMatch matches,
            Mat homography
        )
        {
            //Debug.Log ("matches " + matches.ToString ());

            int minNumberMatchesAllowed = 8;

            List<KeyPoint> queryKeypointsList = queryKeypoints.toList();
            List<KeyPoint> trainKeypointsList = trainKeypoints.toList();
            List<DMatch> matchesList = matches.toList();

            if (matchesList.Count < minNumberMatchesAllowed)
                return false;

            // Prepare data for cv::findHomography
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

                //Debug.Log ("srcPoints " + srcPoints.ToString ());
                //Debug.Log ("dstPoints " + dstPoints.ToString ());

                Calib3d.findHomography(srcPoints,
                    dstPoints,
                    Calib3d.FM_RANSAC,
                    reprojectionThreshold,
                    inliersMask, 2000, 0.955).copyTo(homography);

                if (homography.rows() != 3 || homography.cols() != 3)
                    return false;

                //Debug.Log ("homography " + homography.ToString ());
                //Debug.Log ("inliersMask " + inliersMask.dump ());

                List<byte> inliersMaskList = inliersMask.toList();

                List<DMatch> inliers = new List<DMatch>();
                for (int i = 0; i < inliersMaskList.Count; i++)
                {
                    if (inliersMaskList[i] == 1)
                        inliers.Add(matchesList[i]);
                }

                matches.fromList(inliers);
                //Debug.Log ("matches " + matches.ToString ());
            }

            return matchesList.Count > minNumberMatchesAllowed;
        }
    }
}