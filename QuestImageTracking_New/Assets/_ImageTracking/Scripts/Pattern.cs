using OpenCVForUnity.CoreModule;

namespace OpenCVMarkerLessAR
{
    /// <summary>
    /// Pattern.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter3_MarkerlessAR using "OpenCV for Unity".
    /// </summary>
    public class Pattern
    {
        /// <summary>
        /// The size.
        /// </summary>
        public Size size;

        /// <summary>
        /// The frame.
        /// </summary>
        public Mat frame;

        /// <summary>
        /// The gray image.
        /// </summary>
        public Mat grayImg;

        /// <summary>
        /// The keypoints.
        /// </summary>
        public MatOfKeyPoint keypoints;

        /// <summary>
        /// The descriptors.
        /// </summary>
        public Mat descriptors;

        /// <summary>
        /// The points2d.
        /// </summary>
        public MatOfPoint2f points2d;

        /// <summary>
        /// The points3d.
        /// </summary>
        public MatOfPoint3f points3d;

        /// <summary>
        /// Initializes a new instance of the <see cref="Pattern"/> class.
        /// </summary>
        public Pattern()
        {
            size = new Size();
            frame = new Mat();
            grayImg = new Mat();
            keypoints = new MatOfKeyPoint();
            descriptors = new Mat();
            points2d = new MatOfPoint2f();
            points3d = new MatOfPoint3f();
        }
    }
}