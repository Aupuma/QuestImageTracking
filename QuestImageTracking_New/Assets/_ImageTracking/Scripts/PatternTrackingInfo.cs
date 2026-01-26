using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using System.Collections.Generic;
using UnityEngine;

namespace OpenCVMarkerLessAR
{
    /// <summary>
    /// Pattern tracking info.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter3_MarkerlessAR using "OpenCV for Unity".
    /// </summary>
    public class PatternTrackingInfo
    {
        /// <summary>
        /// The homography.
        /// </summary>
        public Mat homography;

        /// <summary>
        /// The points2d.
        /// </summary>
        public MatOfPoint2f points2d;

        /// <summary>
        /// The pose3d.
        /// </summary>
        public Matrix4x4 pose3d;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternTrackingInfo"/> class.
        /// </summary>
        public PatternTrackingInfo()
        {
            homography = new Mat();
            points2d = new MatOfPoint2f();
            pose3d = new Matrix4x4();
        }

        /// <summary>
        /// Computes the pose.
        /// </summary>
        /// <param name="pattern">Pattern.</param>
        /// <param name="camMatrix">Cam matrix.</param>
        /// <param name="distCoeff">Dist coeff.</param>
        public void computePose(Pattern pattern, Mat camMatrix, MatOfDouble distCoeff)
        {
            Mat Rvec = new Mat();
            Mat Tvec = new Mat();
            Mat raux = new Mat();
            Mat taux = new Mat();

            Calib3d.solvePnP(pattern.points3d, points2d, camMatrix, distCoeff, raux, taux);
            raux.convertTo(Rvec, CvType.CV_32F);
            taux.convertTo(Tvec, CvType.CV_32F);

            Mat rotMat = new Mat(3, 3, CvType.CV_64FC1);
            Calib3d.Rodrigues(Rvec, rotMat);

            pose3d.SetRow(0, new Vector4((float)rotMat.get(0, 0)[0], (float)rotMat.get(0, 1)[0], (float)rotMat.get(0, 2)[0], (float)Tvec.get(0, 0)[0]));
            pose3d.SetRow(1, new Vector4((float)rotMat.get(1, 0)[0], (float)rotMat.get(1, 1)[0], (float)rotMat.get(1, 2)[0], (float)Tvec.get(1, 0)[0]));
            pose3d.SetRow(2, new Vector4((float)rotMat.get(2, 0)[0], (float)rotMat.get(2, 1)[0], (float)rotMat.get(2, 2)[0], (float)Tvec.get(2, 0)[0]));
            pose3d.SetRow(3, new Vector4(0, 0, 0, 1));

            Rvec.Dispose();
            Tvec.Dispose();
            raux.Dispose();
            taux.Dispose();
            rotMat.Dispose();
        }

        /// <summary>
        /// Draw2ds the contour.
        /// </summary>
        /// <param name="image">Image.</param>
        /// <param name="color">Color.</param>
        public void draw2dContour(Mat image, Scalar color)
        {
            List<Point> points2dList = points2d.toList();

            for (int i = 0; i < points2dList.Count; i++)
            {
                Imgproc.line(image, points2dList[i], points2dList[(i + 1) % points2dList.Count], color, 2, Imgproc.LINE_AA, 0);
            }
        }
    }
}