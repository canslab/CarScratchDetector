#pragma once
#include <vector>
#include <unordered_map>
#include <opencv2\opencv.hpp>
#include "..\DBSCAN.h"

// 디버깅용 전처리기
#ifndef NDEBUG
#define FOR_DEBUG			true
#endif

// Class forward declartion
class MeanShiftCluster;

struct ImageDescriptor
{
public:
	int m_totalNumberOfPointsInROI = 0;
	cv::Point2f m_globalMeanPosition;
	std::vector<float> m_globalStdDev = { 0, 0 }; 	// global standard deviation of x and y
	float m_globalSkewness = 0;						// global skewness = standard deviation of x / standard deviation of y
	float m_globalDensityInROI = 0;					// global density = (# of scratch points) / ROI
	float m_globalDensityInEffectiveROI = 0;		// effective density = (# of scratch points ) / (bounding box of all scratch points)

	int m_totalNumberOfPointsOfLargestCluster = 0;			// # of points in the largest cluster
	cv::Point2f m_largestClusterMeanPosition;				// mean position of largest cluster
	std::vector<float> m_largestClusterStdDev = { 0, 0 };	// largest cluster's standard deviation of x and y
	float m_largestClusterSkewness = 0;						// skewness = standard deviation of x / standard deviation of y

	int m_numberOfDenseClusters = 0;				// # of dense clusters

	double m_largestClusterEigenvalueRatio = 0;
	double m_largestClusterLargeEigenValue = 0;
	double m_largestClusterSmallEigenValue = 0;
	double m_largestClusterOrientation = 0;

public:
	static double CalculateFeatureDistance(ImageDescriptor a, ImageDescriptor b)
	{
		double orientationDiff = 0;
		double eigValueRatioDiff = 0;
		double penalty = 0;

		if (a.m_totalNumberOfPointsOfLargestCluster > 50 && b.m_totalNumberOfPointsOfLargestCluster > 50)
		{
			orientationDiff = std::abs(a.m_largestClusterOrientation - b.m_largestClusterOrientation) * 50;

			auto upper = (a.m_largestClusterEigenvalueRatio > b.m_largestClusterEigenvalueRatio) ? a.m_largestClusterEigenvalueRatio : b.m_largestClusterEigenvalueRatio;
			auto down = (a.m_largestClusterEigenvalueRatio <= b.m_largestClusterEigenvalueRatio) ? a.m_largestClusterEigenvalueRatio : b.m_largestClusterEigenvalueRatio;

			eigValueRatioDiff = (upper / down) * 50;
		}

		double largestClusterPointsRatio = std::max(a.m_totalNumberOfPointsOfLargestCluster, b.m_totalNumberOfPointsOfLargestCluster) / std::min(a.m_totalNumberOfPointsOfLargestCluster, b.m_totalNumberOfPointsOfLargestCluster) * 100;
		double skewRatio = std::max(a.m_globalSkewness, b.m_globalSkewness) / std::min(a.m_globalSkewness, b.m_globalSkewness) * 50;
		double densityRatio = std::max(a.m_globalDensityInEffectiveROI, b.m_globalDensityInEffectiveROI) / std::min(a.m_globalDensityInEffectiveROI, b.m_globalDensityInEffectiveROI) * 50;
		double totalPointRatio = std::max(a.m_totalNumberOfPointsInROI, b.m_totalNumberOfPointsInROI) / std::min(a.m_totalNumberOfPointsInROI, b.m_totalNumberOfPointsInROI) * 20;
		double numOfDenseClustersDiff = std::abs(a.m_numberOfDenseClusters - b.m_numberOfDenseClusters) * 1000;

		return orientationDiff + largestClusterPointsRatio + totalPointRatio + eigValueRatioDiff + penalty + skewRatio + densityRatio + numOfDenseClustersDiff;
	}
};

/************************************************************************/
/**************			Utility Functions					*************/
/************************************************************************/
void GetTopNClusters(const std::unordered_map<int, MeanShiftCluster>& in_clusters, const int in_N, std::vector<MeanShiftCluster>& out_sortedArray);
void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, MeanShiftCluster>& out_clusters);
void CaclculateGradientMap(const cv::Mat &in_imageMat, cv::Mat& out_edgeMap, double in_gradX_alpha, double in_gradY_beta);
bool IsThisPointCloseToOneOfContours(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point in_thisPoint, double in_distance);
bool IsThisPointInsideOneOfContours(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point& in_thisPoint);
bool IsThisContourContainedInROI(const std::vector<cv::Point>& in_points, const cv::Size in_imageSize, const cv::Rect in_ROI);
bool IsThisPointInROI(const cv::Rect in_roi, const cv::Point in_point);
bool IsThisPointNearByROI(const cv::Rect& in_roi, const cv::Point& in_point, unsigned int in_distance);
bool IsContourInsideCarBody(const std::vector<cv::Point>& in_contour, const std::vector<cv::Point>& in_carBodyContourPoints);
void GetPointsInContour(const std::vector<cv::Point> &in_contour, const double in_distanceFromBoundaryToBeInsidePoint, std::vector<cv::Point> &out_insidePoints);
void ResizeImageUsingThreshold(cv::Mat& in_targetImage, int in_totalPixelThreshold);

/************************************************************************/
/**************			Main Functions						*************/
/************************************************************************/
bool ExtractCarBody(const cv::Mat& in_srcImage, const cv::Rect in_ROI, cv::Mat& out_carBodyBinaryImage, std::vector<cv::Point>& out_carBodyContour);
void DetectScratchPointsInsideCarBody(const cv::Mat in_targetImage, const cv::Rect in_ROI, const cv::Mat in_carBodyBinaryImage, const std::vector<cv::Point> in_carBodyContourPoints,
	std::vector<cv::Point2f> &out_scratchPoints);
void GroupClusters(const std::vector<cv::Point2f>& in_scratchPoints, std::vector<Cluster_DBSCAN>& out_clusters);
void ExtractImageDescriptorFromMat(const cv::Mat& in_targetImage, ImageDescriptor& out_analyzeResult, bool in_bShowExperimentResult = false);

/************************************************************************/
/************       Functions for Feature Extraction       **************/
/************************************************************************/
void GetMeanStdDevFromPoints(const std::vector<cv::Point2f>& in_scratchPoints, cv::Point2f &out_meanPoint, std::vector<float>& out_stddev);
double GetSkewness(double in_xStdDev, double in_yStdDev);
void GetBoundingBoxOfScratchPoints(const cv::Size& in_imageSize, const std::vector<cv::Point2f>& in_scratchPoints, bool in_bExcludingOutlier, cv::Rect& out_boundingBox,
	const unsigned int in_outlierTolerance = 5);
void ComputePCA_VisualizeItsResult(const cv::Mat& img, const std::vector<cv::Point2f>& in_pts, double &out_largeEigValue, double &out_smallEigValue, double &out_degree, cv::Mat& out_drawnImage);