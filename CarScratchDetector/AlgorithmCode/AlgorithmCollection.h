#pragma once
#include <vector>
#include <unordered_map>
#include <opencv2\opencv.hpp>
#include "..\DBSCAN.h"

// ������ ��ó����
#ifndef NDEBUG
#define FOR_DEBUG			true
#endif


// Class forward declartion
class MeanShiftCluster;

struct AlgorithmParameter
{
public:
	// Mean Shift �ϱ� ������ L���� �� ��� ������ �����ϴ� ����
	double m_lValueDivider;

	// Mean Shift Segmentation ���� �Ķ����
	double m_spatialBandwidth;
	double m_colorBandwidth;
	// �ν���Ʈ ���׸����̼� ����� ���� ������
	bool m_bGetMeanShiftSegmentationResult;

	// �÷� ���̺���� ���� ������
	bool m_bGetColorLabelMap;

	// �׷����Ʈ magnitude �̹����� ���� ������
	bool m_bGetGradientMap;

	bool m_bGetCornerMap;

	// ����� �ִ� �̹����� ���� ������
	bool m_bGetContouredMap;

public:
	void SetParameterValues
	(
		double in_lValueDivider,
		double in_sp,
		double in_sr,
		bool in_bGetMeanShiftSegmentationResult,
		bool in_bGetColorLabelMap,
		bool in_bGetGradientMap,
		bool in_bGetCornerMap,
		bool in_bGetContouredMap
	)
	{
		m_lValueDivider = in_lValueDivider;
		m_spatialBandwidth = in_sp;
		m_colorBandwidth = in_sr;
		m_bGetMeanShiftSegmentationResult = in_bGetMeanShiftSegmentationResult;
		m_bGetGradientMap = in_bGetGradientMap;
		m_bGetColorLabelMap = in_bGetColorLabelMap;
		m_bGetGradientMap = in_bGetGradientMap;
		m_bGetContouredMap = in_bGetContouredMap;
	}
	void SetToDefault()
	{
		m_lValueDivider = 1.0;
		m_spatialBandwidth = 16;
		m_colorBandwidth = 16;
		m_bGetColorLabelMap = false;
		m_bGetGradientMap = false;
		m_bGetMeanShiftSegmentationResult = false;
		m_bGetContouredMap = true;
	}
};
struct AlgorithmResult
{
public:
	enum class TimerIdentifier
	{
		BGRToLuvElapsedTime = 0,
		LDivisionElapsedTime,
		MeanShiftElapsedTime,
		FindingSeedClusterElapsedTime,
		MergingTaskElapsedTime,
		ExtractOuterContourElapsedTime,
		TotalElapsedTime
	};

	inline cv::Mat& GetResultMat()
	{
		return m_resultMat;
	}
	inline cv::Mat& GetSegmentedMat()
	{
		return m_segmentedMat;
	}
	inline cv::Mat& GetClusteredMat()
	{
		return m_clusteredMat;
	}
	double GetElapsedTime(TimerIdentifier in_identifier) const
	{
		return m_elapsedTimes.at(in_identifier);
	}
	double GetFinalSpatialBandwidth() const
	{
		return m_finalSpatialBandwidth;
	}
	double GetFinalColorBandwidth() const
	{
		return m_finalColorBandwidth;
	}
	double GetFinalLDivider() const
	{
		return m_finalLDivider;
	}

public:
	void SetElapsedTime(TimerIdentifier in_identifier, double in_time)
	{
		m_elapsedTimes[in_identifier] = in_time;
	}
	void SetFinalSpatialBandwidth(double in_sp)
	{
		m_finalSpatialBandwidth = in_sp;
	}
	void SetFinalColorBandwidth(double in_sr)
	{
		m_finalColorBandwidth = in_sr;
	}
	void SetFinalLDivider(double in_lDivider)
	{
		m_finalLDivider = in_lDivider;
	}

	void SetResultMat(cv::Mat in_mat)
	{
		m_resultMat = in_mat;
	}
	void SetSegmentedMat(cv::Mat in_segmentedMat)
	{
		m_segmentedMat = in_segmentedMat;
	}
	void SetClusteredMat(cv::Mat in_clusteredMat)
	{
		m_clusteredMat = in_clusteredMat;
	}

	void Reset()
	{
		m_resultMat.release();
		m_segmentedMat.release();
		m_clusteredMat.release();
		m_elapsedTimes.clear();

		m_finalColorBandwidth = m_finalSpatialBandwidth = m_finalLDivider = 0;
	}

private:
	cv::Mat m_resultMat;
	cv::Mat m_segmentedMat;
	cv::Mat m_clusteredMat;
	std::unordered_map<TimerIdentifier, double> m_elapsedTimes;

	double m_finalSpatialBandwidth;
	double m_finalColorBandwidth;
	double m_finalLDivider;
};

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
#pragma optimize("gpsy", off)
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
		double skewRatio = std::max(a.m_globalSkewness,b.m_globalSkewness) / std::min (a.m_globalSkewness, b.m_globalSkewness) * 50;
		double densityRatio = std::max(a.m_globalDensityInEffectiveROI, b.m_globalDensityInEffectiveROI) / std::min(a.m_globalDensityInEffectiveROI, b.m_globalDensityInEffectiveROI) * 50;
		double totalPointRatio = std::max(a.m_totalNumberOfPointsInROI, b.m_totalNumberOfPointsInROI) / std::min(a.m_totalNumberOfPointsInROI, b.m_totalNumberOfPointsInROI) * 20;
		double numOfDenseClustersDiff = std::abs(a.m_numberOfDenseClusters - b.m_numberOfDenseClusters) * 1000;

		return orientationDiff + largestClusterPointsRatio + totalPointRatio + eigValueRatioDiff + penalty + skewRatio + densityRatio + numOfDenseClustersDiff;
	}
#pragma optimize("gpsy", on)
};


/************************************************************************/
/**************			Internal Functions					*************/
/************************************************************************/

// Blob Detection�� ���� ������ �������� ������ ã�´�.
void FindPossibleDefectAreasUsingBlobDetection(const cv::Mat &in_imageMat, const std::vector<cv::Point> &out_centerPointsOfPossibleAreas);

// in_cluster (�Է¹��� Ŭ������)�� �������� Label Map�� ������Ʈ �Ѵ�. 
void UpdateLabelMap(const std::unordered_map<int, MeanShiftCluster>& in_clusters, cv::Mat& inout_labelMap);


void GetTopNClusters(const std::unordered_map<int, MeanShiftCluster>& in_clusters, const int in_N, std::vector<MeanShiftCluster>& out_sortedArray);

// Color Merging�� �ϵ��� �Ѵ�.
void PerformColorMergingFromSeedClusterAndUpdateClusterList(std::unordered_map <int, MeanShiftCluster> &in_updatedClusterList, const int in_seedIndex);

void Find_TopN_BiggestClusters(const std::unordered_map<int, MeanShiftCluster> &in_clusters, const int in_count, std::vector<int> &out_labels);

void GetAdequacyScoresToBeSeed(const cv::Size in_imageSize,
	const std::unordered_map<int, MeanShiftCluster> &in_clusters,
	const int in_numberOfCandidates,
	const int in_numberOfRandomSamples,
	const std::vector<int> in_candidateClusterLabels,
	int& out_seedLabel);

/************************************************************************/
/**************			Utility Functions					*************/
/************************************************************************/
// mean shift filtering�� �̹����� in_ROI �������� clustering�� �����Ѵ�. 
void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, MeanShiftCluster>& out_clusters);

// �Է¹��� �̹������� EdgeMap�� ������ش�.
void CaclculateGradientMap(const cv::Mat &in_imageMat, cv::Mat& out_edgeMap, double in_gradX_alpha, double in_gradY_beta);

// �簢���� ���� �̹����� �������� ������� Ȯ���Ѵ�.
bool IsOutOfRange(const cv::Mat &in_originalImage, const cv::Rect &in_rect);

// �Է¹��� ��(in_map)���� Value�� �ִ��� �༮�� ã��, value�� kMinimumValue���� ū �༮�� ã�´�. ���� ���ٸ� false�� ��ȯ�Ѵ�.
bool SearchMapForMaxPair(const std::unordered_map<int, int>& in_map, const int kMinimumValue, std::pair<int, int> &out_keyValuePair);

// in_clusters ���ο��� ���� ū �Ը��� Cluster�� ���ϰ� �� ���̺��� �����Ѵ�.
void FindBiggestCluster(const std::unordered_map<int, MeanShiftCluster>& in_clusters, int& out_biggestClusterIndex);

// LabelMap�� Visualize���ش�.
void VisualizeLabelMap(const cv::Mat& in_labelMap, cv::Mat& out_colorLabelMap);

bool IsThisPointCloseToGivenContour(const cv::Point& in_point, const std::vector<cv::Point> &in_givenContour, double in_distance);
bool IsThisPointCloseToOneOfContours(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point in_thisPoint, double in_distance);
bool IsThisPointInsideOneOfContours(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point& in_thisPoint);
cv::Scalar GetAverageColorOfPointsArray(cv::Mat in_srcImage, const std::vector<cv::Point> &in_points);
bool IsThisContourContainedInROI(const std::vector<cv::Point>& in_points, const cv::Size in_imageSize, const cv::Rect in_ROI);
bool IsThisPointInROI(const cv::Rect in_roi, const cv::Point in_point);

// ���� ����Ʈ�� ROI ��ó���� �Ǵ����ش�. ������ in_distance
bool IsThisPointNearByROI(const cv::Rect& in_roi, const cv::Point& in_point, unsigned int in_distance);
bool IsContourInsideCarBody(const std::vector<cv::Point>& in_contour, const std::vector<cv::Point>& in_carBodyContourPoints);
// Contour�ȿ� �����ϴ� Points���� ��´�.
void GetPointsInContour(const std::vector<cv::Point> &in_contour, const double in_distanceFromBoundaryToBeInsidePoint, std::vector<cv::Point> &out_insidePoints);

void ResizeImageUsingThreshold(cv::Mat& in_targetImage, int in_totalPixelThreshold);

void GaussianBlurToContour(cv::Mat& in_targetImage, const std::vector<cv::Point> &in_contour);

/************************************************************************/
/**************			Main Functions						*************/
/************************************************************************/

// �Էµ� �̹�������, �ٵ�κи� �̾Ƴ���.
bool ExtractCarBody(const cv::Mat& in_srcImage, const cv::Rect in_ROI, cv::Mat& out_carBodyBinaryImage, std::vector<cv::Point>& out_carBodyContour,
	const AlgorithmParameter& in_parameter, AlgorithmResult& out_finalParameter);

// ��ũ��ġ ������ 
void DetectScratchPointsFromExtractionResult(const cv::Mat in_targetImage, const cv::Rect in_ROI, const cv::Mat in_carBodyBinaryImage, const std::vector<cv::Point> in_carBodyContourPoints,
	std::vector<cv::Point2f> &out_scratchPoints);

// DBSCAN �̿��Ͽ� ��ũ��ġ ���� �׷���
void ConstructScratchClusters(const std::vector<cv::Point2f>& in_scratchPoints, std::vector<Cluster_DBSCAN>& out_clusters);

/**********************************************************************/
/***************** Functions for Feature Extraction *******************/
/**********************************************************************/

// ��ũ��ġ ����Ʈ���� ��հ� ǥ�������� ���Ѵ�. 
void GetMeanStdDevFromPoints(const std::vector<cv::Point2f>& in_scratchPoints, cv::Point2f &out_meanPoint, std::vector<float>& out_stddev);

double GetSkewness(double in_xStdDev, double in_yStdDev);

void GetBoundingBoxOfScratchPoints(const cv::Size& in_imageSize, const std::vector<cv::Point2f>& in_scratchPoints, bool in_bExcludingOutlier, cv::Rect& out_boundingBox,
	const unsigned int in_outlierTolerance = 5);

double ComputePCA_VisualizeItsResult(const cv::Mat& img, const std::vector<cv::Point2f>& in_pts, double &out_largeEigValue, double &out_smallEigValue, cv::Mat& out_drawnImage);

void ExtractImageDescriptorFromMat(const cv::Mat& in_targetImage, ImageDescriptor& out_analyzeResult, bool in_bShowExperimentResult = false);