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

struct AlgorithmParameter
{
public:
	// Mean Shift 하기 이전에 L값을 몇 배로 나눌지 결정하는 숫자
	double m_lValueDivider;

	// Mean Shift Segmentation 관련 파라미터
	double m_spatialBandwidth;
	double m_colorBandwidth;
	// 민쉬프트 세그멘테이션 결과도 구할 것인지
	bool m_bGetMeanShiftSegmentationResult;

	// 컬러 레이블맵을 얻을 것인지
	bool m_bGetColorLabelMap;

	// 그래디언트 magnitude 이미지를 얻을 것인지
	bool m_bGetGradientMap;

	bool m_bGetCornerMap;

	// 컨투어가 있는 이미지를 얻을 것인지
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

// Blob Detection을 통해 결함이 있을만한 영역을 찾는다.
void FindPossibleDefectAreasUsingBlobDetection(const cv::Mat &in_imageMat, const std::vector<cv::Point> &out_centerPointsOfPossibleAreas);

// in_cluster (입력받은 클러스터)를 기준으로 Label Map을 업데이트 한다. 
void UpdateLabelMap(const std::unordered_map<int, MeanShiftCluster>& in_clusters, cv::Mat& inout_labelMap);


void GetTopNClusters(const std::unordered_map<int, MeanShiftCluster>& in_clusters, const int in_N, std::vector<MeanShiftCluster>& out_sortedArray);

// Color Merging을 하도록 한다.
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
// mean shift filtering된 이미지의 in_ROI 영역에서 clustering을 수행한다. 
void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, MeanShiftCluster>& out_clusters);

// 입력받은 이미지에서 EdgeMap을 만들어준다.
void CaclculateGradientMap(const cv::Mat &in_imageMat, cv::Mat& out_edgeMap, double in_gradX_alpha, double in_gradY_beta);

// 사각형이 원본 이미지의 영역에서 벗어났는지 확인한다.
bool IsOutOfRange(const cv::Mat &in_originalImage, const cv::Rect &in_rect);

// 입력받은 맵(in_map)에서 Value가 최대인 녀석을 찾되, value가 kMinimumValue보다 큰 녀석을 찾는다. 만일 없다면 false를 반환한다.
bool SearchMapForMaxPair(const std::unordered_map<int, int>& in_map, const int kMinimumValue, std::pair<int, int> &out_keyValuePair);

// in_clusters 내부에서 가장 큰 규모의 Cluster를 구하고 그 레이블을 저장한다.
void FindBiggestCluster(const std::unordered_map<int, MeanShiftCluster>& in_clusters, int& out_biggestClusterIndex);

// LabelMap을 Visualize해준다.
void VisualizeLabelMap(const cv::Mat& in_labelMap, cv::Mat& out_colorLabelMap);

bool IsThisPointCloseToGivenContour(const cv::Point& in_point, const std::vector<cv::Point> &in_givenContour, double in_distance);
bool IsThisPointCloseToOneOfContours(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point in_thisPoint, double in_distance);
bool IsThisPointInsideOneOfContours(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point& in_thisPoint);
cv::Scalar GetAverageColorOfPointsArray(cv::Mat in_srcImage, const std::vector<cv::Point> &in_points);
bool IsThisContourContainedInROI(const std::vector<cv::Point>& in_points, const cv::Size in_imageSize, const cv::Rect in_ROI);
bool IsThisPointInROI(const cv::Rect in_roi, const cv::Point in_point);

// 현재 포인트가 ROI 근처인지 판단해준다. 기준은 in_distance
bool IsThisPointNearByROI(const cv::Rect& in_roi, const cv::Point& in_point, unsigned int in_distance);
bool IsContourInsideCarBody(const std::vector<cv::Point>& in_contour, const std::vector<cv::Point>& in_carBodyContourPoints);
// Contour안에 존재하는 Points들을 얻는다.
void GetPointsInContour(const std::vector<cv::Point> &in_contour, const double in_distanceFromBoundaryToBeInsidePoint, std::vector<cv::Point> &out_insidePoints);

void ResizeImageUsingThreshold(cv::Mat& in_targetImage, int in_totalPixelThreshold);

void GaussianBlurToContour(cv::Mat& in_targetImage, const std::vector<cv::Point> &in_contour);

/************************************************************************/
/**************			Main Functions						*************/
/************************************************************************/

// 입력된 이미지에서, 바디부분만 뽑아낸다.
bool ExtractCarBody(const cv::Mat& in_srcImage, const cv::Rect in_ROI, cv::Mat& out_carBodyBinaryImage, std::vector<cv::Point>& out_carBodyContour,
	const AlgorithmParameter& in_parameter, AlgorithmResult& out_finalParameter);

// 스크래치 영역을 
void DetectScratchPointsFromExtractionResult(const cv::Mat in_targetImage, const cv::Rect in_ROI, const cv::Mat in_carBodyBinaryImage, const std::vector<cv::Point> in_carBodyContourPoints,
	std::vector<cv::Point2f> &out_scratchPoints);

// DBSCAN 이용하여 스크래치 점들 그룹핑
void ConstructScratchClusters(const std::vector<cv::Point2f>& in_scratchPoints, std::vector<Cluster_DBSCAN>& out_clusters);

/**********************************************************************/
/***************** Functions for Feature Extraction *******************/
/**********************************************************************/

// 스크래치 포인트들의 평균과 표준편차를 구한다. 
void GetMeanStdDevFromPoints(const std::vector<cv::Point2f>& in_scratchPoints, cv::Point2f &out_meanPoint, std::vector<float>& out_stddev);

double GetSkewness(double in_xStdDev, double in_yStdDev);

void GetBoundingBoxOfScratchPoints(const cv::Size& in_imageSize, const std::vector<cv::Point2f>& in_scratchPoints, bool in_bExcludingOutlier, cv::Rect& out_boundingBox,
	const unsigned int in_outlierTolerance = 5);

double ComputePCA_VisualizeItsResult(const cv::Mat& img, const std::vector<cv::Point2f>& in_pts, double &out_largeEigValue, double &out_smallEigValue, cv::Mat& out_drawnImage);

void ExtractImageDescriptorFromMat(const cv::Mat& in_targetImage, ImageDescriptor& out_analyzeResult, bool in_bShowExperimentResult = false);