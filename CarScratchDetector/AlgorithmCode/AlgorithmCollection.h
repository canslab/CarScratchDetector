#pragma once
#include <vector>
#include <unordered_map>
#include <opencv2\opencv.hpp>

// ������ ��ó����
#ifndef NDEBUG
#define FOR_DEBUG			true
#endif

// Class forward declartion
class Cluster;

struct AlgorithmParameter
{
private:
	// Mean Shift �ϱ� ������ L���� �� ��� ������ �����ϴ� ����
	double m_lValueDivider;

	// Mean Shift Segmentation ���� �Ķ����
	double m_spatialBandwidth;
	double m_colorBandwidth;
		
	// Ŭ������ �̹����� ���������
	bool m_bGetClusterImage;

	// �ν���Ʈ ���׸����̼� ����� ���� ������
	bool m_bGetMeanShiftSegmentationResult;

public:
	void SetLValueDivider(double in_valueDivider)
	{
		m_lValueDivider = in_valueDivider;
	}
	void SetSpatialBandwidth(double in_sp)
	{
		m_spatialBandwidth = in_sp;
	}
	void SetColorBandwidth(double in_sr)
	{
		m_colorBandwidth = in_sr;
	}
	void SetToGetClusterImage(bool in_bSet)
	{
		m_bGetClusterImage = in_bSet;
	}
	void SetToGetSegmentedImage(bool in_bSet)
	{
		m_bGetMeanShiftSegmentationResult = in_bSet;
	}

public:
	void SetParameterValues(double in_lValueDivider, double in_sp, double in_sr, int in_backgroundTolerance, bool in_bGetClusterImage, bool in_bGetMeanShiftSegmentationResult)
	{
		m_lValueDivider = in_lValueDivider;
		m_spatialBandwidth = in_sp;
		m_colorBandwidth = in_sr;
		m_bGetClusterImage = in_bGetClusterImage;
		m_bGetMeanShiftSegmentationResult = in_bGetMeanShiftSegmentationResult;
	}
	void Reset()
	{
		m_lValueDivider = m_spatialBandwidth = m_colorBandwidth = m_bGetClusterImage = m_bGetMeanShiftSegmentationResult = 0;
	}
public:
	inline double GetLValueDivider() const
	{
		return m_lValueDivider;
	}
	inline double GetSpatialBandwidth() const
	{
		return m_spatialBandwidth;
	}
	inline double GetColorBandwidth() const
	{
		return m_colorBandwidth;
	}
	
	inline bool IsSetToGetClusterImage() const
	{
		return m_bGetClusterImage;
	}
	inline bool IsSetToGetSegmentedImage() const
	{
		return m_bGetMeanShiftSegmentationResult;
	}

public:
	// �Ǻλ��� �����Ұ�����.. (�̰� �ٸ� ������Ʈ������ �ִ� ��..)
	AlgorithmParameter() :m_spatialBandwidth(8), m_colorBandwidth(8), m_lValueDivider(2.5), m_bGetClusterImage(false), m_bGetMeanShiftSegmentationResult(false)
	{
	}
	AlgorithmParameter(double in_lValueDivider, double in_sp, double in_sr, int in_backgroundTolerance, bool in_bGetClusterImage, bool in_bGetMeanShiftSegmentationResult):
		m_lValueDivider(in_lValueDivider), m_spatialBandwidth(in_sp), m_colorBandwidth(in_sr), m_bGetClusterImage(in_bGetClusterImage), m_bGetMeanShiftSegmentationResult(in_bGetMeanShiftSegmentationResult)
	{

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

/************************************************************************/
/**************			Clinet Functions					*************/
/************************************************************************/

// �Էµ� �̹�������, �ٵ�κи� �̾Ƴ���.
bool ExtractCarBody(const cv::Mat& in_srcImage, const AlgorithmParameter& in_parameter, AlgorithmResult& out_finalParameter);


/************************************************************************/
/**************			Internal Functions					*************/
/************************************************************************/

// in_limitBox�� �׵θ��� ���� �ʴ� ������, in_rect�� (offsetX, offsetY)��ŭ �����̵��ϸ�, diffWidth, diffHeight��ŭ in_rect�� �ʺ�, ���̸� ��ȭ��Ų��.
void ExpandRectInAnyFourDirections(cv::Size in_limitBox, cv::Rect& in_rect, int offsetX, int offsetY, int diffWidth, int diffHeight);

// in_cluster1�� ���� ������ in_cluster2�� ���� ������ ���Ͽ� Bhattacharyya Coefficient�� ���Ѵ�. 0�� ����� ���� �����ϴ�.
double GetHSVBhattaCoefficient(const cv::Mat& in_referenceLuvImage, const Cluster &in_cluster1, const Cluster &in_cluster2, int channelNumber, int in_nBin);

// mean shift filtering�� �̹����� in_ROI �������� clustering�� �����Ѵ�. 
void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, Cluster>& out_clusters);

// ������ ROI(in_ROI)���� ���� ������ ū Ŭ�������� ���̺��� ã�� �����Ѵ�.
void FindSeedClusterInROI(const cv::Mat &in_labelMap, const std::set<int> &in_backgroundIndices, const cv::Rect& in_ROI, int& out_seedLabel);

// ��׶��� Ŭ�����͵��� ���̺��� ��´�. ����� out_backgroundIndicesSet�� ����ȴ�.
void GetBackgroundClusterIndices(const cv::Size &in_originalImageSize, const cv::Mat &in_labelMap, int in_marginLegnth, std::set<int> &out_backgroundIndiciesSet);

/************************************************************************/
/**************			Utility Functions					*************/
/************************************************************************/
// �־��� �̹��� (in_givenImage)���� �󱸰� (in_range)�� �ش��ϴ� ���� �����ϸ� �װ��� ��� 255�� ä���. ����, bInversion�� true�̸� ������Ų��.
void ThresholdImageWithinCertainInterval(cv::Mat& in_givenImage, std::vector<int>& in_range, bool bInversion, cv::Mat& out_binaryImage);

// �־��� ���� Ž������(�ܰ������� �־���)���� ������ �����Ѵ�.
void CaclculateEdgeMap(const cv::Mat &in_imageMat, cv::Mat& out_edgeMap);

void FindPossibleDefectAreasUsingBlobDetection(const cv::Mat &in_imageMat, const std::vector<cv::Point> &out_centerPointsOfPossibleAreas);

void GetPointsInContour(const cv::Size& in_imageSize, const std::vector<cv::Point> &in_contour, std::vector<cv::Point> &out_insidePoints);

void UpdateLabelMap(cv::Mat& inout_labelMap, const std::unordered_map<int, Cluster>& in_clusters);

// Ŭ�����͵��� Contour�� �׸���.
void DrawContoursOfClusters(cv::Mat & in_targetImage, const std::unordered_map<int, Cluster>& in_clusters, cv::Scalar in_color);
// ���� Ŭ������ Contour�� �׸���.
void DrawOuterContourOfCluster(cv::Mat &in_targetImage, const Cluster& in_cluster, cv::Scalar in_color);
// �簢���� ���� �̹����� �������� ������� Ȯ���Ѵ�.
bool IsOutOfRange(const cv::Mat &in_originalImage, const cv::Rect &in_rect);
// �Է¹��� ��(in_map)���� Value�� �ִ��� �༮�� ã��, value�� kCriteria���� ū �༮�� ã�´�. ���� ���ٸ� false�� ��ȯ�Ѵ�.
bool SearchMapForMaxPair(const std::unordered_map<int, int>& in_map, const int kMinimumValue, std::pair<int, int> &out_keyValuePair);
// �̰� CreateAlphaMap�Լ��� ���������� ȣ����.
void FindAllOuterPointsOfCluster(const cv::Size& in_frameSize, const Cluster &in_cluster, std::vector<cv::Point> &out_points);
// �Է¹��� Ŭ������ (in_cluster)�� AlphaMap(out_alphaMap)�� �����.
void CreateAlphaMapFromCluster(const cv::Size& in_alphaMapSize, const Cluster& in_cluster, cv::Mat& out_alphaMap);
// Cluster ���� Point���� ������ ��� out_mat�� �ѷ��ش�. 
void ProjectClusterIntoMat(const Cluster& in_cluster, cv::Mat &out_mat);

// in_clusters ���ο��� ���� ū �Ը��� Cluster�� ���ϰ� �� ���̺��� �����Ѵ�.
void FindBiggestCluster(const std::unordered_map<int, Cluster>& in_clusters, int& out_biggestClusterIndex);