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

void PerformColorMergingTask(const Cluster & inout_seedCluster, const std::unordered_map<int, Cluster>& in_clusters,
	const cv::Mat& in_lValueDividedHSVMat, cv::Mat& inout_labelMap,
	const double in_lValudDivider, const std::set<int>& in_backgroundIndices, const cv::Size& in_limitBox, int in_maxTrial, Cluster& out_mergedCluster);

// Ŭ������ (in_centerCluster)�� ������ Ŭ�����͵��� ���̺���� ��ȯ�Ѵ�.
void GetLabelsOfAdjacentClusters(const cv::Mat &in_labelMap, const Cluster& in_centerCluster, const cv::Rect& in_ROI, std::set<int> &out_labels);

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

// �Է¹��� Ŭ�����Ͱ� ���̺� ��(in_labelMap) �������� Ư�� Ŭ�����Ϳ� ���� �ѷ��ο��ִ��� �Ǵ��ϴ� �Լ�. in_rangeToCover�� �ѷ��ο� �ִ� ������ ����. in_ratio�� �� %�� �ѷ��ξ� �����޴��� ��Ÿ���� ����
bool IsClusterWrappedByCertainCluster(const Cluster &in_cluster, const cv::Mat & in_labelMap, int in_rangeToCover, float in_ratio, int& out_labelOfWrapperCluster);

// in_cluster�� ������ Ŭ�����͵��� ��ȯ�Ѵ�, ���� -1�� �����ϴٸ� �װͰ� ���õ� ����Ʈ�鵵 ��ȯ�Ѵ�.
void GetAllAdjacentLabelsAndTheirFrequency(const Cluster& in_cluster, const cv::Mat& in_labelMap, int in_rangeToCover, std::unordered_map<int, int> &out_labelAndItsFrequency, std::vector<cv::Point>& out_minusPoints);

// â���� �ڵ�
cv::Mat GetAlphaMap(const cv::Mat &labelMap, const cv::Rect &ROI, const Cluster& mainCluster);

/************************************************************************/
/**************			Utility Functions					*************/
/************************************************************************/
// Ŭ�����͵��� Contour�� �׸���.
void DrawContoursOfClusters(cv::Mat & in_targetImage, const std::unordered_map<int, Cluster>& in_clusters, cv::Scalar in_color);
// ���� Ŭ������ Contour�� �׸���.
void DrawOuterContourOfCluster(cv::Mat &in_targetImage, const Cluster& in_cluster, cv::Scalar in_color);
// �Էµ� Luv Color�� HSV Color�� ��ȯ�Ѵ�. (����! ���ο� scaling factor�� �ϵ��ڵ��Ǿ� �ִ�.)
void GetOriginalHSVColorFromHalfedLuv(const cv::Point3i& in_luvColor, double in_factor, cv::Point3i& out_hsvColor);
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
