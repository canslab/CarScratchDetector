#pragma once
#include <vector>
#include <unordered_map>
#include <opencv2\opencv.hpp>

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

/************************************************************************/
/**************			Clinet Functions					*************/
/************************************************************************/
// �Էµ� �̹�������, �ٵ�κи� �̾Ƴ���.
bool ExtractCarBody(const cv::Mat& in_srcImage, const AlgorithmParameter& in_parameter, AlgorithmResult& out_finalParameter);

/************************************************************************/
/**************			Internal Functions					*************/
/************************************************************************/
// mean shift filtering�� �̹����� in_ROI �������� clustering�� �����Ѵ�. 
void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, MeanShiftCluster>& out_clusters);

// �Է¹��� �̹������� EdgeMap�� ������ش�.
void CaclculateGradientMap(const cv::Mat &in_imageMat, cv::Mat& out_edgeMap);

// Blob Detection�� ���� ������ �������� ������ ã�´�.
void FindPossibleDefectAreasUsingBlobDetection(const cv::Mat &in_imageMat, const std::vector<cv::Point> &out_centerPointsOfPossibleAreas);

// Contour�ȿ� �����ϴ� Points���� ��´�.
void GetPointsInContour(const cv::Size& in_imageSize, const std::vector<cv::Point> &in_contour, std::vector<cv::Point> &out_insidePoints);

// in_cluster (�Է¹��� Ŭ������)�� �������� Label Map�� ������Ʈ �Ѵ�. 
void UpdateLabelMap(const std::unordered_map<int, MeanShiftCluster>& in_clusters, cv::Mat& inout_labelMap);

// Ŭ�����͵��� Contour�� �׸���.
void DrawContoursOfClusters(cv::Mat & in_targetImage, const std::unordered_map<int, MeanShiftCluster>& in_clusters, cv::Scalar in_color);
// ���� Ŭ������ Contour�� �׸���.
void DrawOuterContourOfCluster(cv::Mat &in_targetImage, const MeanShiftCluster& in_cluster, cv::Scalar in_color);
// �簢���� ���� �̹����� �������� ������� Ȯ���Ѵ�.
bool IsOutOfRange(const cv::Mat &in_originalImage, const cv::Rect &in_rect);
// �Է¹��� ��(in_map)���� Value�� �ִ��� �༮�� ã��, value�� kMinimumValue���� ū �༮�� ã�´�. ���� ���ٸ� false�� ��ȯ�Ѵ�.
bool SearchMapForMaxPair(const std::unordered_map<int, int>& in_map, const int kMinimumValue, std::pair<int, int> &out_keyValuePair);
// �̰� CreateAlphaMap�Լ��� ���������� ȣ����.
void FindAllOuterPointsOfCluster(const cv::Size& in_frameSize, const MeanShiftCluster &in_cluster, std::vector<cv::Point> &out_points);
// �Է¹��� Ŭ������ (in_cluster)�� AlphaMap(out_alphaMap)�� �����.
void CreateAlphaMapFromCluster(const cv::Size& in_alphaMapSize, const MeanShiftCluster& in_cluster, cv::Mat& out_alphaMap);
// Cluster ���� Point���� ������ ��� out_mat�� �ѷ��ش�. 
void ProjectClusterIntoMat(const MeanShiftCluster& in_cluster, cv::Mat &out_mat);
// in_clusters ���ο��� ���� ū �Ը��� Cluster�� ���ϰ� �� ���̺��� �����Ѵ�.
void FindBiggestCluster(const std::unordered_map<int, MeanShiftCluster>& in_clusters, int& out_biggestClusterIndex);

// Color Merging�� �ϵ��� �Ѵ�.
void PerformColorMergingFromSeedClusterAndUpdateClusterList(std::unordered_map <int, MeanShiftCluster> &in_updatedClusterList, const int in_seedIndex);

void Find_TopN_BiggestClusters(const std::unordered_map<int, MeanShiftCluster> &in_clusters, const int in_count, std::vector<int> &out_labels);

void GetAdequacyScoresToBeSeed(const cv::Size in_imageSize,
	const std::unordered_map<int, MeanShiftCluster> &in_clusters,
	const int in_numberOfCandidates,
	const int in_numberOfRandomSamples,
	const std::vector<int> in_candidateClusterLabels,
	int& out_seedLabel);

void VisualizeLabelMap(const cv::Mat& in_labelMap, cv::Mat& out_colorLabelMap);

// DBSCAN Method
//void PeformDBSCAN(std::vector<cv::Point> &in_points, std::vector<)