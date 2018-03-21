#pragma once
#include <vector>
#include <unordered_map>
#include <opencv2\opencv.hpp>

// 디버깅용 전처리기
#ifndef NDEBUG
#define FOR_DEBUG			true
#endif

// Class forward declartion
class Cluster;

struct AlgorithmParameter
{
private:
	// Mean Shift 하기 이전에 L값을 몇 배로 나눌지 결정하는 숫자
	double m_lValueDivider;

	// Mean Shift Segmentation 관련 파라미터
	double m_spatialBandwidth;
	double m_colorBandwidth;
		
	// 클러스터 이미지도 만들것인지
	bool m_bGetClusterImage;

	// 민쉬프트 세그멘테이션 결과도 구할 것인지
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
	// 피부색도 검출할것인지.. (이건 다른 프로젝트때문에 있는 것..)
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

// 입력된 이미지에서, 바디부분만 뽑아낸다.
bool ExtractCarBody(const cv::Mat& in_srcImage, const AlgorithmParameter& in_parameter, AlgorithmResult& out_finalParameter);


/************************************************************************/
/**************			Internal Functions					*************/
/************************************************************************/

void PerformColorMergingTask(const Cluster & inout_seedCluster, const std::unordered_map<int, Cluster>& in_clusters,
	const cv::Mat& in_lValueDividedHSVMat, cv::Mat& inout_labelMap,
	const double in_lValudDivider, const std::set<int>& in_backgroundIndices, const cv::Size& in_limitBox, int in_maxTrial, Cluster& out_mergedCluster);

// 클러스터 (in_centerCluster)에 인접한 클러스터들의 레이블들을 반환한다.
void GetLabelsOfAdjacentClusters(const cv::Mat &in_labelMap, const Cluster& in_centerCluster, const cv::Rect& in_ROI, std::set<int> &out_labels);

// in_limitBox의 테두리를 넘지 않는 선에서, in_rect를 (offsetX, offsetY)만큼 평행이동하며, diffWidth, diffHeight만큼 in_rect의 너비, 높이를 변화시킨다.
void ExpandRectInAnyFourDirections(cv::Size in_limitBox, cv::Rect& in_rect, int offsetX, int offsetY, int diffWidth, int diffHeight);

// in_cluster1의 색상 분포와 in_cluster2의 색상 분포를 비교하여 Bhattacharyya Coefficient를 구한다. 0에 가까울 수록 유사하다.
double GetHSVBhattaCoefficient(const cv::Mat& in_referenceLuvImage, const Cluster &in_cluster1, const Cluster &in_cluster2, int channelNumber, int in_nBin);

// mean shift filtering된 이미지의 in_ROI 영역에서 clustering을 수행한다. 
void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, Cluster>& out_clusters);

// 지정된 ROI(in_ROI)에서 가장 비중이 큰 클러스터의 레이블을 찾아 리턴한다.
void FindSeedClusterInROI(const cv::Mat &in_labelMap, const std::set<int> &in_backgroundIndices, const cv::Rect& in_ROI, int& out_seedLabel);

// 백그라운드 클러스터들의 레이블을 얻는다. 결과는 out_backgroundIndicesSet에 저장된다.
void GetBackgroundClusterIndices(const cv::Size &in_originalImageSize, const cv::Mat &in_labelMap, int in_marginLegnth, std::set<int> &out_backgroundIndiciesSet);

// 입력받은 클러스터가 레이블 맵(in_labelMap) 기준으로 특정 클러스터에 의해 둘러싸여있는지 판단하는 함수. in_rangeToCover은 둘러싸여 있는 범위를 말함. in_ratio는 몇 %를 둘러싸야 인정받는지 나타내는 지수
bool IsClusterWrappedByCertainCluster(const Cluster &in_cluster, const cv::Mat & in_labelMap, int in_rangeToCover, float in_ratio, int& out_labelOfWrapperCluster);

// in_cluster에 인접한 클러스터들을 반환한다, 또한 -1이 인접하다면 그것과 관련된 포인트들도 반환한다.
void GetAllAdjacentLabelsAndTheirFrequency(const Cluster& in_cluster, const cv::Mat& in_labelMap, int in_rangeToCover, std::unordered_map<int, int> &out_labelAndItsFrequency, std::vector<cv::Point>& out_minusPoints);

// 창섭이 코드
cv::Mat GetAlphaMap(const cv::Mat &labelMap, const cv::Rect &ROI, const Cluster& mainCluster);

/************************************************************************/
/**************			Utility Functions					*************/
/************************************************************************/
// 클러스터들의 Contour를 그린다.
void DrawContoursOfClusters(cv::Mat & in_targetImage, const std::unordered_map<int, Cluster>& in_clusters, cv::Scalar in_color);
// 단일 클러스의 Contour를 그린다.
void DrawOuterContourOfCluster(cv::Mat &in_targetImage, const Cluster& in_cluster, cv::Scalar in_color);
// 입력된 Luv Color를 HSV Color로 변환한다. (주의! 내부에 scaling factor가 하드코딩되어 있다.)
void GetOriginalHSVColorFromHalfedLuv(const cv::Point3i& in_luvColor, double in_factor, cv::Point3i& out_hsvColor);
// 사각형이 원본 이미지의 영역에서 벗어났는지 확인한다.
bool IsOutOfRange(const cv::Mat &in_originalImage, const cv::Rect &in_rect);
// 입력받은 맵(in_map)에서 Value가 최대인 녀석을 찾되, value가 kCriteria보다 큰 녀석을 찾는다. 만일 없다면 false를 반환한다.
bool SearchMapForMaxPair(const std::unordered_map<int, int>& in_map, const int kMinimumValue, std::pair<int, int> &out_keyValuePair);
// 이게 CreateAlphaMap함수를 내부적으로 호출함.
void FindAllOuterPointsOfCluster(const cv::Size& in_frameSize, const Cluster &in_cluster, std::vector<cv::Point> &out_points);
// 입력받은 클러스터 (in_cluster)의 AlphaMap(out_alphaMap)을 만든다.
void CreateAlphaMapFromCluster(const cv::Size& in_alphaMapSize, const Cluster& in_cluster, cv::Mat& out_alphaMap);
// Cluster 내부 Point들의 색상을 모두 out_mat에 뿌려준다. 
void ProjectClusterIntoMat(const Cluster& in_cluster, cv::Mat &out_mat);
// in_clusters 내부에서 가장 큰 규모의 Cluster를 구하고 그 레이블을 저장한다.
void FindBiggestCluster(const std::unordered_map<int, Cluster>& in_clusters, int& out_biggestClusterIndex);
