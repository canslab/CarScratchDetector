#pragma once
#include <opencv2\opencv.hpp>
#include <unordered_map>

// 점들을 보유하는 Cluster 클래스
class Cluster
{
private:
	// 클러스터 대표 Luv컬러 값
	cv::Point3i m_colorInLuv;
	// 클러스터 대표 HSV 컬러 값
	cv::Point3i m_colorInHSV;
	// 클러스터 내부에 존재하는 모든 점(x, y좌표)들을 열벡터에 저장. 각 원소는 좌표 값이므로 2채널
	std::vector<cv::Point> m_pointsArray;
	// 클러스터내부의 레이블 값
	int m_label;

public:
	cv::Point2d GetCenterPoint() const
	{
		cv::Point2d centerPoint(0, 0);

		for (auto& eachPoint : m_pointsArray)
		{
			centerPoint.x += eachPoint.x;
			centerPoint.y += eachPoint.y;
		}
		centerPoint.x /= (double)m_pointsArray.size();
		centerPoint.y /= (double)m_pointsArray.size();

		return centerPoint;
	}

	// 클러스터를 바운딩하는 사각형
	cv::Rect m_boundedBox;

	// 클러스터가 보유하고 있는 점의 갯수를 반환				
	inline int GetTotalPoints() const
	{
		return m_pointsArray.size();
	}

	// 점들을 보관하고 있는 열 벡터를 반환함.
	inline const std::vector<cv::Point>& GetPointsArray() const
	{
		return m_pointsArray;
	}

	int GetLabel() const
	{
		return m_label;
	}

	// 클러스터 대표 색상값을 추출.
	inline const cv::Point3i& GetLuvColor() const
	{
		return m_colorInLuv;
	}
	inline const cv::Point3i& GetHSVColor() const
	{
		return m_colorInHSV;
	}
	inline const cv::Rect& GetBoundedBox() const
	{
		return m_boundedBox;
	}

public:
	void SetColorUsingLuvVector(const cv::Vec3b& in_luvColorVector)
	{
		this->m_colorInLuv.x = in_luvColorVector[0];
		this->m_colorInLuv.y = in_luvColorVector[1];
		this->m_colorInLuv.z = in_luvColorVector[2];

		// Luv 포맷을 HSV로도 저장해둔다.
		cv::Mat tempMat(1, 1, CV_8UC3, cv::Scalar(in_luvColorVector));
		cv::cvtColor(tempMat, tempMat, CV_Luv2BGR);
		cv::cvtColor(tempMat, tempMat, CV_BGR2HSV);

		auto& hsvValue = tempMat.at<cv::Vec3b>(0, 0);
		this->m_colorInHSV.x = hsvValue[0];
		this->m_colorInHSV.y = hsvValue[1];
		this->m_colorInHSV.z = hsvValue[2];
	}
	void SetBoundBox(const cv::Rect& in_rect)
	{
		m_boundedBox = in_rect;
	}
	void SetLabel(int in_label)
	{
		assert(in_label >= 0);
		m_label = in_label;
	}

	void AddPointsFromArray(const cv::Point in_points[], int nTotalPoints)
	{
		m_pointsArray.insert(m_pointsArray.begin(), in_points, in_points + nTotalPoints);
		// 바운딩 박스 업데이트
		m_boundedBox = cv::boundingRect(m_pointsArray);
	}
	bool AddPoint(const cv::Point &in_point)
	{
		// 없으면.. 등록해..
		if (std::find(m_pointsArray.begin(), m_pointsArray.end(), in_point) == m_pointsArray.end())
		{
			m_pointsArray.push_back(in_point);
			m_boundedBox = cv::boundingRect(m_pointsArray);
			return true;
		}
		else
		{
			return false;
		}
	}

public:
	// assignment
	Cluster& operator=(const Cluster& in_cluster)
	{
		auto &in_rect = in_cluster.m_boundedBox;
		m_pointsArray = in_cluster.m_pointsArray;	// 이렇게 대입만 해도 Deep Copy가 발생함.

		this->m_colorInLuv = in_cluster.m_colorInLuv;
		this->m_colorInHSV = in_cluster.m_colorInHSV;
		this->m_boundedBox = cv::Rect(in_rect.x, in_rect.y, in_rect.width, in_rect.height);
		this->m_label = in_cluster.m_label;
		return *this;
	}

	// 복사생성자
	Cluster(const Cluster& in_cluster)
	{
		m_pointsArray = in_cluster.m_pointsArray;
		this->m_colorInLuv = in_cluster.m_colorInLuv;
		this->m_colorInHSV = in_cluster.m_colorInHSV;
		this->m_boundedBox = in_cluster.m_boundedBox;
		this->m_label = in_cluster.m_label;
	}

	// 기본생성자
	Cluster() :m_pointsArray(0), m_label(-1)
	{

	}

public:
	// in_cluster를 병합한다.
	void Consume(const Cluster& in_cluster)
	{
		m_pointsArray.insert(m_pointsArray.end(), in_cluster.m_pointsArray.begin(), in_cluster.m_pointsArray.end());
		this->m_boundedBox = this->m_boundedBox | in_cluster.m_boundedBox;
		this->m_colorInLuv = (in_cluster.m_colorInLuv + this->m_colorInLuv) / 2;
		this->m_colorInHSV = (in_cluster.m_colorInHSV + this->m_colorInHSV) / 2;
	}
};
