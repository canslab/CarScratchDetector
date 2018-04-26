#pragma once
#include <opencv2\opencv.hpp>
#include <unordered_map>

// ������ �����ϴ� Cluster Ŭ����
class Cluster
{
private:
	// Ŭ������ ��ǥ Luv�÷� ��
	cv::Point3i m_colorInLuv;
	// Ŭ������ ��ǥ HSV �÷� ��
	cv::Point3i m_colorInHSV;
	// Ŭ������ ���ο� �����ϴ� ��� ��(x, y��ǥ)���� �����Ϳ� ����. �� ���Ҵ� ��ǥ ���̹Ƿ� 2ä��
	std::vector<cv::Point> m_pointsArray;
	// Ŭ�����ͳ����� ���̺� ��
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

	// Ŭ�����͸� �ٿ���ϴ� �簢��
	cv::Rect m_boundedBox;

	// Ŭ�����Ͱ� �����ϰ� �ִ� ���� ������ ��ȯ				
	inline int GetTotalPoints() const
	{
		return m_pointsArray.size();
	}

	// ������ �����ϰ� �ִ� �� ���͸� ��ȯ��.
	inline const std::vector<cv::Point>& GetPointsArray() const
	{
		return m_pointsArray;
	}

	int GetLabel() const
	{
		return m_label;
	}

	// Ŭ������ ��ǥ ������ ����.
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

		// Luv ������ HSV�ε� �����صд�.
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
		// �ٿ�� �ڽ� ������Ʈ
		m_boundedBox = cv::boundingRect(m_pointsArray);
	}
	bool AddPoint(const cv::Point &in_point)
	{
		// ������.. �����..
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
		m_pointsArray = in_cluster.m_pointsArray;	// �̷��� ���Ը� �ص� Deep Copy�� �߻���.

		this->m_colorInLuv = in_cluster.m_colorInLuv;
		this->m_colorInHSV = in_cluster.m_colorInHSV;
		this->m_boundedBox = cv::Rect(in_rect.x, in_rect.y, in_rect.width, in_rect.height);
		this->m_label = in_cluster.m_label;
		return *this;
	}

	// ���������
	Cluster(const Cluster& in_cluster)
	{
		m_pointsArray = in_cluster.m_pointsArray;
		this->m_colorInLuv = in_cluster.m_colorInLuv;
		this->m_colorInHSV = in_cluster.m_colorInHSV;
		this->m_boundedBox = in_cluster.m_boundedBox;
		this->m_label = in_cluster.m_label;
	}

	// �⺻������
	Cluster() :m_pointsArray(0), m_label(-1)
	{

	}

public:
	// in_cluster�� �����Ѵ�.
	void Consume(const Cluster& in_cluster)
	{
		m_pointsArray.insert(m_pointsArray.end(), in_cluster.m_pointsArray.begin(), in_cluster.m_pointsArray.end());
		this->m_boundedBox = this->m_boundedBox | in_cluster.m_boundedBox;
		this->m_colorInLuv = (in_cluster.m_colorInLuv + this->m_colorInLuv) / 2;
		this->m_colorInHSV = (in_cluster.m_colorInHSV + this->m_colorInHSV) / 2;
	}
};
