#pragma once
#include <opencv2\opencv.hpp>
struct Point_DBSCAN
{
	Point_DBSCAN()
	{
		m_id = 0;
		m_point.x = 0;
		m_point.y = 0;
		m_bVisited = false;
		m_bJoined = false;
	}

	Point_DBSCAN(int x, int y, int id):Point_DBSCAN()
	{
		m_id = id;
		m_point.x = x;
		m_point.y = y;
	}

	Point_DBSCAN(cv::Point in_givenPoint, int id)
	{
		m_point = in_givenPoint;
		m_id = id;
		m_bVisited = false;
		m_bJoined = false;
	}
public:
	unsigned int m_id;
	cv::Point m_point;
	bool m_bVisited;
	bool m_bJoined;
};

struct Cluster_DBSCAN
{
public:
	void AddPoint(Point_DBSCAN* in_givenPoint) 
	{
		m_points.push_back(in_givenPoint);
	}
	const std::list<Point_DBSCAN*>& GetPointsList()
	{
		return m_points;
	}
	int GetSize() const
	{
		return m_points.size();
	}
	void GetVectorVersion(std::vector<cv::Point2f> &out_clusterVectorVersion) const 
	{
		out_clusterVectorVersion.clear();
		out_clusterVectorVersion.resize(m_points.size());
		int idx = 0;
		for (auto& eachPoint : m_points)
		{
			out_clusterVectorVersion[idx] = eachPoint->m_point;
			idx++;
		}
	}

private:
	std::list<Point_DBSCAN*> m_points;
};

