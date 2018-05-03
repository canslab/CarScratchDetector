#include "DBSCAN.h"
#include <list>
#include <unordered_set>

#pragma optimize("gpsy", off)
static std::unordered_set<Point_DBSCAN*> GetEpsilonNeighborhood(const Point_DBSCAN* in_givenPoint, const std::list<Point_DBSCAN*>& in_wholePoints, const int in_epsilon)
{
	static double gDistanceTable[300][300] = { 0 };
	std::unordered_set<Point_DBSCAN*> neighboringPoints;

	for (const auto& eachPoint : in_wholePoints)
	{
		double distance = 0.0;
		if (in_givenPoint->m_point == eachPoint->m_point)
		{
			continue;
		}

	/*	if (gDistanceTable[in_givenPoint->m_id][eachPoint->m_id] == 0)
		{
			gDistanceTable[in_givenPoint->m_id][eachPoint->m_id] = gDistanceTable[eachPoint->m_id][in_givenPoint->m_id] = cv::norm(eachPoint->m_point - in_givenPoint->m_point);
		}
*/
	
		if (cv::norm(eachPoint->m_point - in_givenPoint->m_point) <= in_epsilon)
		{
			neighboringPoints.insert(eachPoint);
		}
	}
	return neighboringPoints;
}
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
void GeneratePointsForDBSCAN(const std::vector<cv::Point2f>& corners, std::list<Point_DBSCAN*>& out_pointsForDBSCAN)
{
	out_pointsForDBSCAN.clear();

	for (int i = 0; i < corners.size(); ++i)
	{
		out_pointsForDBSCAN.push_back(new Point_DBSCAN(corners[i], i));
	}
}
#pragma optimize("gpsy", on)

void PerformDBSCAN(const std::list<Point_DBSCAN*>& in_wholePoints, const double in_epsilon, const int in_minPoints, std::vector<Cluster_DBSCAN>& out_clusterList)
{
	// Marker for in_points
	std::list<Point_DBSCAN*> unvisitedPoints = in_wholePoints;

	// clean up before clustering
	out_clusterList.clear();

	while (unvisitedPoints.size() != 0)
	{
		Point_DBSCAN* chosenPoint = unvisitedPoints.front();
		chosenPoint->m_bVisited = true;
		unvisitedPoints.pop_front();

		auto neighboringPoints = GetEpsilonNeighborhood(chosenPoint, in_wholePoints, in_epsilon);
		if (neighboringPoints.size() >= in_minPoints - 1)
		{
			out_clusterList.push_back(Cluster_DBSCAN());
			Cluster_DBSCAN& newCluster = out_clusterList.back();
			newCluster.AddPoint(chosenPoint);

			std::unordered_set<Point_DBSCAN*> *nowTest = new std::unordered_set<Point_DBSCAN*>(neighboringPoints);
			std::unordered_set<Point_DBSCAN*> *candidate = new std::unordered_set<Point_DBSCAN*>();
			while (nowTest->size() != 0)
			{
				for (Point_DBSCAN* eachPoint : *nowTest) //    auto& eachPoint : *nowTest)
				{
					if (eachPoint->m_bVisited == false)
					{
						eachPoint->m_bVisited = true;

						// unvisited point에서 eachPoint를 찾아야함...그래서 제거! (방문했으니까)
						auto t = std::find(unvisitedPoints.begin(), unvisitedPoints.end(), eachPoint);
						unvisitedPoints.erase(t);

						auto neighborhoodOfEachPoint = GetEpsilonNeighborhood(eachPoint, in_wholePoints, in_epsilon);

						if (neighborhoodOfEachPoint.size() >= in_minPoints - 1)
						{
							for (Point_DBSCAN* eachPoint : neighborhoodOfEachPoint)
							{
								bool bEachPointVisited = (std::find(unvisitedPoints.begin(), unvisitedPoints.end(), eachPoint) == unvisitedPoints.end());
								
								if (nowTest->count(eachPoint) == 0 && bEachPointVisited == false)
								{
									candidate->insert(eachPoint);
								}
							}
						}					
					}
					if (eachPoint->m_bJoined == false)
					{
						newCluster.AddPoint(eachPoint);
					}
				}

				nowTest->clear();
				std::unordered_set<Point_DBSCAN*>* k = nowTest;
				nowTest = candidate;
				candidate = k;
			}

			delete nowTest;
			delete candidate;
		}
	}
}
#pragma optimize("gpsy", off)