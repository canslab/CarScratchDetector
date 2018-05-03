#include "AlgorithmCode\MeanShiftCluster.h"
#include "ClusterForDBSCAN.h"
#include <list>

// input = std::vector<cv::Point2f> corners
// output = std::list<Point_DBSCAN*> out_points

void GeneratePointsForDBSCAN(const std::vector<cv::Point2f> &corners, std::list<Point_DBSCAN*> &out_pointsForDBSCAN);
void PerformDBSCAN(const std::list<Point_DBSCAN*>& in_wholePoints, const double in_epsilon, const int in_minPoints, std::vector<Cluster_DBSCAN>& out_clusterList);