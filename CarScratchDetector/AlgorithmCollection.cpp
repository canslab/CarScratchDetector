#include "stdafx.h"
#include "AlgorithmCollection.h"
#include "Cluster.h"
#include "UtilityCode\Timer.h"

#define BEZEL_LABEL	INT_MAX
#define NOT_CLUSTER_LABEL -1

void ExpandRectInAnyFourDirections(cv::Size in_limitBox, cv::Rect& in_rect, int offsetX, int offsetY, int diffWidth, int diffHeight)
{
	if ((in_rect.x + offsetX) >= 0)
	{
		in_rect.x += offsetX;
	}
	else
	{
		in_rect.x = 0;
	}
	if ((in_rect.y + offsetY) >= 0)
	{
		in_rect.y += offsetY;
	}
	else
	{
		in_rect.y = 0;
	}

	if ((in_rect.x + in_rect.width + diffWidth) <= in_limitBox.width)
	{
		in_rect.width += diffWidth;
	}
	else
	{
		in_rect.width = in_limitBox.width - in_rect.x;
	}
	if ((in_rect.y + in_rect.height + diffHeight) <= in_limitBox.height)
	{
		in_rect.height += diffHeight;
	}
	else
	{
		in_rect.height = in_limitBox.height - in_rect.y;
	}
}

void GetLabelsOfAdjacentClusters(const cv::Mat & in_labelMap, const Cluster & in_centerCluster, const cv::Rect& in_ROI, std::set<int>& out_labels)
{
	const cv::Rect& kROI = in_ROI;
	const int kRowStartIndex = kROI.y; // 0 based value임을 잊지말자.
	const int kRowEndIndex = kRowStartIndex + kROI.height - 1;
	const int kColStartIndex = kROI.x;
	const int kColEndIndex = kColStartIndex + kROI.width - 1;

	// 4 방향 (위, 아래, 왼쪽, 오른쪽)에서 범위를 좁혀와야함.
	// 왼쪽 (행이동)
	for (int rowIndex = kRowStartIndex; rowIndex <= kRowEndIndex; ++rowIndex)
	{
		for (int colIndex = kColStartIndex; colIndex <= kColEndIndex; ++colIndex)
		{
			auto currentLabel = in_labelMap.at<int>(rowIndex, colIndex);

			if (in_centerCluster.DoesContain(currentLabel) == false)
			{
				if (out_labels.count(currentLabel) == 0)
				{
					// currentLabel이 mainCluster에 속하지 않는다는것이므로,
					// 인접한 영역의 레이블이라 할 수 있다.
					out_labels.insert(currentLabel);
				}
			}
		}
	}
}
double GetHSVBhattaCoefficient(const cv::Mat& in_img, const Cluster &in_cluster1, const Cluster &in_cluster2, int channelNumber, int in_nBin)
{
	auto& pointArray1 = in_cluster1.GetPointsArray();
	auto& pointArray2 = in_cluster2.GetPointsArray();

	cv::Mat cluster1_Mat(1, pointArray1.size(), CV_8UC3);
	cv::Mat cluster2_Mat(1, pointArray2.size(), CV_8UC3);

	// cluster1, cluster2 내부의 점들의 픽셀 색상값들을 행벡터로 옮김.
	int nColCount = 0;
	for (auto& eachPoint : pointArray1)
	{
		cluster1_Mat.at<cv::Vec3b>(0, nColCount) = in_img.at<cv::Vec3b>(eachPoint);
		nColCount++;
	}

	nColCount = 0;
	for (auto& eachPoint : pointArray2)
	{
		cluster2_Mat.at<cv::Vec3b>(0, nColCount) = in_img.at<cv::Vec3b>(eachPoint);
		nColCount++;
	}

	// 이제 만들어진 두 개의 이미지(Mat)을 Histogram Comparison한다.
	int nDims = 1;
	int histSize[] = { in_nBin };

	int kMaxRange = (channelNumber == 0) ? 180 : 256;
	float lRanges[] = { 0, kMaxRange };

	const float* rangeArray[] = { lRanges }; // Luv 모두 [0,256) 이므로
	int channels[] = { channelNumber }; // L->u->v 순서로 색상값이 배치되어 있으므로

	cv::Mat cluster1_Mat_Hist;
	cv::Mat cluster2_Mat_Hist;

	cv::calcHist(&cluster1_Mat, 1, channels, cv::Mat(), cluster1_Mat_Hist, nDims, histSize, rangeArray, true, false);
	cv::calcHist(&cluster2_Mat, 1, channels, cv::Mat(), cluster2_Mat_Hist, nDims, histSize, rangeArray, true, false);
	cv::normalize(cluster1_Mat_Hist, cluster1_Mat_Hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(cluster2_Mat_Hist, cluster2_Mat_Hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	
	auto coefficient = cv::compareHist(cluster1_Mat_Hist, cluster2_Mat_Hist, CV_COMP_BHATTACHARYYA);
	return coefficient;
}
void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, Cluster>& out_clusters)
{
	//FloodFillPostprocess_Cluster(in_luvWholeImage, out_labelMap, out_clusters, cv::Scalar::all(1));
	int kInputImageWidth = in_luvWholeImage.cols;
	int kInputImageHeight = in_luvWholeImage.rows;

	//cv::Mat replacedMat(in_excludeRect.height, in_excludeRect.width, CV_8UC1, cv::Scalar::all(1));
	cv::Mat afterFloodFillMask(kInputImageHeight + 2, kInputImageWidth + 2, CV_8UC1, cv::Scalar::all(0));
	cv::Mat beforeFloodFillMask(kInputImageHeight + 2, kInputImageWidth + 2, CV_8UC1, cv::Scalar::all(0));
	cv::Mat subtractedMatrix(kInputImageHeight + 2, kInputImageWidth + 2, CV_8UC1, cv::Scalar(0));
	cv::Mat findIndexColumnVector;

	// label map 할당
	cv::Mat labelMap(kInputImageHeight, kInputImageWidth, CV_32SC1, cv::Scalar::all(NOT_CLUSTER_LABEL));
	out_labelMap = labelMap;

	// Boundary를 고려하기 위해, ROI 지정
	cv::Rect roi;
	roi.x = roi.y = 1;
	roi.width = in_luvWholeImage.cols;
	roi.height = in_luvWholeImage.rows;
	int clusterIndex = 0;

	for (int y = in_ROI.y; y < (in_ROI.y + in_ROI.height); ++y)
	{
		for (int x = in_ROI.x; x < (in_ROI.x + in_ROI.width); x++)
		{
			// 여기서 x, y가 지정된 테투리 이내의 지점이면 클러스터링 자체를 하지 않는다.
			if (afterFloodFillMask.at<uchar>(y + 1, x + 1) == 0)
			{
				cv::Rect boundedBox;
				auto &replacedVector = in_luvWholeImage.at<cv::Vec3b>(y, x);
								
				// 동일한 색으로 채우기
				cv::floodFill(in_luvWholeImage, afterFloodFillMask, cv::Point(x, y), replacedVector, &boundedBox, cv::Scalar::all(1), cv::Scalar::all(1));

				// 마스크 행렬의 뺄셈을 통해, 이번 루프에서 어떤 픽셀들이 동일한 색으로 채워졌는가를 구한다.
				// 바로 그 동일한 녀석들이 한 클러스터를 이룬다.
				cv::subtract(afterFloodFillMask, beforeFloodFillMask, subtractedMatrix);
				afterFloodFillMask.copyTo(beforeFloodFillMask);

				cv::Mat roiMat = subtractedMatrix(roi);

				// 클러스터를 이루는 점들의 위치를 findIndexColumnVector가 가지고 있음
				cv::findNonZero(roiMat, findIndexColumnVector);

				int nPointsInThisCluster = findIndexColumnVector.rows;
				// if # of elements in a cluster is less than a certian number (0.5% of total number of pixels), -1 is assigned to that pixel
				if (nPointsInThisCluster > in_thresholdToBeCluster)
				{
					// label map 만들기
					for (int i = 0; i < findIndexColumnVector.rows; ++i)
					{
						auto& pt = findIndexColumnVector.at<cv::Point>(i, 0);
						out_labelMap.at<int>(pt) = clusterIndex;
					}

					// 클러스터 컨테이너(unordered_map)에 Cluster를 등록
					out_clusters[clusterIndex] = Cluster();
					Cluster& eachCluster = out_clusters[clusterIndex];

					// cluster에 본인이 몇 번째 레이블인지 저장.
					eachCluster.RegisterLabel(clusterIndex);

					// Copy all points to eachcluster
					cv::Point* ptData = (cv::Point*)findIndexColumnVector.data;
					eachCluster.AddPointsFromArray(ptData, findIndexColumnVector.rows);

					// assign bound box & color value to eachCluster
					eachCluster.SetBoundBox(boundedBox);
					eachCluster.SetColorUsingLuvVector(replacedVector);
					clusterIndex++;
				}
			}
		}
	}

}
void FindSeedClusterInROI(const cv::Mat & in_labelMap, const std::set<int> &in_backgroundIndices, const cv::Rect & in_ROI, int& out_seedLabel)
{
	int startRowIndex = in_ROI.y;
	int endRowIndex = in_ROI.y + in_ROI.height;
	int startColIndex = in_ROI.x;
	int endColIndex = in_ROI.x + in_ROI.width;
	int totalPoints = 0;

	// Key는 클러스터 번호(레이블), Value는 레이블이 in_ROI에서 출현한 횟수
	// in_ROI내부에서 등장하는 레이블과 그 레이블의 빈도수를 기록하는 딕셔너리
	std::unordered_map<int, int> labelToOccurenceMap;

	for (int rowIndex = startRowIndex; rowIndex < endRowIndex; ++rowIndex)
	{
		for (int colIndex = startColIndex; colIndex < endColIndex; ++colIndex)
		{
			auto currentLabel = in_labelMap.at<int>(rowIndex, colIndex);

			// 미세한 픽셀은 씨드의 자격이 없다.
			if (currentLabel == NOT_CLUSTER_LABEL)
			{
				continue;
			}

			// 기존에 관리하던 레이블이 아니다 => 새로 등록
			if (labelToOccurenceMap.count(currentLabel) == 0)
			{
				labelToOccurenceMap.insert(std::make_pair(currentLabel, 0));
			}

			labelToOccurenceMap[currentLabel]++;
			totalPoints++;
		}
	}
	std::pair<int, int> maxPair;

	SearchMapForMaxPair(labelToOccurenceMap, 0, maxPair);

	// 백그라운드가 Seed Segment라고 나왔다면, 정말 ROI에는 백그라운드 클러스터밖에 없는지 확인해야 한다. (70%이상 차지하는지 봐야한다는 의미)
	// 만일, 백그라운드클러스가 ROI에서 과반영역을 차지하지 않고 다른 레이블 영역들도 있다면, 그 다음으로 규모가 큰 세그멘트가
	// Seed Segment가 될것이다.
	if (in_backgroundIndices.count(maxPair.first) == 1 && (((double)(maxPair.second / totalPoints) < 0.7)))
	{
		// 그러므로 background Label은 labelToOccurenceMap에서 지워버리고, 다시 가장 큰 Seed Segment를 찾는다.
		labelToOccurenceMap.erase(maxPair.first);
		SearchMapForMaxPair(labelToOccurenceMap, 0, maxPair);
	}

	out_seedLabel = maxPair.first;
}
void GetBackgroundClusterIndices(const cv::Size & in_originalImageSize, const cv::Mat& in_labelMap, int in_marginLegnth, std::set<int>& out_backgroundIndiciesSet)
{
	for (int rowIndex = 0; rowIndex < in_originalImageSize.height; ++rowIndex)
	{
		// 윗 덩어리, 아랫덩어리
		if (rowIndex < in_marginLegnth || rowIndex >= (in_originalImageSize.height - in_marginLegnth))
		{
			// 전체탐색
			for (int colIndex = 0; colIndex < in_originalImageSize.width; ++colIndex)
			{
				out_backgroundIndiciesSet.insert(in_labelMap.at<int>(rowIndex, colIndex));
			}
		}
		else
		{
			// 가운데 파인 곳
			for (int colIndex = 0; colIndex < in_marginLegnth; ++colIndex)
			{
				out_backgroundIndiciesSet.insert(in_labelMap.at<int>(rowIndex, colIndex));
			}

			for (int colIndex = (in_originalImageSize.width - in_marginLegnth); colIndex < in_originalImageSize.width; ++colIndex)
			{
				out_backgroundIndiciesSet.insert(in_labelMap.at<int>(rowIndex, colIndex));
			}
		}
	}
	out_backgroundIndiciesSet.erase(NOT_CLUSTER_LABEL);
}
cv::Mat GetAlphaMap(const cv::Mat& labelMap, const cv::Rect& ROI, const Cluster &mainCluster)
{
	cv::Mat alphaMap = cv::Mat::zeros(labelMap.size(), CV_8UC1);
	// Find contour that has minimum area.
	std::vector<cv::Point> leftLine;
	std::vector<cv::Point> rightLine;
	std::vector<cv::Point> topLine;
	std::vector<cv::Point> botLine;

	for (int y = ROI.y; y < ROI.y + ROI.height; y++)
	{
		leftLine.push_back(cv::Point(ROI.x, y));
		rightLine.push_back(cv::Point(ROI.x + ROI.width - 1, y));
	}

	for (int x = ROI.x; x < ROI.x + ROI.width; x++)
	{
		topLine.push_back(cv::Point(x, ROI.y));
		botLine.push_back(cv::Point(x, ROI.y + ROI.height - 1));
	}

	for (int i = 0; i < leftLine.size(); i++)
	{
		cv::Point curLeftPt = leftLine[i];
		cv::Point curRightPt = rightLine[i];
		bool isLeftConverged = false;
		bool isRightConverged = false;
		while (1)
		{
			if (curLeftPt.x - curRightPt.x < 2 && curLeftPt.x - curRightPt.x > -2)
			{
				// Not converged. Odd situation
				//if(!isLeftConverged)
				curLeftPt = cv::Point(-1, -1);
				//if(!isRightConverged)
				curRightPt = cv::Point(-1, -1);
				break;
			}

			if (!isLeftConverged && mainCluster.DoesContain(labelMap.at<int>(curLeftPt)))
			{
				isLeftConverged = true;
			}
			else if (!isLeftConverged)
			{
				curLeftPt.x += 1;
			}

			if (!isRightConverged &&
				mainCluster.DoesContain(labelMap.at<int>(curRightPt)))
			{
				isRightConverged = true;
			}
			else if (!isRightConverged)
			{
				curRightPt.x -= 1;
			}

			if (isRightConverged && isLeftConverged)
				break;
		}
		leftLine[i] = curLeftPt;
		rightLine[i] = curRightPt;
	}

	for (int i = 0; i < topLine.size(); i++)
	{
		cv::Point curTopPt = topLine[i];
		cv::Point curBotPt = botLine[i];
		bool isTopConverged = false;
		bool isBotConverged = false;
		while (1)
		{
			if (curTopPt.y - curBotPt.y < 2 && curTopPt.y - curBotPt.y > -2)
			{
				// Not converged. Odd situation
				curTopPt = cv::Point(-1, -1);
				curBotPt = cv::Point(-1, -1);
				break;
			}

			if (!isTopConverged &&
				mainCluster.DoesContain(labelMap.at<int>(curTopPt)))
			{
				isTopConverged = true;
			}
			else if (!isTopConverged)
			{
				curTopPt.y += 1;
			}

			if (!isBotConverged &&
				mainCluster.DoesContain(labelMap.at<int>(curBotPt)))
			{
				isBotConverged = true;
			}
			else if (!isBotConverged)
			{
				curBotPt.y -= 1;
			}

			if (isTopConverged && isBotConverged)
				break;
		}
		topLine[i] = curTopPt;
		botLine[i] = curBotPt;
	}

	cv::Mat debug = cv::Mat::zeros(alphaMap.size(), CV_8UC3);
	for (int i = 0; i < leftLine.size(); i++)
	{
		if (leftLine[i].x == -1)
			continue;
		//debug.at<cv::Vec3b>(leftLine[i]) = cv::Vec3b(255, 255, 255);
		alphaMap.at<uchar>(leftLine[i]) = 255;
		for (int x = leftLine[i].x; x < rightLine[i].x; x++)
			//debug.at<cv::Vec3b>(leftLine[i].y, x) = cv::Vec3b(255, 255, 255);
			alphaMap.at<uchar>(leftLine[i].y, x) = 255;
	}
	for (int i = 0; i < rightLine.size(); i++)
	{
		if (rightLine[i].x == -1)
			continue;
		//debug.at<cv::Vec3b>(rightLine[i]) = cv::Vec3b(0, 0, 255);
		alphaMap.at<uchar>(rightLine[i]) = 255;
	}
	for (int i = 0; i < topLine.size(); i++)
	{
		if (topLine[i].x == -1)
			continue;
		//debug.at<cv::Vec3b>(topLine[i]) = cv::Vec3b(255, 0, 0);
		alphaMap.at<uchar>(topLine[i]) = 255;
		for (int y = topLine[i].y; y < botLine[i].y; y++)
			//debug.at<cv::Vec3b>(y, topLine[i].x) = cv::Vec3b(255, 255, 255);
			alphaMap.at<uchar>(y, topLine[i].x) = 255;

	}
	for (int i = 0; i < botLine.size(); i++)
	{
		if (botLine[i].x == -1)
			continue;
		//debug.at<cv::Vec3b>(botLine[i]) = cv::Vec3b(0, 255, 0);
		alphaMap.at<uchar>(botLine[i]) = 255;
	}
	cv::rectangle(debug, ROI, cv::Scalar(255, 255, 255), 1, CV_AA);

	return alphaMap;
}

bool IsClusterWrappedByCertainCluster(const Cluster &in_cluster, const cv::Mat & in_labelMap, int in_rangeToCover, float in_ratio, int& out_labelOfWrapperCluster)
{
	bool bWrapped = false;

	std::vector<cv::Point> allEdgePoints;
	std::unordered_map<int, int> labelsFrequencyOfAllEdgePoints;
	std::pair<int, int> maxPair;

	// Cluster의 Outer Contour를 이루고 있는 모든 점들을 구한다.
	FindAllOuterPointsOfCluster(in_labelMap.size(), in_cluster, allEdgePoints);
	const int kThreshold = (int)(allEdgePoints.size() * in_ratio);

	// Step 1. 
	// Search all edge points for the adjacent clusters
	// Edge상에 존재하는 모든 점들에 대하여 생각한다.
	for (const cv::Point& eachPoint : allEdgePoints)
	{
		cv::Point targetPoint;
		std::unordered_map<int, int> frequencyOfLabels; // Key = Label, Value = # of occurence

		// 한 Point를 기준으로 [in_rangeToCover x in_rangeToCover]의 영역을 생각한다.
		for (int rowIndex = -in_rangeToCover; rowIndex <= in_rangeToCover; ++rowIndex)
		{
			targetPoint.y = eachPoint.y + rowIndex;

			for (int colIndex = -in_rangeToCover; colIndex <= in_rangeToCover; ++colIndex)
			{
				targetPoint.x = eachPoint.x + colIndex;

				int adjacentLabel = in_labelMap.at<int>(targetPoint);

				// if adjacentLabel is certainly outer part of the cluster (in_cluster)
				if (!in_cluster.DoesContain(adjacentLabel))
				{
					if (frequencyOfLabels.count(adjacentLabel) == 0)
					{
						frequencyOfLabels[adjacentLabel] = 0;
					}
					frequencyOfLabels[adjacentLabel]++;
				}
			}
		}

		// eachPoint를 둘러싸는 클러스터중 가장 비중이 큰 녀석을 구해 maxPair에 저장한다. 
		// maxPair의 key값은 레이블이고 value는 등장 횟수이다.
		SearchMapForMaxPair(frequencyOfLabels, 0, maxPair);

		if (labelsFrequencyOfAllEdgePoints.count(maxPair.first) == 0)
		{
			labelsFrequencyOfAllEdgePoints[maxPair.first] = 0;
		}

		// 현재 포인트를 감싸고 있는 레이블을 등록하고 빈도를 1증가시킨다.
		labelsFrequencyOfAllEdgePoints[maxPair.first]++;
	}

	// Step 2.
	// labelsFrequencyOfAllEdgePoints는 edge의 모든 점들이 생각하는 인접 레이블과 그 레이블의 발생빈도를 담고 있다.
	// Now adjaceLabelOfAllEdgePoints contains adjacent labels of all edge points
	// 만일, 압도적으로 (kThreshold)이상 차지하는 label이 있다면, 그 레이블은 감싸는 레이블이라고 생각한다.
	bWrapped = SearchMapForMaxPair(labelsFrequencyOfAllEdgePoints, kThreshold, maxPair);

	// bWrapped가 true라면, maxPair.first에는 감싸고 있는 클러스터의 레이블이 담긴다.
	out_labelOfWrapperCluster = maxPair.first;
	return bWrapped;
}

void GetAllAdjacentLabelsAndTheirFrequency(const Cluster& in_cluster, const cv::Mat& in_labelMap, int in_rangeToCover, std::unordered_map<int, int> &out_labelAndItsFrequency, std::vector<cv::Point>& out_minusPoints)
{
	std::vector<cv::Point> allEdgePoints;
	std::pair<int, int> maxPair;

	// Cluster의 Outer Contour를 이루고 있는 모든 점들을 구한다.
	FindAllOuterPointsOfCluster(in_labelMap.size(), in_cluster, allEdgePoints);

	// Step 1. 
	// Search all edge points for the adjacent clusters
	// Edge상에 존재하는 모든 점들에 대하여 생각한다.
	for (const cv::Point& eachPoint : allEdgePoints)
	{
		cv::Point targetPoint;
		std::unordered_map<int, int> frequencyOfLabels; // Key = Label, Value = # of occurence

											  // 한 Point를 기준으로 [in_rangeToCover x in_rangeToCover]의 영역을 생각한다.
		for (int rowIndex = -in_rangeToCover; rowIndex <= in_rangeToCover; ++rowIndex)
		{
			targetPoint.y = eachPoint.y + rowIndex;

			for (int colIndex = -in_rangeToCover; colIndex <= in_rangeToCover; ++colIndex)
			{
				targetPoint.x = eachPoint.x + colIndex;

				int adjacentLabel = in_labelMap.at<int>(targetPoint);

				// if adjacentLabel is certainly outer part of the cluster (in_cluster)
				if (!in_cluster.DoesContain(adjacentLabel))
				{
					if (frequencyOfLabels.count(adjacentLabel) == 0)
					{
						frequencyOfLabels[adjacentLabel] = 0;
					}
					frequencyOfLabels[adjacentLabel]++;
				}

				if (adjacentLabel == NOT_CLUSTER_LABEL && std::find(out_minusPoints.begin(), out_minusPoints.end(), targetPoint) == out_minusPoints.end())
				{
					out_minusPoints.push_back(targetPoint);
				}
			}
		}

		// eachPoint를 둘러싸는 클러스터중 가장 비중이 큰 녀석을 구해 maxPair에 저장한다. 
		// maxPair의 key값은 레이블이고 value는 등장 횟수이다.
		SearchMapForMaxPair(frequencyOfLabels, 0, maxPair);

		// 기존에 없다면 등록
		if (out_labelAndItsFrequency.count(maxPair.first) == 0)
		{
			out_labelAndItsFrequency[maxPair.first] = 0;
		}

		// 현재 포인트를 감싸고 있는 레이블의 빈도를 1증가시킨다.
		out_labelAndItsFrequency[maxPair.first]++;
	}
}

/**************************************************/
/****       Utility Functions' Implementation *****/
/**************************************************/

void DrawContoursOfClusters(cv::Mat & in_targetImage, const std::unordered_map <int, Cluster>& in_clusters, cv::Scalar in_color)
{
	for (auto& eachCluster : in_clusters)
	{
		DrawOuterContourOfCluster(in_targetImage, eachCluster.second, in_color);
	}
}

void DrawOuterContourOfCluster(cv::Mat & in_targetImage, const Cluster & in_cluster, cv::Scalar in_color)
{
	const auto& points = in_cluster.GetPointsArray();
	const auto nTotalPoints = points.size();
	cv::Mat tempGrayImage(in_targetImage.rows, in_targetImage.cols, CV_8UC1, cv::Scalar(0));
	std::vector<std::vector<cv::Point>> contour;

	for (int i = 0; i < nTotalPoints; ++i)
	{
		auto& point = points[i];
		tempGrayImage.at<unsigned char>(point) = 255;
	}
	cv::findContours(tempGrayImage, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Even though there is an only one cluster, it can have more than a 1 contour line. So draw all contours by giving -1 as a parameter
	cv::drawContours(in_targetImage, contour, -1, in_color, 2);
}

void GetOriginalHSVColorFromHalfedLuv(const cv::Point3i& in_luv, double in_factor, cv::Point3i& out_hsv)
{
	cv::Mat tempMat(1, 1, CV_8UC3);

	// 2.5로 나눠줬으니 2.5로 원복
	tempMat.at<cv::Vec3b>(0, 0)[0] = in_luv.x * in_factor;
	tempMat.at<cv::Vec3b>(0, 0)[1] = in_luv.y;
	tempMat.at<cv::Vec3b>(0, 0)[2] = in_luv.z;

	cv::cvtColor(tempMat, tempMat, CV_Luv2BGR);
	cv::cvtColor(tempMat, tempMat, CV_BGR2HSV);

	out_hsv.x = tempMat.at<cv::Vec3b>(0, 0)[0];
	out_hsv.y = tempMat.at<cv::Vec3b>(0, 0)[1];
	out_hsv.z = tempMat.at<cv::Vec3b>(0, 0)[2];
}

bool IsOutOfRange(const cv::Mat &in_originalImage, const cv::Rect &in_rect)
{
	cv::Rect originalRect(0, 0, in_originalImage.cols, in_originalImage.rows);
	cv::Rect intersection = in_rect & originalRect;

	if (intersection.area() == in_rect.area())
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool SearchMapForMaxPair(const std::unordered_map<int, int>& in_map, const int kMinimumValue, std::pair<int, int> &out_keyValuePair)
{
	int index = 0;
	int currentMaxLabel = -1;
	int currentMaxCount = -1;
	out_keyValuePair.first = 0;
	out_keyValuePair.second = 0;

	// now frequencyOfLabels contains all labels of pixels, other than labels of in_cluster, closed to 'eachPoint'
	for (auto eachLabel : in_map)
	{
		if (currentMaxCount < eachLabel.second)
		{
			currentMaxLabel = eachLabel.first;
			currentMaxCount = eachLabel.second;
		}
	}

	if (currentMaxCount >= kMinimumValue)
	{
		out_keyValuePair.first = currentMaxLabel;
		out_keyValuePair.second = currentMaxCount;
		return true;
	}
	else
	{
		return false;
	}
}

void FindAllOuterPointsOfCluster(const cv::Size& in_frameSize, const Cluster & in_cluster, std::vector<cv::Point> &out_points)
{
	cv::Mat alphaMap;
	std::vector<std::vector<cv::Point>> contours;

	CreateAlphaMapFromCluster(in_frameSize, in_cluster, alphaMap);
	cv::findContours(alphaMap, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// Mostly, contours has only one set of points. 
	// now out_points has all points lying in the edge of the cluster (in_cluster)
	out_points = contours[0];
}

void CreateAlphaMapFromCluster(const cv::Size & in_alphaMapSize, const Cluster & in_cluster, cv::Mat & out_alphaMap)
{
	cv::Mat alphaMap(in_alphaMapSize, CV_8UC1, cv::Scalar(0));
	auto& arrayOfPoints = in_cluster.GetPointsArray();

	for (auto& eachPoint : arrayOfPoints)
	{
		alphaMap.at<uchar>(eachPoint) = 255;
	}

	out_alphaMap = alphaMap;
}

void ProjectClusterIntoMat(const Cluster & in_cluster, cv::Mat & out_mat)
{
	auto pointsArray = in_cluster.GetPointsArray();
	auto colorInLuv = in_cluster.GetLuvColor();

	for (auto& eachPoint : pointsArray)
	{
		auto& pixel = out_mat.at<cv::Vec3b>(eachPoint);
		pixel[0] = colorInLuv.x;
		pixel[1] = colorInLuv.y;
		pixel[2] = colorInLuv.z;
	}
}

void FindBiggestCluster(const std::unordered_map<int, Cluster>& in_clusters, int & out_biggestClusterLabel)
{
	// find Max Cluster among negihbor clusters
	int maxLabel = -10;
	int maxNumber = -10;
	for (auto& eachCluster : in_clusters)
	{
		int currentLabel = eachCluster.first;
		int currentClusterSize = eachCluster.second.GetTotalPoints();
		if (maxNumber <= currentClusterSize)
		{
			maxLabel = currentLabel;
			maxNumber = currentClusterSize;
		}
	}

	out_biggestClusterLabel = maxLabel;
}

void PerformColorMergingTask(const Cluster & in_seedCluster, const std::unordered_map<int, Cluster>& in_clusters,
	const cv::Mat& in_lValueDividedHSVMat, cv::Mat& inout_labelMap,
	const double in_lValudDivider, const std::set<int>& in_backgroundIndices, const cv::Size& in_limitBox, int in_maxTrial, Cluster& out_mergedCluster)
{
	// Seed Cluster를 중심으로 확장 시켜야 한다.
	auto &seedColorInHSV = in_seedCluster.GetHSVColor();
	// 점점 확장해 나갈 ROI (extendedRect)
	auto extendedRect = in_seedCluster.GetBoundedBox();
	out_mergedCluster = in_seedCluster;

	// SeedCluster의 HSV Color를 구해서 hsvColorOfSeedCluster에 저장한다.
	cv::Point3i originalHSVColorOfSeedCluster;
	// hsvColorOfSeedCluster는 Seed Cluster의 색상을 담고있다.
	GetOriginalHSVColorFromHalfedLuv(in_seedCluster.GetLuvColor(), in_lValudDivider, originalHSVColorOfSeedCluster);

	// 병합을 시도했었던 레이블들을 기록.
	std::set<int> triedLabelsSet;

	// Region Merging 작업
	while (true)
	{
		std::set<int> labelsOfAdjacentRegions;
		std::set<int> labelsOfSimilarRegions;

		// mainCluster에 인접한 영역들을 구한다. 
		// 다만, mainCluster의 외부이며 extendedRect내부에서만 인접한 영역을 탐색한다.
		GetLabelsOfAdjacentClusters(inout_labelMap, in_seedCluster, extendedRect, labelsOfAdjacentRegions);
		for (auto eachLabel : labelsOfAdjacentRegions)
		{
			if (triedLabelsSet.count(eachLabel) == 0)
			{
				triedLabelsSet.insert(eachLabel);
			}
			else
			{
				// 이전에 병합을 시도했었던 레이블이므로, Pass
				continue;
			}

			// 백 그라운드 클러스터 OR 소규모 클러스터들은 병합대상이 아니다.
			if (in_backgroundIndices.count(eachLabel) || eachLabel == NOT_CLUSTER_LABEL)
			{
				continue;
			}

			// 검은색끼리 머징하는 방법. seedRegion의 L값이 20보다 작고 인접하다고 생각되는 영역의 L값이 20보다 작으면,
			// 둘다 모두 검은색 영역이므로 합친다.
			if (in_seedCluster.GetLuvColor().x <= 20 && in_clusters.at(eachLabel).GetLuvColor().x <= 20)
			{
				labelsOfSimilarRegions.insert(eachLabel);
				continue;
			}

			cv::Point3i hsvColorOfAdjacentCluster;
			GetOriginalHSVColorFromHalfedLuv(in_clusters.at(eachLabel).GetLuvColor(), in_lValudDivider, hsvColorOfAdjacentCluster);

			// 채도값을 비교(둘다 S가 20보다 작고 V가 100보다 큰 경우)해서 
			// 둘다 어느정도 비슷한 회색계열이면 그림자로 취급하고 합친다.
			if ((originalHSVColorOfSeedCluster.z >= 100 && hsvColorOfAdjacentCluster.z >= 100) && (originalHSVColorOfSeedCluster.y <= 20 && hsvColorOfAdjacentCluster.y <= 20))
			{
				labelsOfSimilarRegions.insert(eachLabel);
				continue;
			}

			// disSimilarity 값이 0에 가까울 수록 유사하다. Bhattacharyya Coefficient를 계산.
			// 원래 Lvalud Divided HSV를 사용하였음..
			auto hueDissimilarity = GetHSVBhattaCoefficient(in_lValueDividedHSVMat, in_clusters.at(eachLabel), in_seedCluster, 0, 30);
			auto satDissimilarity = GetHSVBhattaCoefficient(in_lValueDividedHSVMat, in_clusters.at(eachLabel), in_seedCluster, 1, 16);
			auto valueDissimilarity = GetHSVBhattaCoefficient(in_lValueDividedHSVMat, in_clusters.at(eachLabel), in_seedCluster, 2, 4);

			// 가장 먼저, 색상이 비슷한지를 따져보자.
			// 색상이 비슷하다면, 그다음에는 명도, 채도를 순서로 따져본다.
			if (hueDissimilarity < 0.30)
			{
				// Hue값은 동일하나, 명도가 아주 차이나는 경우가 있다. (흰색, 검은색)
				// 그런 경우에는, 병합의 대상으로 생각하지 않는다.
				// 또한 채도가 일정수준 이상 차이 나버리면, 동일한 색상으로 보지않도록 한다.
				if (valueDissimilarity <= 0.9 && satDissimilarity <= 0.9)
				{
					// 현재 mainCluster와 비슷한 클러스터들의 레이블을 기록해놓는다.
					labelsOfSimilarRegions.insert(eachLabel);
				}
			}
		}

		// 더이상 합칠 클러스터가 없다면, nTrial을 감소시키고 ROI(extendedRect) 또한 증가시킨다.
		if (labelsOfSimilarRegions.size() == 0)
		{
			if (in_maxTrial > 0)
			{
				ExpandRectInAnyFourDirections(in_limitBox, extendedRect, -10, -10, 20, 20);
				in_maxTrial--;
			}
			else
			{
				break;
			}
		}

		// 합칠 인접영역이 있다면 합치고, nTrial은 3으로 다시 원복!
		else
		{
			for (auto similarRegionLabel : labelsOfSimilarRegions)
			{
				// mainCluster에 clusters[similarRegionLabel]을 합병시킨다.
				// 그 결과, mainCluster.m_labels에는 similarRegionLabel이 추가된다.
				out_mergedCluster.Consume(in_clusters.at(similarRegionLabel));
			}
			in_maxTrial = 3;
		}
	}
}

/**************************************************/
/****      For Client Function Implementation *****/
/**************************************************/
bool ExtractObjectFromSourceImage(const cv::Mat & in_srcImage, const AlgorithmParameter& in_parameter, AlgorithmResult& out_result)
{
	Timer totalTimer;
	Timer partialTimer;
	cv::Mat filteredImageMat_luv;												// 원본이미지의 민 쉬프트 필터링된 버젼

	totalTimer.StartStopWatch();

	cv::Mat originalImage;														// 원본이미지
	cv::Mat segmentedImage;														// 민 쉬프트 세그멘테이션 결과 이미지
	cv::Mat hsvOriginalImageMat;												// 원본이미지의 HSV format
	cv::Mat luvOriginalImageMat;												// 원본이미지의 LUV format

	in_srcImage.copyTo(originalImage);											// 입력받은 이미지를 deep copy해옴.

	partialTimer.StartStopWatch();
	cv::cvtColor(originalImage, hsvOriginalImageMat, CV_BGR2HSV);				// 원본이미지 Color Space 변환 (BGR -> Hsv)
	cv::cvtColor(originalImage, luvOriginalImageMat, CV_BGR2Luv);				// 원본이미지 Color Space 변환 (BGR -> Luv)
	out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::BGRToLuvElapsedTime, partialTimer.EndStopWatch());

	const int kOriginalImageWidth = originalImage.cols;							// 원본 사진의 너비
	const int kOriginalImageHeight = originalImage.rows;						// 원본 사진의 높이
	const int kCenterX = kOriginalImageWidth / 2;								// ROI의 중앙 좌표(x)
	const int kCenterY = kOriginalImageHeight / 2;								// ROI의 중앙 좌표(y)
	const int kROI_RectWidth = kOriginalImageWidth / 3;							// ROI 초기 너비
	const int kROI_RectHeight = kOriginalImageHeight / 3;						// ROI 초기 높이
	const cv::Rect kROI_Rect(kCenterX - (kROI_RectWidth / 2), kCenterY - (kROI_RectHeight / 2), kROI_RectWidth, kROI_RectHeight); // ROI 생성

	double sp = in_parameter.GetSpatialBandwidth();								// Mean Shift Filtering을 위한 spatial radius
	double sr = in_parameter.GetColorBandwidth();								// Mean Shift Filtering을 위한 range (color) radius
	const int kBackgroundTolerance = in_parameter.GetBackgroundTolerance();		// kBackgroundTolerance 이내에 들어오는 클러스터들은 배경클러스터
	double lValueDivider = in_parameter.GetLValueDivider();

	std::unordered_map<int, Cluster> clusters; 									// Cluster 모음, Key = Label, Value = Cluster
	std::set<int> backgroundClusterIndexSet;									// 백그라운드 클러스터들의 레이블을 저장하고 있음.
	cv::Mat labelMap;															// 레이블맵
	cv::Mat lValueDividedHSVMat;												// 원본 Luv 이미지에서 L값이 lDivider에 의해 나눠진 상태에서 HSV로 변환된 이미지

	int maxLabel = -1;
	const int kTotalIteration = 5;

	/*******************************************/
	/*******************************************/
	/****** Adaptive Mean Shift Merging ********/
	/*******************************************/
	/*******************************************/
	for (int iteration = 0; iteration < kTotalIteration && sr > 0 && lValueDivider > 0; ++iteration)
	{
		cv::Mat tempLDividedMat;
		luvOriginalImageMat.copyTo(tempLDividedMat);

		partialTimer.StartStopWatch();
		for (int rowIndex = 0; rowIndex < luvOriginalImageMat.rows; ++rowIndex)
		{
			for (int colIndex = 0; colIndex < luvOriginalImageMat.cols; ++colIndex)
			{
				tempLDividedMat.at<cv::Vec3b>(rowIndex, colIndex)[0] = static_cast<uchar>( tempLDividedMat.at<cv::Vec3b>(rowIndex, colIndex)[0] / lValueDivider);
			}
		}
		out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::LDivisionElapsedTime, partialTimer.EndStopWatch());

		// Mean Shift Filtering을 하기 이전에, 밝기 값을 모두 절반으로 해서,
		// 색상값은 비슷한데, 밝기값이 조금 차이나는 경우를 모두 한 클러스터로 만든다. (색상이 비슷하면 묶어버리는 전략)
		partialTimer.StartStopWatch();
		// Mean Shift Filtering 코드 (OpenCV)
		cv::pyrMeanShiftFiltering(tempLDividedMat, filteredImageMat_luv, sp, sr);

		// Clustering을 수행해서 클러스터들을 clusters에 저장한다.
		PerformClustering(filteredImageMat_luv, cv::Rect(0, 0, kOriginalImageWidth, kOriginalImageHeight), (kOriginalImageWidth * kOriginalImageHeight) / 300, labelMap, clusters);

		out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::MeanShiftElapsedTime, partialTimer.EndStopWatch());

		// 이제 배경이라고 생각되는 클러스터들을 제거해야한다.
		// 기준: 경계선의 +10 픽셀 이내까지 퍼져있는 클러스트들은 배경이다.
		GetBackgroundClusterIndices(cv::Size(kOriginalImageWidth, kOriginalImageHeight), labelMap, kBackgroundTolerance, backgroundClusterIndexSet);

		partialTimer.StartStopWatch();
		// 씨드클러스터를 찾는다. 
		FindSeedClusterInROI(labelMap, backgroundClusterIndexSet, kROI_Rect, maxLabel);
		out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::FindingSeedClusterElapsedTime, partialTimer.EndStopWatch());

		// ROI영역에 거의 배경 클러스터밖에 없다면
		if (backgroundClusterIndexSet.count(maxLabel) == 1)
		{
			cv::Point3i hsvColor;
			GetOriginalHSVColorFromHalfedLuv(clusters[maxLabel].GetLuvColor(), lValueDivider, hsvColor);

			// seed cluster가 배경이면서 동시에 무채색이라면, 주변 배경에 의해 seed가 묻혔을 가능성이 큼.
			// 그러므로 color radius를 줄여서 MeanShift
			if (hsvColor.y < 50)
			{
				// 무채색임 => Re-Mean Shift Procedure
				lValueDivider -= 0.5;	// L을 더 중요시 여겨야함.
				sr -= 1;			// color bandwidth를 줄여서, 배경과 씨드를 구분가게 하기 위함.
				maxLabel = -1;
				backgroundClusterIndexSet.clear();
				clusters.clear();
				labelMap.release();
			}
			else
			{
				// 배경이랑 가운데부분이 동일하게 묵였는데, 게다가 유채색이다 => 알고리즘 종료해야함..
				break;
			}
		}
		// 배경클러스터가 아니면 루프를 종료하고 다음 알고리즘으로 이동한다.
		else
		{
			// filtering된 HSV이미지를 보고싶어서 삽입한 코드이다.
			cv::cvtColor(tempLDividedMat, lValueDividedHSVMat, CV_Luv2BGR);
			cv::cvtColor(lValueDividedHSVMat, lValueDividedHSVMat, CV_BGR2HSV);
			break;
		}
	}
	if (clusters.size() == 0 || backgroundClusterIndexSet.count(maxLabel) == 1)
	{
		return false;
	}

	/*******************************************/
	/*******************************************/
	/******      Color Merging Task     ********/
	/*******************************************/
	/*******************************************/
	partialTimer.StartStopWatch();

	Cluster mergedCluster;		// Merging의 최종결과가 담길 변수 
	PerformColorMergingTask(clusters[maxLabel], clusters, lValueDividedHSVMat, labelMap, lValueDivider, backgroundClusterIndexSet, cv::Size(kOriginalImageWidth, kOriginalImageHeight), 3, mergedCluster);
	out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::MergingTaskElapsedTime, partialTimer.EndStopWatch());

	partialTimer.StartStopWatch();

	int wrappedLabel = 0;
	bool bWrapped = IsClusterWrappedByCertainCluster(mergedCluster, labelMap, 5, 0.6, wrappedLabel);
	bool bBezelMerged = false;

	/*******************************************/
	/*******************************************/
	/******      Bezel Merging Task     ********/
	/*******************************************/
	/*******************************************/
	{
		std::vector<cv::Point> minusPoints;
		Cluster tempClusterForRectDecision;

		cv::Mat tempCanvas(luvOriginalImageMat.rows, luvOriginalImageMat.cols, CV_8UC3, cv::Scalar::all(0));
		std::unordered_map<int, int> labelAndItsFrequency;
		GetAllAdjacentLabelsAndTheirFrequency(mergedCluster, labelMap, 10, labelAndItsFrequency, minusPoints);

		for (auto& eachMinusPoint : minusPoints)
		{
			tempCanvas.at<cv::Vec3b>(eachMinusPoint) = filteredImageMat_luv.at<cv::Vec3b>(eachMinusPoint);
		}

		tempClusterForRectDecision.AddPointsFromArray(minusPoints.data(), minusPoints.size());
		for (auto eachPair : labelAndItsFrequency)
		{
			if (eachPair.first != NOT_CLUSTER_LABEL && backgroundClusterIndexSet.count(eachPair.first) == 0)
			{
				auto& eachCluster = clusters[eachPair.first];
				ProjectClusterIntoMat(eachCluster, tempCanvas);
				tempClusterForRectDecision.Consume(eachCluster);
			}
		}

		std::unordered_map<int, Cluster> neighborClusters;
		cv::Mat tempLabelMap;
		std::vector<int> toBeRemovedKey;

		// 포인트들이 그려진 부분에 대해서만 ROI를 잡고 Re-clustering을 한다.
		auto roi = tempClusterForRectDecision.GetBoundedBox();
		// 주변부에 대해서 세밀하게 클러스터링을 한다.
		PerformClustering(tempCanvas, roi, 50, tempLabelMap, neighborClusters);

		// L=u=v=0인 클러스터는 의미없는 클러스터이므로 제거한다.
		for (auto& eachCluster : neighborClusters)
		{
			auto color = eachCluster.second.GetLuvColor();
			// 기존에 채워두었던 검은색 영역들은 의미있는 클러스터들이 아니므로, 제거대상에 추가
			if (color.x == 0 && color.y == 0 && color.z == 0)
			{
				toBeRemovedKey.push_back(eachCluster.first);
			}
		}
		for (auto& key : toBeRemovedKey)
		{
			neighborClusters.erase(key);
		}

		// find Max Cluster among negihbor clusters
		int maxLabel = 0;
		FindBiggestCluster(neighborClusters, maxLabel);

		// 가장큰 클러스터를 'biggestClusterInNeighborhood'라고 부르자
		Cluster& biggestClusterInNeighborhood = neighborClusters[maxLabel];
		const auto biggestClusterLuvColor = biggestClusterInNeighborhood.GetLuvColor();

		// biggestClusterInNeighborhood와 색상이 유사한 주변 클러스터들을 모두 병합.
		for (auto& eachClusterInfo : neighborClusters)
		{
			if (eachClusterInfo.first == maxLabel)
			{
				continue;
			}
			const Cluster& eachCluster = eachClusterInfo.second;
			const auto eachClusterLuvColor = eachCluster.GetLuvColor();

			auto lDiff = std::abs(eachClusterLuvColor.x - biggestClusterLuvColor.x);
			auto uDiff = std::abs(eachClusterLuvColor.y - biggestClusterLuvColor.y);
			auto vDiff = std::abs(eachClusterLuvColor.z - biggestClusterLuvColor.z);

			if (lDiff <= 10 && uDiff <= 10 && vDiff <= 10)
			{
				biggestClusterInNeighborhood.Consume(eachCluster);
			}
		}

		// 기존 레이블맵에 biggestCluster를 등록해준다, 대신 이영역들은 BEZEL_LABEL번호를 할당받음
		const auto biggestClusterLabelInCurrentLabelMap = BEZEL_LABEL;
		for (auto& eachPoint : biggestClusterInNeighborhood.GetPointsArray())
		{
			clusters.erase(labelMap.at<int>(eachPoint));
			labelMap.at<int>(eachPoint) = biggestClusterLabelInCurrentLabelMap;
		}
		biggestClusterInNeighborhood.RemoveLabelInformation();
		biggestClusterInNeighborhood.RegisterLabel(biggestClusterLabelInCurrentLabelMap);

		// 기존 Cluster 딕셔너리에 biggestClusterInNeighborhood를 등록
		clusters[biggestClusterLabelInCurrentLabelMap] = biggestClusterInNeighborhood;

		if (biggestClusterInNeighborhood.GetTotalPoints() > 0)
		{
			// 베젤 영역의 rotate rect를 구한다.
			auto roatedRectOfBezelRegion = cv::minAreaRect(clusters.at(BEZEL_LABEL).GetPointsArray());
			cv::Point2f fourPointsOfRotatedRect[4];
			roatedRectOfBezelRegion.points(fourPointsOfRotatedRect);

			std::vector<cv::Point2f> vectorVersionOfFourPoints(fourPointsOfRotatedRect, fourPointsOfRotatedRect + 4);

			bool bDummy = false;
			bool bInside = false;
			std::map<double, int> bInsideRecord;
			bInsideRecord[-1] = 0;
			bInsideRecord[1] = 0;
			bInsideRecord[0] = 0;

			int endCol = roatedRectOfBezelRegion.boundingRect().width;
			int endRow = roatedRectOfBezelRegion.boundingRect().height;

			int nImpureThings = 0;
			// 베젤영역을 rotatedRect 영역으로 근사시켰을 때, 베젤영역내부에 존재하는 레이블들을 보고, 그 레이블이 배경레이블이거나 -1이거나 내부 Cluster에 속해있지 않다면
			// nItmputerThings를 증가시키자.
			for (int rowIndex = 0; rowIndex < endRow; ++rowIndex)
			{
				for (int colIndex = 0; colIndex < endCol; ++colIndex)
				{
					if (cv::pointPolygonTest(vectorVersionOfFourPoints, cv::Point(colIndex, rowIndex), bDummy) > 0)
					{
						int label = labelMap.at<int>(rowIndex, colIndex);
						if (backgroundClusterIndexSet.count(label) == 1 || (label != BEZEL_LABEL && label != NOT_CLUSTER_LABEL && mergedCluster.DoesContain(label) == false))
						{
							nImpureThings++;
						}
					}
				}
			}

			if (nImpureThings < 20)
			{
				for (auto& eachPoint : mergedCluster.GetPointsArray())
				{
					// bInsideRecord에서 Key = -1이면 mergedCluster 외부, 0이면 edge상, 1이면 내부이다.
					bInsideRecord[cv::pointPolygonTest(vectorVersionOfFourPoints, eachPoint, bDummy)]++;
				}

				bInside = ((double)bInsideRecord[1] / (bInsideRecord[0] + bInsideRecord[1] + bInsideRecord[-1])) > 0.7;

				// 만일, 베젤 영역이 완전히 Merged Cluster를 포함하고 있다면, Consume한다
				if (bInside == true && (biggestClusterInNeighborhood.GetTotalPoints() < mergedCluster.GetTotalPoints()))
				{
					mergedCluster.Consume(biggestClusterInNeighborhood);
					bBezelMerged = true;
				}
			}

			// 위 과정에 의해서 합쳐지지 않은 경우에는 아래의 케이스에서 처리한다.
			int wrappedLabel = 0;
			if (IsClusterWrappedByCertainCluster(mergedCluster, labelMap, 5, 0.6, wrappedLabel) && wrappedLabel == BEZEL_LABEL && clusters[wrappedLabel].GetTotalPoints() < mergedCluster.GetTotalPoints())
			{
				mergedCluster.Consume(biggestClusterInNeighborhood);
				bBezelMerged = true;
			}
		}
	}

	/*******************************************/
	/*******************************************/
	/******   Extracting Outer Contour  ********/
	/*******************************************/
	/*******************************************/
	{
		ExpandRectInAnyFourDirections(cv::Size(kOriginalImageWidth, kOriginalImageHeight), mergedCluster.m_boundedBox, -5, -5, 10, 10);
		cv::Mat alphaMap = GetAlphaMap(labelMap, mergedCluster.m_boundedBox, mergedCluster);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(alphaMap, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		cv::drawContours(originalImage, contours, -1, cv::Scalar(255, 0, 0), 2);
		cv::rectangle(originalImage, mergedCluster.m_boundedBox, cv::Scalar(0, 255, 0), 2);
		out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::ExtractOuterContourElapsedTime, partialTimer.EndStopWatch());
	}


	/*******************************************/
	/*******************************************/
	/******   Experiment Result Save    ********/
	/*******************************************/
	/*******************************************/
	{
		// Update Final Parameters
		out_result.SetFinalSpatialBandwidth(sp);
		out_result.SetFinalColorBandwidth(sr);
		out_result.SetFinalLDivider(lValueDivider);

		backgroundClusterIndexSet.clear();
		GetBackgroundClusterIndices(cv::Size(kOriginalImageWidth, kOriginalImageHeight), labelMap, kBackgroundTolerance, backgroundClusterIndexSet);
		// 민쉬프트 된 결과이미지도 필요하다면 그려준다.
		if (in_parameter.IsSetToGetSegmentedImage())
		{
			cv::cvtColor(filteredImageMat_luv, out_result.GetSegmentedMat(), CV_Luv2BGR);
			out_result.SetSegmentedMat(out_result.GetSegmentedMat());
		}
		// 클러스터링 결과도 그려준다.
		if (in_parameter.IsSetToGetClusterImage())
		{
			auto& tempClusteredMat = out_result.GetClusteredMat();
			in_srcImage.copyTo(tempClusteredMat);

			for (auto& cluster : clusters)
			{
				if (backgroundClusterIndexSet.count(cluster.first) == 1)
				{
					// 배경클러스터들은 초록색으로 그린다.
					DrawOuterContourOfCluster(tempClusteredMat, cluster.second, cv::Scalar(0, 255, 0));
				}
				else
				{
					if (cluster.first == maxLabel)
					{
						// Seed는 빨간색
						DrawOuterContourOfCluster(tempClusteredMat, cluster.second, cv::Scalar(0, 0, 255));
					}
					else if (mergedCluster.DoesContain(cluster.first))
					{
						if (cluster.first == BEZEL_LABEL && bBezelMerged == true)
						{
							// 베젤부위는 오렌지색
							DrawOuterContourOfCluster(tempClusteredMat, cluster.second, cv::Scalar(39, 127, 255));
						}
						else
						{
							// Merging되는 애들은 하늘색
							DrawOuterContourOfCluster(tempClusteredMat, cluster.second, cv::Scalar(255, 255, 0));
						}
					}
				}
			}
		}

		// 실험결과로 기록
		out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::TotalElapsedTime, totalTimer.EndStopWatch());
		out_result.SetResultMat(originalImage);
	}
	return true;
}