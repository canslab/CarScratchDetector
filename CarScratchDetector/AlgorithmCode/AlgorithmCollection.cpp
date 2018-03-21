#include "AlgorithmCollection.h"
#include "Cluster.h"
#include "..\UtilityCode\Timer.h"
#include "..\CarNumberRemoveCode\LPdetection.h"

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
	const int kRowStartIndex = kROI.y; // 0 based value���� ��������.
	const int kRowEndIndex = kRowStartIndex + kROI.height - 1;
	const int kColStartIndex = kROI.x;
	const int kColEndIndex = kColStartIndex + kROI.width - 1;

	// 4 ���� (��, �Ʒ�, ����, ������)���� ������ �����;���.
	// ���� (���̵�)
	for (int rowIndex = kRowStartIndex; rowIndex <= kRowEndIndex; ++rowIndex)
	{
		for (int colIndex = kColStartIndex; colIndex <= kColEndIndex; ++colIndex)
		{
			auto currentLabel = in_labelMap.at<int>(rowIndex, colIndex);

			if (in_centerCluster.DoesContain(currentLabel) == false)
			{
				if (out_labels.count(currentLabel) == 0)
				{
					// currentLabel�� mainCluster�� ������ �ʴ´ٴ°��̹Ƿ�,
					// ������ ������ ���̺��̶� �� �� �ִ�.
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

	// cluster1, cluster2 ������ ������ �ȼ� ���󰪵��� �຤�ͷ� �ű�.
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

	// ���� ������� �� ���� �̹���(Mat)�� Histogram Comparison�Ѵ�.
	int nDims = 1;
	int histSize[] = { in_nBin };

	int kMaxRange = (channelNumber == 0) ? 180 : 256;
	float lRanges[] = { 0, kMaxRange };

	const float* rangeArray[] = { lRanges }; // Luv ��� [0,256) �̹Ƿ�
	int channels[] = { channelNumber }; // L->u->v ������ ������ ��ġ�Ǿ� �����Ƿ�

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

	// label map �Ҵ�
	cv::Mat labelMap(kInputImageHeight, kInputImageWidth, CV_32SC1, cv::Scalar::all(NOT_CLUSTER_LABEL));
	out_labelMap = labelMap;

	// Boundary�� �����ϱ� ����, ROI ����
	cv::Rect roi;
	roi.x = roi.y = 1;
	roi.width = in_luvWholeImage.cols;
	roi.height = in_luvWholeImage.rows;
	int clusterIndex = 0;

	for (int y = in_ROI.y; y < (in_ROI.y + in_ROI.height); ++y)
	{
		for (int x = in_ROI.x; x < (in_ROI.x + in_ROI.width); x++)
		{
			// ���⼭ x, y�� ������ ������ �̳��� �����̸� Ŭ�����͸� ��ü�� ���� �ʴ´�.
			if (afterFloodFillMask.at<uchar>(y + 1, x + 1) == 0)
			{
				cv::Rect boundedBox;
				auto &replacedVector = in_luvWholeImage.at<cv::Vec3b>(y, x);

				// ������ ������ ä���
				cv::floodFill(in_luvWholeImage, afterFloodFillMask, cv::Point(x, y), replacedVector, &boundedBox, cv::Scalar::all(1), cv::Scalar::all(1));

				// ����ũ ����� ������ ����, �̹� �������� � �ȼ����� ������ ������ ä�����°��� ���Ѵ�.
				// �ٷ� �� ������ �༮���� �� Ŭ�����͸� �̷��.
				cv::subtract(afterFloodFillMask, beforeFloodFillMask, subtractedMatrix);
				afterFloodFillMask.copyTo(beforeFloodFillMask);

				cv::Mat roiMat = subtractedMatrix(roi);

				// Ŭ�����͸� �̷�� ������ ��ġ�� findIndexColumnVector�� ������ ����
				cv::findNonZero(roiMat, findIndexColumnVector);

				int nPointsInThisCluster = findIndexColumnVector.rows;
				// if # of elements in a cluster is less than a certian number (0.5% of total number of pixels), -1 is assigned to that pixel
				if (nPointsInThisCluster > in_thresholdToBeCluster)
				{
					// label map �����
					for (int i = 0; i < findIndexColumnVector.rows; ++i)
					{
						auto& pt = findIndexColumnVector.at<cv::Point>(i, 0);
						out_labelMap.at<int>(pt) = clusterIndex;
					}

					// Ŭ������ �����̳�(unordered_map)�� Cluster�� ���
					out_clusters[clusterIndex] = Cluster();
					Cluster& eachCluster = out_clusters[clusterIndex];

					// cluster�� ������ �� ��° ���̺����� ����.
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

	// Key�� Ŭ������ ��ȣ(���̺�), Value�� ���̺��� in_ROI���� ������ Ƚ��
	// in_ROI���ο��� �����ϴ� ���̺��� �� ���̺��� �󵵼��� ����ϴ� ��ųʸ�
	std::unordered_map<int, int> labelToOccurenceMap;

	for (int rowIndex = startRowIndex; rowIndex < endRowIndex; ++rowIndex)
	{
		for (int colIndex = startColIndex; colIndex < endColIndex; ++colIndex)
		{
			auto currentLabel = in_labelMap.at<int>(rowIndex, colIndex);

			// �̼��� �ȼ��� ������ �ڰ��� ����.
			if (currentLabel == NOT_CLUSTER_LABEL)
			{
				continue;
			}

			// ������ �����ϴ� ���̺��� �ƴϴ� => ���� ���
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

	// ��׶��尡 Seed Segment��� ���Դٸ�, ���� ROI���� ��׶��� Ŭ�����͹ۿ� ������ Ȯ���ؾ� �Ѵ�. (70%�̻� �����ϴ��� �����Ѵٴ� �ǹ�)
	// ����, ��׶���Ŭ������ ROI���� ���ݿ����� �������� �ʰ� �ٸ� ���̺� �����鵵 �ִٸ�, �� �������� �Ը� ū ���׸�Ʈ��
	// Seed Segment�� �ɰ��̴�.
	if (in_backgroundIndices.count(maxPair.first) == 1 && (((double)(maxPair.second / totalPoints) < 0.7)))
	{
		// �׷��Ƿ� background Label�� labelToOccurenceMap���� ����������, �ٽ� ���� ū Seed Segment�� ã�´�.
		labelToOccurenceMap.erase(maxPair.first);
		SearchMapForMaxPair(labelToOccurenceMap, 0, maxPair);
	}

	out_seedLabel = maxPair.first;
}
void GetBackgroundClusterIndices(const cv::Size & in_originalImageSize, const cv::Mat& in_labelMap, int in_marginLegnth, std::set<int>& out_backgroundIndiciesSet)
{
	for (int rowIndex = 0; rowIndex < in_originalImageSize.height; ++rowIndex)
	{
		// �� ���, �Ʒ����
		if (rowIndex < in_marginLegnth || rowIndex >= (in_originalImageSize.height - in_marginLegnth))
		{
			// ��üŽ��
			for (int colIndex = 0; colIndex < in_originalImageSize.width; ++colIndex)
			{
				out_backgroundIndiciesSet.insert(in_labelMap.at<int>(rowIndex, colIndex));
			}
		}
		else
		{
			// ��� ���� ��
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

	// Cluster�� Outer Contour�� �̷�� �ִ� ��� ������ ���Ѵ�.
	FindAllOuterPointsOfCluster(in_labelMap.size(), in_cluster, allEdgePoints);
	const int kThreshold = (int)(allEdgePoints.size() * in_ratio);

	// Step 1. 
	// Search all edge points for the adjacent clusters
	// Edge�� �����ϴ� ��� ���鿡 ���Ͽ� �����Ѵ�.
	for (const cv::Point& eachPoint : allEdgePoints)
	{
		cv::Point targetPoint;
		std::unordered_map<int, int> frequencyOfLabels; // Key = Label, Value = # of occurence

		// �� Point�� �������� [in_rangeToCover x in_rangeToCover]�� ������ �����Ѵ�.
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

		// eachPoint�� �ѷ��δ� Ŭ�������� ���� ������ ū �༮�� ���� maxPair�� �����Ѵ�. 
		// maxPair�� key���� ���̺��̰� value�� ���� Ƚ���̴�.
		SearchMapForMaxPair(frequencyOfLabels, 0, maxPair);

		if (labelsFrequencyOfAllEdgePoints.count(maxPair.first) == 0)
		{
			labelsFrequencyOfAllEdgePoints[maxPair.first] = 0;
		}

		// ���� ����Ʈ�� ���ΰ� �ִ� ���̺��� ����ϰ� �󵵸� 1������Ų��.
		labelsFrequencyOfAllEdgePoints[maxPair.first]++;
	}

	// Step 2.
	// labelsFrequencyOfAllEdgePoints�� edge�� ��� ������ �����ϴ� ���� ���̺��� �� ���̺��� �߻��󵵸� ��� �ִ�.
	// Now adjaceLabelOfAllEdgePoints contains adjacent labels of all edge points
	// ����, �е������� (kThreshold)�̻� �����ϴ� label�� �ִٸ�, �� ���̺��� ���δ� ���̺��̶�� �����Ѵ�.
	bWrapped = SearchMapForMaxPair(labelsFrequencyOfAllEdgePoints, kThreshold, maxPair);

	// bWrapped�� true���, maxPair.first���� ���ΰ� �ִ� Ŭ�������� ���̺��� ����.
	out_labelOfWrapperCluster = maxPair.first;
	return bWrapped;
}
void GetAllAdjacentLabelsAndTheirFrequency(const Cluster& in_cluster, const cv::Mat& in_labelMap, int in_rangeToCover, std::unordered_map<int, int> &out_labelAndItsFrequency, std::vector<cv::Point>& out_minusPoints)
{
	std::vector<cv::Point> allEdgePoints;
	std::pair<int, int> maxPair;

	// Cluster�� Outer Contour�� �̷�� �ִ� ��� ������ ���Ѵ�.
	FindAllOuterPointsOfCluster(in_labelMap.size(), in_cluster, allEdgePoints);

	// Step 1. 
	// Search all edge points for the adjacent clusters
	// Edge�� �����ϴ� ��� ���鿡 ���Ͽ� �����Ѵ�.
	for (const cv::Point& eachPoint : allEdgePoints)
	{
		cv::Point targetPoint;
		std::unordered_map<int, int> frequencyOfLabels; // Key = Label, Value = # of occurence

											  // �� Point�� �������� [in_rangeToCover x in_rangeToCover]�� ������ �����Ѵ�.
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

		// eachPoint�� �ѷ��δ� Ŭ�������� ���� ������ ū �༮�� ���� maxPair�� �����Ѵ�. 
		// maxPair�� key���� ���̺��̰� value�� ���� Ƚ���̴�.
		SearchMapForMaxPair(frequencyOfLabels, 0, maxPair);

		// ������ ���ٸ� ���
		if (out_labelAndItsFrequency.count(maxPair.first) == 0)
		{
			out_labelAndItsFrequency[maxPair.first] = 0;
		}

		// ���� ����Ʈ�� ���ΰ� �ִ� ���̺��� �󵵸� 1������Ų��.
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

	// 2.5�� ���������� 2.5�� ����
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
	// Seed Cluster�� �߽����� Ȯ�� ���Ѿ� �Ѵ�.
	auto &seedColorInHSV = in_seedCluster.GetHSVColor();
	// ���� Ȯ���� ���� ROI (extendedRect)
	auto extendedRect = in_seedCluster.GetBoundedBox();
	out_mergedCluster = in_seedCluster;

	// SeedCluster�� HSV Color�� ���ؼ� hsvColorOfSeedCluster�� �����Ѵ�.
	cv::Point3i originalHSVColorOfSeedCluster;
	// hsvColorOfSeedCluster�� Seed Cluster�� ������ ����ִ�.
	GetOriginalHSVColorFromHalfedLuv(in_seedCluster.GetLuvColor(), in_lValudDivider, originalHSVColorOfSeedCluster);

	// ������ �õ��߾��� ���̺����� ���.
	std::set<int> triedLabelsSet;

	// Region Merging �۾�
	while (true)
	{
		std::set<int> labelsOfAdjacentRegions;
		std::set<int> labelsOfSimilarRegions;

		// mainCluster�� ������ �������� ���Ѵ�. 
		// �ٸ�, mainCluster�� �ܺ��̸� extendedRect���ο����� ������ ������ Ž���Ѵ�.
		GetLabelsOfAdjacentClusters(inout_labelMap, in_seedCluster, extendedRect, labelsOfAdjacentRegions);
		for (auto eachLabel : labelsOfAdjacentRegions)
		{
			if (triedLabelsSet.count(eachLabel) == 0)
			{
				triedLabelsSet.insert(eachLabel);
			}
			else
			{
				// ������ ������ �õ��߾��� ���̺��̹Ƿ�, Pass
				continue;
			}

			// �� �׶��� Ŭ������ OR �ұԸ� Ŭ�����͵��� ���մ���� �ƴϴ�.
			if (in_backgroundIndices.count(eachLabel) || eachLabel == NOT_CLUSTER_LABEL)
			{
				continue;
			}

			// ���������� ��¡�ϴ� ���. seedRegion�� L���� 20���� �۰� �����ϴٰ� �����Ǵ� ������ L���� 20���� ������,
			// �Ѵ� ��� ������ �����̹Ƿ� ��ģ��.
			if (in_seedCluster.GetLuvColor().x <= 20 && in_clusters.at(eachLabel).GetLuvColor().x <= 20)
			{
				labelsOfSimilarRegions.insert(eachLabel);
				continue;
			}

			cv::Point3i hsvColorOfAdjacentCluster;
			GetOriginalHSVColorFromHalfedLuv(in_clusters.at(eachLabel).GetLuvColor(), in_lValudDivider, hsvColorOfAdjacentCluster);

			// ä������ ��(�Ѵ� S�� 20���� �۰� V�� 100���� ū ���)�ؼ� 
			// �Ѵ� ������� ����� ȸ���迭�̸� �׸��ڷ� ����ϰ� ��ģ��.
			if ((originalHSVColorOfSeedCluster.z >= 100 && hsvColorOfAdjacentCluster.z >= 100) && (originalHSVColorOfSeedCluster.y <= 20 && hsvColorOfAdjacentCluster.y <= 20))
			{
				labelsOfSimilarRegions.insert(eachLabel);
				continue;
			}

			// disSimilarity ���� 0�� ����� ���� �����ϴ�. Bhattacharyya Coefficient�� ���.
			// ���� Lvalud Divided HSV�� ����Ͽ���..
			auto hueDissimilarity = GetHSVBhattaCoefficient(in_lValueDividedHSVMat, in_clusters.at(eachLabel), in_seedCluster, 0, 30);
			auto satDissimilarity = GetHSVBhattaCoefficient(in_lValueDividedHSVMat, in_clusters.at(eachLabel), in_seedCluster, 1, 16);
			auto valueDissimilarity = GetHSVBhattaCoefficient(in_lValueDividedHSVMat, in_clusters.at(eachLabel), in_seedCluster, 2, 4);

			// ���� ����, ������ ��������� ��������.
			// ������ ����ϴٸ�, �״������� ����, ä���� ������ ��������.
			if (hueDissimilarity < 0.30)
			{
				// Hue���� �����ϳ�, ������ ���� ���̳��� ��찡 �ִ�. (���, ������)
				// �׷� ��쿡��, ������ ������� �������� �ʴ´�.
				// ���� ä���� �������� �̻� ���� ��������, ������ �������� �����ʵ��� �Ѵ�.
				if (valueDissimilarity <= 0.9 && satDissimilarity <= 0.9)
				{
					// ���� mainCluster�� ����� Ŭ�����͵��� ���̺��� ����س��´�.
					labelsOfSimilarRegions.insert(eachLabel);
				}
			}
		}

		// ���̻� ��ĥ Ŭ�����Ͱ� ���ٸ�, nTrial�� ���ҽ�Ű�� ROI(extendedRect) ���� ������Ų��.
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

		// ��ĥ ���������� �ִٸ� ��ġ��, nTrial�� 3���� �ٽ� ����!
		else
		{
			for (auto similarRegionLabel : labelsOfSimilarRegions)
			{
				// mainCluster�� clusters[similarRegionLabel]�� �պ���Ų��.
				// �� ���, mainCluster.m_labels���� similarRegionLabel�� �߰��ȴ�.
				out_mergedCluster.Consume(in_clusters.at(similarRegionLabel));
			}
			in_maxTrial = 3;
		}
	}
}

template <typename T>
static int FindMaxIndexInArray(std::vector<T> &in_vector, int in_totalSize)
{
	int currentMaxIndex = 0;
	T currentMaxValue = in_vector[0];

	for (int i = 0; i < in_totalSize; ++i)
	{
		if (currentMaxValue < in_vector[i])
		{
			currentMaxValue = in_vector[i];
			currentMaxIndex = i;
		}
	}
	return currentMaxIndex;
}

/*****************************************************/
/****      For Client Function Implementation    *****/
/*****************************************************/
#pragma optimize("gpsy", off)
bool ExtractCarBody(const cv::Mat & in_srcImage, const AlgorithmParameter& in_parameter, AlgorithmResult& out_result)
{
	Timer totalTimer;
	Timer partialTimer;
	cv::Mat filteredImageMat_luv;												// �����̹����� �� ����Ʈ ���͸��� ����
	cv::Mat originalImage;														// �����̹���
	cv::Mat segmentedImage;														// �� ����Ʈ ���׸����̼� ��� �̹���
	cv::Mat hsvOriginalImageMat;												// �����̹����� HSV format
	cv::Mat luvOriginalImageMat;												// �����̹����� LUV format

	in_srcImage.copyTo(originalImage);											// �Է¹��� �̹����� deep copy�ؿ�.
	cv::cvtColor(originalImage, hsvOriginalImageMat, CV_BGR2HSV);				// �����̹��� Color Space ��ȯ (BGR -> Hsv)
	cv::cvtColor(originalImage, luvOriginalImageMat, CV_BGR2Luv);				// �����̹��� Color Space ��ȯ (BGR -> Luv)
	out_result.SetElapsedTime(AlgorithmResult::TimerIdentifier::BGRToLuvElapsedTime, partialTimer.EndStopWatch());

	const int kOriginalImageWidth = originalImage.cols;							// ���� ������ �ʺ�
	const int kOriginalImageHeight = originalImage.rows;						// ���� ������ ����
	const int kCenterX = kOriginalImageWidth / 2;								// ROI�� �߾� ��ǥ(x)
	const int kCenterY = kOriginalImageHeight / 2;								// ROI�� �߾� ��ǥ(y)
	const int kROI_RectWidth = kOriginalImageWidth / 3;							// ROI �ʱ� �ʺ�
	const int kROI_RectHeight = kOriginalImageHeight / 3;						// ROI �ʱ� ����
	const cv::Rect kROI_Rect(kCenterX - (kROI_RectWidth / 2), kCenterY - (kROI_RectHeight / 2), kROI_RectWidth, kROI_RectHeight); // ROI ����

	double sp = in_parameter.GetSpatialBandwidth();								// Mean Shift Filtering�� ���� spatial radius
	double sr = in_parameter.GetColorBandwidth();								// Mean Shift Filtering�� ���� range (color) radius

	std::unordered_map<int, Cluster> clusters; 									// Cluster ����, Key = Label, Value = Cluster
	cv::Mat labelMap;															// ���̺���

	int maxLabel = -1;
	const int kTotalIteration = 5;


	////////////////////////////////////////////////////////////////////
	//							                                      //
	// 00.  Mean Shift Clustering & Find Largest Cluster (Car Frame)  //
	//							                                      //
	////////////////////////////////////////////////////////////////////
	// Mean Shift Filtering �ڵ� (OpenCV)
	cv::pyrMeanShiftFiltering(luvOriginalImageMat, filteredImageMat_luv, sp, sr);
	int minThresholdToBeCluster = (int)(kOriginalImageHeight * kOriginalImageWidth * 0.02);
	PerformClustering(filteredImageMat_luv, cv::Rect(0, 0, kOriginalImageWidth, kOriginalImageHeight), minThresholdToBeCluster, labelMap, clusters);

	const int kNumberOfCandidateClusters = 4;
	std::vector<int> candidateClusterLabels(kNumberOfCandidateClusters);
	std::vector<double> candidateClusterMagnitudes(kNumberOfCandidateClusters);
	std::vector<double> candidateClusterWeights(kNumberOfCandidateClusters);
	std::vector<double> candidateClusterScores(kNumberOfCandidateClusters);

	// Get Top 3 Clusters in terms of the number of elements of each each cluster
	// �Ը� ū #(kNumberOfCandidateClusters)���� Ŭ������ ���̺��� ����Ѵ�.
	for (const auto& eachCluster : clusters)
	{
		int currentLabel = eachCluster.first;
		int currentSize = eachCluster.second.GetTotalPoints();
		int i = 0;

		while (i <= (kNumberOfCandidateClusters - 1))
		{
			if (currentSize > clusters[candidateClusterLabels[i]].GetTotalPoints())
			{
				// move one by one until i becomes 2
				while (i <= (kNumberOfCandidateClusters - 1))
				{
					int temp = candidateClusterLabels[i];
					candidateClusterLabels[i] = currentLabel;
					currentLabel = temp;
					i++;
				}
			}
			i++;
		}
	}

	cv::Point2d imageCenter(kCenterX, kCenterY);
	auto largerLength = (kOriginalImageWidth > kOriginalImageHeight) ? kOriginalImageWidth : kOriginalImageHeight;
	const double expCoefficient = -12.5 / pow(largerLength, 2);

	for (int i = 0; i < kNumberOfCandidateClusters; ++i)
	{
		candidateClusterMagnitudes[i] = clusters[candidateClusterLabels[i]].GetTotalPoints();
		candidateClusterWeights[i] = exp(expCoefficient * pow(cv::norm(clusters[candidateClusterLabels[i]].GetCenterPoint() - imageCenter), 2));
		candidateClusterScores[i] = candidateClusterWeights[i] * candidateClusterMagnitudes[i];
	}

#if true
	cv::Mat filteredImageInBGR;
	cv::Mat filteredImageInHSV;

	cv::cvtColor(filteredImageMat_luv, filteredImageInBGR, CV_Luv2BGR);
	cv::cvtColor(filteredImageInBGR, filteredImageInHSV, CV_BGR2HSV);
	cv::imshow("Filtered Image", filteredImageInBGR);
#endif

	/////////////////////////////////
	//							   //
	// 02.  Color Merging		   //
	//							   //
	/////////////////////////////////
	int seedClusterIndex = candidateClusterLabels[FindMaxIndexInArray<double>(candidateClusterScores, kNumberOfCandidateClusters)];

	// ���� ������ ���� Ŭ�����Ͱ�, �õ� Ŭ������(��ü �� �Ϻ�)�̴�. 
	// ���� �� Ŭ�����͸� �������� Merging�Ѵ�.
	Cluster seedCluster = clusters[seedClusterIndex];
	cv::Point3i seedClusterHSVColor = seedCluster.GetHSVColor();
	cv::Point3i seedClusterLuvColor = seedCluster.GetLuvColor();

	std::set<int> toBeMergedClusterIndices;

	for (const auto& eachCluster : clusters)
	{
		const cv::Point3i& eachClusterHSVColor = eachCluster.second.GetHSVColor();
		const cv::Point3i& eachClusterLuvColor = eachCluster.second.GetLuvColor();

		auto hueDiff = std::abs(seedClusterHSVColor.x - eachClusterHSVColor.x);
		auto satDiff = std::abs(seedClusterHSVColor.y - eachClusterHSVColor.y);
		auto valDiff = std::abs(seedClusterHSVColor.z - eachClusterHSVColor.z);

		auto lDiff = std::abs(seedClusterLuvColor.x - eachClusterLuvColor.x);
		auto uDiff = std::abs(seedClusterLuvColor.y - eachClusterLuvColor.y);
		auto vDiff = std::abs(seedClusterLuvColor.z - eachClusterLuvColor.z);

		bool bOkayToMerge = false;

		if (hueDiff <= 5 && satDiff <= 10 && valDiff <= 10)
		{
			// similar! 
			bOkayToMerge = true;
		}
		else if (hueDiff <= 5 && seedClusterHSVColor.y >= 200 && eachClusterHSVColor.y >= 200 && seedClusterHSVColor.z >= 100 && eachClusterHSVColor.z >= 100)
		{
			bOkayToMerge = true;
		}
		else if (lDiff <= 50 && uDiff <= 15 && vDiff <= 15)
		{
			bOkayToMerge = true;
		}

		if (bOkayToMerge == true)
		{
			toBeMergedClusterIndices.insert(eachCluster.first);
		}
	}
	for (auto& eachIndex : toBeMergedClusterIndices)
	{
		seedCluster.Consume(clusters[eachIndex]);
	}

	DrawOuterContourOfCluster(originalImage, seedCluster, cv::Scalar(255, 255, 0));

	return true;
}
#pragma optimize("gpsy", on)