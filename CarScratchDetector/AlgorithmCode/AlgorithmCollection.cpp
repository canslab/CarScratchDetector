#include "AlgorithmCollection.h"
#include "MeanShiftCluster.h"
#include "..\UtilityCode\Timer.h"
#include "..\CarNumberRemoveCode\LPdetection.h" 
#include "..\DBSCAN.h"

#define NOT_CLUSTER_LABEL -1

void PerformClustering(cv::Mat& in_luvWholeImage, const cv::Rect& in_ROI, int in_thresholdToBeCluster, cv::Mat& out_labelMap, std::unordered_map<int, MeanShiftCluster>& out_clusters)
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

	// Boundary�� ����ϱ� ����, ROI ����
	cv::Rect roi;
	roi.x = roi.y = 1;
	roi.width = in_luvWholeImage.cols;
	roi.height = in_luvWholeImage.rows;
	int clusterIndex = 0;
	int currentClusterMagnitude = 0;

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
				currentClusterMagnitude = cv::sum(roiMat)[0];

				//int nPointsInThisCluster = findIndexColumnVector.rows;
				// if # of elements in a cluster is less than a certian number (0.5% of total number of pixels), -1 is assigned to that pixel
				if (currentClusterMagnitude > in_thresholdToBeCluster)
				{
					// Ŭ�����͸� �̷�� ������ ��ġ�� findIndexColumnVector�� ������ ����
					cv::findNonZero(roiMat, findIndexColumnVector);

					// label map �����
					for (int i = 0; i < findIndexColumnVector.rows; ++i)
					{
						auto& pt = findIndexColumnVector.at<cv::Point>(i, 0);
						out_labelMap.at<int>(pt) = clusterIndex;
					}

					// Ŭ������ �����̳�(unordered_map)�� Cluster�� ���
					out_clusters[clusterIndex] = MeanShiftCluster();
					MeanShiftCluster& eachCluster = out_clusters[clusterIndex];
					//eachCluster

					// cluster�� ������ �� ��° ���̺����� ����.
					eachCluster.SetLabel(clusterIndex);

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

/**************************************************/
/****       Utility Functions' Implementation *****/
/**************************************************/

void DrawContoursOfClusters(cv::Mat & in_targetImage, const std::unordered_map <int, MeanShiftCluster>& in_clusters, cv::Scalar in_color)
{
	for (auto& eachCluster : in_clusters)
	{
		DrawOuterContourOfCluster(in_targetImage, eachCluster.second, in_color);
	}
}

void DrawOuterContourOfCluster(cv::Mat & in_targetImage, const MeanShiftCluster & in_cluster, cv::Scalar in_color)
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

void FindAllOuterPointsOfCluster(const cv::Size& in_frameSize, const MeanShiftCluster & in_cluster, std::vector<cv::Point> &out_points)
{
	cv::Mat alphaMap;
	std::vector<std::vector<cv::Point>> contours;

	CreateAlphaMapFromCluster(in_frameSize, in_cluster, alphaMap);
	cv::findContours(alphaMap, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// Mostly, contours has only one set of points. 
	// now out_points has all points lying in the edge of the cluster (in_cluster)
	out_points = contours[0];
}

void CreateAlphaMapFromCluster(const cv::Size & in_alphaMapSize, const MeanShiftCluster & in_cluster, cv::Mat & out_alphaMap)
{
	cv::Mat alphaMap(in_alphaMapSize, CV_8UC1, cv::Scalar(0));
	auto& arrayOfPoints = in_cluster.GetPointsArray();

	for (auto& eachPoint : arrayOfPoints)
	{
		alphaMap.at<uchar>(eachPoint) = 255;
	}

	out_alphaMap = alphaMap;
}

void ProjectClusterIntoMat(const MeanShiftCluster & in_cluster, cv::Mat & out_mat)
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

void FindBiggestCluster(const std::unordered_map<int, MeanShiftCluster>& in_clusters, int & out_biggestClusterLabel)
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

template <typename T>
static int FindMaxIndexInArray(std::vector<T> &in_vector, int in_vectorSize)
{
	int currentMaxIndex = 0;
	T currentMaxValue = in_vector[0];

	for (int i = 0; i < in_vectorSize; ++i)
	{
		if (currentMaxValue < in_vector[i])
		{
			currentMaxValue = in_vector[i];
			currentMaxIndex = i;
		}
	}
	return currentMaxIndex;
}

void PerformColorMergingFromSeedClusterAndUpdateClusterList(std::unordered_map <int, MeanShiftCluster> &in_updatedClusterList, const int in_seedIndex)
{
	// SeedCluster (��ü �� �Ϻ�)�� �߽����� �������̰� �� �ȳ��� ������ Cluster��� merging �۾��� �����Ѵ�.
	MeanShiftCluster &seedCluster = in_updatedClusterList[in_seedIndex];
	cv::Point3i seedClusterHSVColor = seedCluster.GetHSVColor();
	cv::Point3i seedClusterLuvColor = seedCluster.GetLuvColor();

	// ����Ŭ�����Ϳ� ������ Ŭ�����͵��� ���̺��� ��� Set
	std::set<int> toBeMergedClusterIndices;
	std::set<int> toBePreservedClusterIndicies;

	// ����Ŭ�����Ϳ� ������ �༮���� ���ʴ�� ��ȸ�ϸ� Ȯ���Ѵ�.
	for (const auto& eachCluster : in_updatedClusterList)
	{
		// ����Ŭ�����ʹ� ��ȸ�� �ʿ䰡 ����.
		if (eachCluster.second.GetLabel() == in_seedIndex)
		{
			continue;
		}

		const cv::Point3i& eachClusterHSVColor = eachCluster.second.GetHSVColor();
		const cv::Point3i& eachClusterLuvColor = eachCluster.second.GetLuvColor();

		// Merging Criteria�� ����� ����. (HSV�� ����, Luv�� ����)
		auto hueDiff = std::abs(seedClusterHSVColor.x - eachClusterHSVColor.x);
		auto satDiff = std::abs(seedClusterHSVColor.y - eachClusterHSVColor.y);
		auto valDiff = std::abs(seedClusterHSVColor.z - eachClusterHSVColor.z);

		auto lDiff = std::abs(seedClusterLuvColor.x - eachClusterLuvColor.x);
		auto uDiff = std::abs(seedClusterLuvColor.y - eachClusterLuvColor.y);
		auto vDiff = std::abs(seedClusterLuvColor.z - eachClusterLuvColor.z);

		// Merge�ص� ������? ���
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
		else
		{
			toBePreservedClusterIndicies.insert(eachCluster.first);
		}
	}

	// ����Ŭ�����͸� ������� Merging �۾� ����
	for (auto& shouldBeMergedClusterLabel : toBeMergedClusterIndices)
	{
		seedCluster.Consume(in_updatedClusterList[shouldBeMergedClusterLabel]);
	}

	// Merging�� ���� Ŭ�����͵� ���
	std::unordered_map<int, MeanShiftCluster> finalClusterList;
	// �����Ǿ��� �༮���� ���� Ŭ������ ��Ͽ� �����Ѵ�.
	for (auto& shouldBePreservedClusterLabel : toBePreservedClusterIndicies)
	{
		finalClusterList[shouldBePreservedClusterLabel] = in_updatedClusterList[shouldBePreservedClusterLabel];
	}
	// ���嵵 �����Ѵ�.
	finalClusterList[seedCluster.GetLabel()] = seedCluster;

	// ������� ������� �ϱ� ������ Copy�Ѵ�
	in_updatedClusterList = finalClusterList;
}

void Find_TopN_BiggestClusters(const std::unordered_map<int, MeanShiftCluster>& in_clusters, const int in_N, std::vector<int>& out_labels)
{
	//�Ը� ū #(kNumberOfCandidateClusters)���� Ŭ������ ���̺��� ����Ѵ�.
	for (const auto& eachCluster : in_clusters)
	{
		int currentLabel = eachCluster.first;
		int currentSize = eachCluster.second.GetTotalPoints();
		int i = 0;

		while (i <= (in_N - 1))
		{
			if (currentSize > in_clusters.at(out_labels[i]).GetTotalPoints())
			{
				// move one by one until i becomes 2
				while (i <= (in_N - 1))
				{
					int temp = out_labels[i];
					out_labels[i] = currentLabel;
					currentLabel = temp;
					i++;
				}
			}
			i++;
		}
	}
}

void GetAdequacyScoresToBeSeed(const cv::Size in_imageSize,
	const std::unordered_map<int, MeanShiftCluster> &in_clusters,
	const int in_numberOfCandidates,
	const int in_numberOfRandomSamples,
	const std::vector<int> in_candidateClusterLabels,
	int &out_seedLabel)
{
	// �̹��� �߽���, Ŭ������ �ĺκ� ũ�� �����ϴ� ����, Ŭ������ �ĺ��� gaussian weight ��� ���.
	// Gaussian weight�� �̹��� �߽������κ��� �־����� �־������� Ŀ����. (exponential�� minums �������̹Ƿ�, �� ���� Ŀ���� ���� score�� �������ٴ� �Ҹ�)
	// (��, �ָ� �������� ������ ���� seed�� �� �� ����)
	cv::Point2d imageCenter(in_imageSize.width / 2, in_imageSize.height / 2);	// �̹��� �߽�
	std::vector<double> candidateClusterSize(in_numberOfCandidates);
	std::vector<double> candidateClusterWeights(in_numberOfCandidates);
	std::vector<double> candidateAdequacyScoresToBeSeed(in_numberOfCandidates);

	// �� Cluster���� �̹��� �߽����κ��� �󸶳� ���� ����Ͽ�, �ָ� �ּ��� ���� Guassian Weight�� �ο��ؼ�
	// Ŭ�������� �߿䵵�� ����߸��� (SeedCluster�� ���ɼ��� �����)
	auto largerLength = (in_imageSize.width > in_imageSize.height) ? in_imageSize.width : in_imageSize.height;
	const double expCoefficient = -12.5 / pow(largerLength, 2);
	for (int i = 0; i < in_numberOfCandidates; ++i)
	{
		const MeanShiftCluster& currentCandidateCluster = in_clusters.at(in_candidateClusterLabels[i]);
		const int currentCandidateClusterSize = currentCandidateCluster.GetTotalPoints();
		const auto& currentCandidateClusterPoints = currentCandidateCluster.GetPointsArray();

		candidateClusterSize[i] = currentCandidateClusterSize;

		double averageDistanceFromCenter = 0.0;
		// weight�� ����� ��, Ŭ�����Ϳ� ���ϴ� �ȼ��� random�ϰ� �� �� ���ø��Ѵ�.
		for (int sampleIndex = 0; sampleIndex < in_numberOfRandomSamples; ++sampleIndex)
		{
			cv::Point2d tempPoint = currentCandidateClusterPoints[std::rand() % currentCandidateClusterSize];
			averageDistanceFromCenter += cv::norm(tempPoint - imageCenter);
		}
		averageDistanceFromCenter /= in_numberOfRandomSamples;

		candidateClusterWeights[i] = exp(expCoefficient * pow(averageDistanceFromCenter, 2));
		candidateAdequacyScoresToBeSeed[i] = candidateClusterWeights[i] * candidateClusterSize[i];
	}

	// ������ ���� Measure (Gaussian-based Score)�� �������� seedCluster�� ã�´�.
	out_seedLabel = in_candidateClusterLabels[FindMaxIndexInArray<double>(candidateAdequacyScoresToBeSeed, in_numberOfCandidates)];
}

void VisualizeLabelMap(const cv::Mat& in_labelMap, cv::Mat& out_colorLabelMap)
{
	cv::normalize(in_labelMap, out_colorLabelMap, 0, 255, NORM_MINMAX);
	out_colorLabelMap.convertTo(out_colorLabelMap, CV_8UC1);
	cv::applyColorMap(out_colorLabelMap, out_colorLabelMap, COLORMAP_JET);
	cv::imshow("Color Map", out_colorLabelMap);
}

#pragma optimize("gpsy", off)
bool IsThisPointCloseToContour(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point in_thisPoint, double in_distance)
{
	for (const auto& eachContour : in_contours)
	{
		double distance = std::abs(cv::pointPolygonTest(eachContour, in_thisPoint, true));
		if (distance <= in_distance)
		{
			return true;
		}
	}

	return false;
}
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
bool IsThisPointInsideOneOfContours(const std::vector<std::vector<cv::Point>>& in_contours, const cv::Point & in_thisPoint)
{
	std::vector<int> distanceRecord;

	for (int index = 0; index < in_contours.size(); ++index)
	{
		const auto& eachContour = in_contours[index];
		double distance = std::abs(cv::pointPolygonTest(eachContour, in_thisPoint, true));
		distanceRecord.push_back(distance);
	}

	double currentMinValue = 1000000000;
	int currentMinIndex = -1;

	for (int i = 0; i < distanceRecord.size(); ++i)
	{
		if (currentMinValue >= distanceRecord[i])
		{
			currentMinIndex = i;
			currentMinValue = distanceRecord[i];
		}
	}

	return cv::pointPolygonTest(in_contours[currentMinIndex], in_thisPoint, false) >= 0;
}
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
cv::Scalar GetAverageColorOfPointsArray(cv::Mat in_srcImage, const std::vector<cv::Point>& in_points)
{
	double c1 = 0, c2 = 0, c3 = 0;
	for (const auto& eachPoint : in_points)
	{
		const auto& eachColor = in_srcImage.at<cv::Vec3b>(eachPoint);
		c1 = c1 + (double)eachColor[0];
		c2 = c2 + (double)eachColor[1];
		c3 = c3 + (double)eachColor[2];
	}

	return cv::Scalar(c1 / in_points.size(), c2 / in_points.size(), c3 / in_points.size());
}
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
bool IsThisContourInROI(const std::vector<cv::Point>& in_contour, const cv::Size in_imageSize, const cv::Rect in_ROI)
{
	for (auto& eachPoint : in_contour)
	{
		if (!(eachPoint.x >= in_ROI.x && eachPoint.x <= (in_ROI.x + in_ROI.width) && eachPoint.y >= in_ROI.y && eachPoint.y <= (in_ROI.y + in_ROI.height)))
		{
			return false;
		}
	}

	return true;
}
#pragma optimize("gpsy", on)

/*****************************************************/
/****      For Client Function Implementation    *****/
/*****************************************************/

#pragma optimize("gpsy", off)
void CaclculateGradientMap(const cv::Mat &in_imageMat, cv::Mat& out_edgeMap, double in_gradX_alpha, double in_gradY_beta)
{
	if (out_edgeMap.data == nullptr)
	{
		cv::Mat copyMap(in_imageMat.rows, in_imageMat.cols, CV_8UC1, cv::Scalar(0));
		copyMap.copyTo(out_edgeMap);
	}

	// out_edgeMap�� gradient magnitude map�� ����.
	cv::Mat copiedBlurImage;
	GaussianBlur(in_imageMat, copiedBlurImage, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);
	cv::cvtColor(copiedBlurImage, copiedBlurImage, CV_BGR2GRAY);

	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	cv::Sobel(copiedBlurImage, grad_x, CV_16S, 1, 0, 3);
	cv::Sobel(copiedBlurImage, grad_y, CV_16S, 0, 1, 3);

	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	/*for (int rowIndex = 0; rowIndex < in_imageMat.rows; ++rowIndex)
	{
		for (int colIndex = 0; colIndex < in_imageMat.cols; ++colIndex)
		{
			int tempGradX = (int)abs_grad_x.at<uchar>(rowIndex, colIndex);
			int tempGradY = (int)abs_grad_y.at<uchar>(rowIndex, colIndex);
			if (tempGradX < tempGradY)
			{
				out_edgeMap.at<uchar>(rowIndex, colIndex) = tempGradY;
			}
			else
			{
				out_edgeMap.at<uchar>(rowIndex, colIndex) = 0;
			}
		}
	}*/

	cv::addWeighted(abs_grad_x, in_gradX_alpha, abs_grad_y, in_gradY_beta, 0, out_edgeMap);
}
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
void FindPossibleDefectAreasUsingBlobDetection(const cv::Mat & in_imageMat, const std::vector<cv::Point>& out_centerPointsOfPossibleAreas)
{
	cv::Mat testImage;

	cv::cvtColor(in_imageMat, testImage, COLOR_BGR2GRAY);
	float totalPixels = in_imageMat.rows * in_imageMat.cols;

	SimpleBlobDetector::Params blobParams;
	// Change thresholds
	blobParams.minThreshold = 10;
	blobParams.maxThreshold = 200;
	// Filter by Area.
	blobParams.filterByArea = true;
	blobParams.minArea = 10;
	// Filter by Circularity
	blobParams.filterByCircularity = false;
	blobParams.minCircularity = 0.05;
	// Filter by Convexity
	blobParams.filterByConvexity = false;
	blobParams.minConvexity = 0.01;
	// Filter by Inertia
	blobParams.filterByInertia = true;
	blobParams.minInertiaRatio = 0.01;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blobParams);
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(testImage, keypoints);

	cv::drawKeypoints(testImage, keypoints, testImage);
	cv::imshow("Blob Detection Result", testImage);
	int a = 30;
}
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
void GetPointsInContour(const cv::Size& in_imageSize, const std::vector<cv::Point>& in_contour, std::vector<cv::Point>& out_insidePoints)
{
	assert(out_insidePoints.size() == 0 && in_imageSize.width > 0 && in_imageSize.height > 0 && in_contour.size() > 0);

	for (auto rowIndex = 0; rowIndex < in_imageSize.height; ++rowIndex)
	{
		for (auto colIndex = 0; colIndex < in_imageSize.width; ++colIndex)
		{
			bool bInside = (cv::pointPolygonTest(in_contour, cv::Point2f(colIndex, rowIndex), false) > 0);

			if (bInside)
			{
				out_insidePoints.push_back(cv::Point(colIndex, rowIndex));
			}
		}
	}
}
#pragma optimize("gpsy", on)

void UpdateLabelMap(const std::unordered_map<int, MeanShiftCluster>& in_clusters, cv::Mat & inout_labelMap)
{
	for (auto eachCluster : in_clusters)
	{
		auto currentLabel = eachCluster.second.GetLabel();
		const auto& points = eachCluster.second.GetPointsArray();

		for (auto eachPoint : points)
		{
			inout_labelMap.at<int>(eachPoint) = currentLabel;
		}
	}
}
bool IsThisPointInROI(const cv::Rect in_roi, const cv::Point in_point)
{
	return (in_point.x >= in_roi.x && in_point.x < in_roi.x + in_roi.width) && (in_point.y >= in_roi.y && in_point.y < in_roi.y + in_roi.height);
}


#pragma optimize("gpsy", off)
bool ExtractCarBody(const cv::Mat & in_srcImage, const AlgorithmParameter& in_parameter, AlgorithmResult& out_result)
{
	cv::Mat originalImage, copiedGrayImage, originalHSVImage, filteredImageMat_luv, luvOriginalImageMat, filteredImageInBGR, filteredImageInHSV, labelMap, edgeGradientMap;

	in_srcImage.copyTo(originalImage);											// �Է¹��� �̹����� deep copy�ؿ�.

	cv::cvtColor(originalImage, copiedGrayImage, CV_BGR2GRAY);					// �Է¹��� �̹��� �׷��̽����� ȭ
	cv::cvtColor(originalImage, luvOriginalImageMat, CV_BGR2Luv);				// �����̹��� Color Space ��ȯ (BGR -> Luv)
	cv::cvtColor(originalImage, originalHSVImage, CV_BGR2HSV);

	const int kTotalPixels = originalImage.total();								// �� �ȼ��� ����
	const double sp = in_parameter.m_spatialBandwidth;							// Mean Shift Filtering�� ���� spatial radius
	const double sr = in_parameter.m_colorBandwidth;							// Mean Shift Filtering�� ���� range (color) radius

	const int kHueIntervals = 9;												// Hue histogram ���� �� �����, Bin�� ����
	const int kSatIntervals = 16;												// Saturation histogram ���� �� �����, Bin�� ����

	// ROI ����
	const int kROIParameter_Dividier = 8;										// ROI�� ���� ��, Width, Height�� ���� �� ������� ��Ÿ��. 
	const int kWidthMargin = originalImage.cols / kROIParameter_Dividier;
	const int kHeightMargin = originalImage.rows / kROIParameter_Dividier;
	const int kROIWidth = originalImage.cols - (2 * kWidthMargin);
	const int kROIHeight = originalImage.rows - (2 * kHeightMargin);
	const cv::Rect kROI(kWidthMargin, kHeightMargin, kROIWidth, kROIHeight);

	const int kTotalPixelsInROI = kROI.area();									// ROI���ο� �����ϴ� �� �ȼ���
	const double kHighThresholdToHighSatImage = 0.65;							// ��ä�� �̹����̱� ���� Saturation ���� ����, ROI�����ȼ��� 80%(=0.65)�� ��ä�� => ��ä���̹����̴�.
	const double kLowThresholdToHaveHighSatImage = 0.3;							// ��ä�� ����(ex. �ϴû�)�� �����ϴ� �̹����� Saturation ������ 30%(=0.3) �����̴�.

	std::unordered_map<int, MeanShiftCluster> clusters; 						// Cluster ����, Key = Label, Value = Cluster
	cv::pyrMeanShiftFiltering(luvOriginalImageMat, filteredImageMat_luv, sp, sr);
	cv::cvtColor(filteredImageMat_luv, filteredImageInBGR, CV_Luv2BGR);
	cv::cvtColor(filteredImageInBGR, filteredImageInHSV, CV_BGR2HSV);

	cv::Mat highThresholdedGradientMap;
	cv::Mat testImage;
	cv::Mat yGradientMap;
	std::vector<std::vector<cv::Point>> shouldBeExcludedContours;
	originalImage.copyTo(testImage);

	// ���� �׷����Ʈ ��, �⽺�� ��� ���� �׷����Ʈ �� (grad_y only)
	CaclculateGradientMap(originalImage, edgeGradientMap, 0.5, 0.5);
	CaclculateGradientMap(originalImage, yGradientMap, 0, 1);
	cv::medianBlur(edgeGradientMap, edgeGradientMap, 5);

	cv::imshow("�׷����Ʈ ��", edgeGradientMap);

	// ���м��� �⽺�������� �����ؾ��ϴ� Connected Component ���� ����.
	{
		cv::threshold(edgeGradientMap, highThresholdedGradientMap, 60, 255, THRESH_BINARY);
		cv::imshow("High Thresholded �׷����Ʈ �� (90-255 ����)", yGradientMap);
		std::vector<std::vector<cv::Point>> strongEdgeContours;
		cv::findContours(highThresholdedGradientMap, strongEdgeContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		for (int j = 0; j < strongEdgeContours.size(); j++)
		{
			double gradientContourLength = cv::arcLength(strongEdgeContours[j], true);

			// ���м��̶�� Ȯ���� �� �� ������, �ܰ� 30 �ȼ� ���� ������ �������ִٸ� ==> �����Ѵ� (������ ���⼭�� �⽺�� ���� �� ���ٴ� ����)
	/*		int marginWidth = (int)((double)originalImage.cols / 8);
			int marginHeight = (int)((double)originalImage.rows / 8);*/
			//cv::Rect roi(0 + marginWidth, 0 + marginHeight, originalImage.cols - (2 * marginWidth), originalImage.rows - (2 * marginHeight));

			// �׷����Ʈ �ʿ��� ����� ���̰� �� �༮�� �� ���м��� ���ɼ��� ũ�Ƿ� �����ؾ� �ϴ� ������ ���(excludedContourLength)�� ���
			if (gradientContourLength >= 500 || (gradientContourLength >= 10 && !IsThisContourInROI(strongEdgeContours[j], cv::Size(originalImage.cols, originalImage.rows), kROI)))
			{
				shouldBeExcludedContours.push_back(strongEdgeContours[j]);
				cv::drawContours(testImage, strongEdgeContours, j, cv::Scalar(rand() % 256, rand() % 256, rand() % 256), 2);
			}
		}
	}
	cv::imshow("���м� �� �ܰ������� ħ���ϰ� �ִ� �������� ==> �ڳ��� ���� ���� ���", testImage);

	cv::Mat hsvPlanes[3];
	cv::Mat LUVPlanes[3];
	cv::split(filteredImageMat_luv, LUVPlanes);
	cv::split(filteredImageInHSV, hsvPlanes);

	std::vector<int> hueArray(kHueIntervals);
	std::vector<int> satArray(kSatIntervals);

	cv::Mat colorMapOfHue;
	cv::applyColorMap(hsvPlanes[0], colorMapOfHue, COLORMAP_HOT);

	cv::Mat colorMapOfSat;
	cv::applyColorMap(hsvPlanes[1], colorMapOfSat, COLORMAP_HOT);


	// �̹����� ���� ������ �ľ��ϴµ� ���
	// Hue����, Saturation ���� ���
	for (int rowIndex = (int)((double)originalImage.rows / kROIParameter_Dividier) - 1; rowIndex < (int)((double)originalImage.rows * (kROIParameter_Dividier - 1) / kROIParameter_Dividier); ++rowIndex)
	{
		for (int colIndex = (int)((double)originalImage.cols / kROIParameter_Dividier) - 1; colIndex < (int)((double)originalImage.cols * (kROIParameter_Dividier - 1) / kROIParameter_Dividier); ++colIndex)
		{
			hueArray[(int)(hsvPlanes[0].at<uchar>(rowIndex, colIndex) / (180 / kHueIntervals))]++;
			satArray[(int)(hsvPlanes[1].at<uchar>(rowIndex, colIndex) / (256 / kSatIntervals))]++;
		}
	}

	// 0.65�� �Ѿ�� �̰� ��� �����̾�.
	// �̹������� ��κ��� �ȼ��� ��ä����. (0.65�� ��ü�̹��� �ȼ� �� 65%�� 0�� ����� ä���̴�.)
	float lowSaturationPixelRatio = (float)(satArray[0] + satArray[1] + satArray[2]) / kTotalPixelsInROI;
	cv::Mat lastBinaryImage(originalImage.rows, originalImage.cols, CV_8UC1, cv::Scalar::all(0));
	bool bImageHasLowSaturationCarColor = (lowSaturationPixelRatio > kHighThresholdToHighSatImage);
	bool bImageHasCertainColor = (lowSaturationPixelRatio < 0.3);
	if (bImageHasLowSaturationCarColor)
	{
		cv::Mat saturationBinaryImage;
		cv::Mat valueBinaryImage;

		cv::inRange(hsvPlanes[1], 0, 40, saturationBinaryImage);
		// ���� 90���� 255������ ������ �̾Ƴ���.
		cv::threshold(hsvPlanes[2], valueBinaryImage, 90, 255, CV_THRESH_BINARY);
		// �̹������� ä���� �����鼭 ���� ���� �� �̻��ϴ� ������ ��������.
		// �������(��ä��)�� �������� ���� ���� ���� �� �̻��̸鼭 ä���� ����.
		lastBinaryImage = saturationBinaryImage & valueBinaryImage;

		cv::medianBlur(lastBinaryImage, lastBinaryImage, 7);
	}

	// ä�� (���� �ִ�!) �����̸� (������ ����, �ϴû�����)
	else if (bImageHasCertainColor)
	{
		cv::Mat hueThresholdedImage;
		const float kThresholdPercentageToBeMajorHue = 0.6;

		// �̹������� ���� �ֵ� Hue���� �������� ������.
		auto maxHueIndex = FindMaxIndexInArray<int>(hueArray, hueArray.size());
		// �� �ֵ� Hue���� 60%�̻��� �����ϴ� Major�� Hue������ �Ǵ�
		bool bIsThisHueMajority = (float)(hueArray[maxHueIndex] / (kROI.area())) > kThresholdPercentageToBeMajorHue;

		if (true)
		{
			// To reduce noise.
			cv::inRange(hsvPlanes[0], maxHueIndex * (180 / kHueIntervals), (maxHueIndex + 1) * (180 / kHueIntervals), hueThresholdedImage);
			cv::medianBlur(hueThresholdedImage, hueThresholdedImage, 5);
			lastBinaryImage = hueThresholdedImage;
		}
	}
	else
	{
		lastBinaryImage.release();
	}

	// �� ������ �Ǽ� �� �������� �ɷ��� ��� (==lastBinaryImage�� �����ϴ°��)
	if (lastBinaryImage.data)
	{
		// �� �������� �����ϴ� ����ȭ ���� ǥ��
		cv::imshow("Binary Image", lastBinaryImage);

		std::vector<std::vector<cv::Point>> carBodyContours;
		cv::findContours(lastBinaryImage, carBodyContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// Contour�� ���� �� �༮�� ��ü �������̶�� �� �� ����
		int currentMax = 0;
		int currentMaxIndex = 0;
		for (int i = 0; i < carBodyContours.size(); ++i)
		{
			if (currentMax < carBodyContours[i].size())
			{
				currentMax = carBodyContours[i].size();
				currentMaxIndex = i;
			}
		}
		std::vector<cv::Point2f> corners;

		cv::goodFeaturesToTrack(copiedGrayImage(kROI), corners, 1200, 0.01, 1, cv::Mat(), 3, false, 0.04);
		for (auto& point : corners)
		{
			point.x += kROI.x;
			point.y += kROI.y;
		}


		// ������ �ڳʵ��� ��ȸ�ϸ�, Scratch ���ɼ��� �ִ� �ڳʵ��� ������ ǥ����.
		for (auto& point : corners)
		{
			// Specular �Ǵ�, �ڳ��� ���� 3 x 3 ������ ���Ǹ�, ��Ⱚ�� 210�� �Ѿ�� �ڳ����� Specular Point�� ����
			int nSpecularPoints = 0;
			for (int rowIndex = -1; rowIndex <= 1; ++rowIndex)
			{
				for (int colIndex = -1; colIndex <= 1; ++colIndex)
				{
					if ((point.x + colIndex >= 0 && point.y + rowIndex >= 0) && copiedGrayImage.at<uchar>(point.y + rowIndex, point.x + colIndex) > 210)
					{
						nSpecularPoints++;
					}
				}
			}

			// ���� �ڳ����� 3 x 3 ������ 1�� �̻��� ��ȭ�� �ȼ��� �����Ѵ� => Specular��� �Ǵ�
			if (nSpecularPoints >= 1)
			{
				continue;
			}

			// �ڳ��� �߿��� Gradient �� ������� ũ�Ⱑ �Ǿ �ǹ��־�� �ϸ�
			// �ڳ����� ������ü ���ο� �ְų� ��ó�� �־����.
			if (yGradientMap.at<uchar>(point) >= 12 && (cv::pointPolygonTest(carBodyContours[currentMaxIndex], point, false) > 0))
			{
				// �ڳ����� �� ���м� ��ó�� ���� �ʴ� ��츸 �ǹ�����
				if (IsThisPointCloseToContour(shouldBeExcludedContours, point, 5) == false && lastBinaryImage.at<uchar>(point) > 0 && IsThisPointInsideOneOfContours(shouldBeExcludedContours, point) == false)
				{
					cv::circle(originalImage, point, 1, cv::Scalar(0, 255, 0), 2);
				}
			}
		}
	}

	cv::rectangle(originalImage, kROI, cv::Scalar(0, 0, 255), 2);
	cv::imshow("Result", originalImage);
	return true;
}
#pragma optimize("gpsy", on)