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

void GetTopNClusters(const std::unordered_map<int, MeanShiftCluster>& in_clusters, const int in_N, std::vector<MeanShiftCluster>& out_sortedArray)
{
	// Declaring the type of Predicate that accepts 2 pairs and return a bool
	typedef std::function<bool(std::pair<int, MeanShiftCluster>, std::pair<int, MeanShiftCluster>)> Comparator;

	// Defining a lambda function to compare two pairs. It will compare two pairs using second field
	Comparator compFunctor =
		[](std::pair<int, MeanShiftCluster> elem1, std::pair<int, MeanShiftCluster> elem2)
	{
		return elem1.second.GetTotalPoints() > elem2.second.GetTotalPoints();
	};

	// Declaring a set that will store the pairs using above comparision logic
	std::set<std::pair<int, MeanShiftCluster>, Comparator> sortedCluster(
		in_clusters.begin(), in_clusters.end(), compFunctor);

	int idx = 0;
	for (auto& eachElem : sortedCluster)
	{
		if (idx < in_N)
		{
			out_sortedArray.push_back(eachElem.second);
		}
		else
		{
			break;
		}
		idx++;
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
bool IsThisPointCloseToGivenContour(const cv::Point & in_point, const std::vector<cv::Point>& in_givenContour, double in_distance)
{
	auto distance = std::abs(cv::pointPolygonTest(in_givenContour, in_point, true));

	return (distance <= in_distance);
}
bool IsThisPointCloseToOneOfContours(const std::vector<std::vector<cv::Point>> &in_contours, const cv::Point in_thisPoint, double in_distance)
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
bool IsThisPointInsideOneOfContours(const std::vector<std::vector<cv::Point>>& in_contours, const cv::Point & in_thisPoint)
{
	std::vector<int> distanceRecord;
	if (in_contours.size() == 0)
	{
		return false;
	}

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
bool IsThisContourContainedInROI(const std::vector<cv::Point>& in_contour, const cv::Size in_imageSize, const cv::Rect in_ROI)
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

/*****************************************************/
/****      For Client Function Implementation    *****/
/*****************************************************/
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
void GetPointsInContour(const std::vector<cv::Point> &in_contour, const double in_distanceFromBoundaryToBeInsidePoint, std::vector<cv::Point> &out_insidePoints)
{
	assert(out_insidePoints.size() == 0 && in_imageSize.width > 0 && in_imageSize.height > 0 && in_contour.size() > 0);
	out_insidePoints.clear();
	auto boundingRect = cv::boundingRect(in_contour);

	for (auto rowIndex = boundingRect.y; rowIndex < boundingRect.y + boundingRect.height; ++rowIndex)
	{
		for (auto colIndex = boundingRect.x; colIndex < boundingRect.x + boundingRect.width; ++colIndex)
		{
			bool bInside = (cv::pointPolygonTest(in_contour, cv::Point2f(colIndex, rowIndex), true) >= in_distanceFromBoundaryToBeInsidePoint);

			if (bInside)
			{
				out_insidePoints.push_back(cv::Point(colIndex, rowIndex));
			}
		}
	}
}
void ResizeImageUsingThreshold(cv::Mat & in_targetImage, int in_totalPixelThreshold)
{
	// To reduce the image's resolution.
	double aspectRatio = (double)in_targetImage.rows / in_targetImage.cols;
	int resizedWidth = 0, resizedHeight = 0, accValue = 0, totalPixel = 0;

	do
	{
		accValue += 10;
		totalPixel = (int)(accValue * accValue * aspectRatio);

	} while (totalPixel <= in_totalPixelThreshold);

	resizedWidth = accValue;
	resizedHeight = aspectRatio * accValue;

	if (resizedWidth <= in_targetImage.cols && resizedHeight <= in_targetImage.rows)
	{
		// �̹��� ������ ����
		cv::resize(in_targetImage, in_targetImage, cv::Size(resizedWidth, resizedHeight));
	}
}
void GaussianBlurToContour(cv::Mat & in_targetGrayImage, const std::vector<cv::Point>& in_contour)
{
	cv::Mat copiedImage;
	in_targetGrayImage.copyTo(copiedImage);

	for (auto& eachPoint : in_contour)
	{
		const int centerPoint_y = eachPoint.y;
		const int centerPoint_x = eachPoint.x;
		int accPixelIntensity = 0;
		int nCount = 0;

		for (int rowIndex = centerPoint_y - 1; rowIndex >= 0 && rowIndex < centerPoint_y + 1; ++rowIndex)
		{
			for (int colIndex = centerPoint_x - 1; colIndex >= 0 && colIndex < centerPoint_x + 1; ++colIndex)
			{
				accPixelIntensity += copiedImage.at<uchar>(rowIndex, colIndex);
				nCount++;
			}
		}

		in_targetGrayImage.at<uchar>(eachPoint) = static_cast<uchar>((float)accPixelIntensity / nCount);
	}
}
bool IsContourInsideCarBody(const std::vector<cv::Point>& in_contour, const std::vector<cv::Point>& in_carBodyContourPoints)
{
	if (in_carBodyContourPoints.size() == 0)
	{
		return true;
	}

	for (auto& eachPoint : in_contour)
	{
		if (cv::pointPolygonTest(in_carBodyContourPoints, eachPoint, false) < 0)
		{
			return false;
		}
	}
	return true;
}
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
bool IsThisPointNearByROI(const cv::Rect & in_roi, const cv::Point & in_point, unsigned int in_distance)
{
	auto ptX = in_point.x;
	auto ptY = in_point.y;

	return (ptX + in_distance >= in_roi.x + in_roi.width) || (ptX - in_distance <= in_roi.x) || (ptY + in_distance >= in_roi.y + in_roi.height) || (ptY - in_distance <= in_roi.y);
}

/******************************************************/
void DetectScratchPointsFromExtractionResult(const cv::Mat in_targetImage, const cv::Rect in_ROI, const cv::Mat in_carBodySegmentedImage, const std::vector<cv::Point> in_carBodyContourPoints,
	std::vector<cv::Point2f> &out_scratchPoints)
{
	cv::Mat copiedGrayImage;
	cv::Mat highThresholdedGradientMap;
	cv::Mat lowThresholdGradientMap;
	cv::Mat gradientForCornerDetection;
	cv::Mat edgeGradientMap;
	cv::Mat excludePartsImage;			// �����ؾ��ϴ� �κе��� ǥ���� �̹���
	cv::Mat scratchDPImage;				// ��ũ��ġ�� ǥ���ϱ� ���� �̹���
	std::vector<std::vector<cv::Point>> shouldBeExcludedContours;
	std::vector<std::vector<cv::Point>> strongEdgeContours;
	std::vector<cv::Point2f> scratchCandidates;

	//in_targetImage.copyTo(excludePartsImage);
	cv::cvtColor(in_targetImage, copiedGrayImage, CV_BGR2GRAY);

	// ���� �׷����Ʈ ��, �⽺�� ��� ���� �׷����Ʈ �� (grad_y only)
	CaclculateGradientMap(in_targetImage, edgeGradientMap, 0.5, 0.5);
	CaclculateGradientMap(in_targetImage, gradientForCornerDetection, 0.3, 0.7);
	out_scratchPoints.clear();

	// ���м��� �⽺�������� �����ؾ��ϴ� Connected Component ���� ����.
	cv::threshold(edgeGradientMap, highThresholdedGradientMap, 55, 255, THRESH_BINARY);
	gradientForCornerDetection = gradientForCornerDetection & in_carBodySegmentedImage;

	// �ð�ȭ
	//cv::imshow("�׷����Ʈ ��", edgeGradientMap);
	//cv::imshow("y Gradient �׷����Ʈ �� (90-255 ����)", gradientForCornerDetection);

	// Gradient Map���� Connected Component�� ���
	cv::findContours(highThresholdedGradientMap, strongEdgeContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// �� ���м� ���� & �β��� ���� �⽺ ���� ����
	for (int j = 0; j < strongEdgeContours.size(); j++)
	{
		std::vector<std::vector<cv::Point>> hull(1);
		cv::convexHull(cv::Mat(strongEdgeContours[j]), hull[0]);

		double gradientContourLength = cv::arcLength(hull[0], true);

		std::vector<cv::Point> innerPoints;
		GetPointsInContour(strongEdgeContours[j], 2.5, innerPoints);

		// �׷����Ʈ �ʿ��� ����� ���̰� �� �༮�� �� ���м��� ���ɼ��� ũ�Ƿ� �����ؾ� �ϴ� ������ ���(excludedContourLength)�� ���
		if (gradientContourLength >= 500 || (!IsThisContourContainedInROI(strongEdgeContours[j], cv::Size(in_targetImage.cols, in_targetImage.rows), in_ROI)))
		{
			cv::Scalar color(0, 200, 200);

			// ���м����� �����Ѵ�.
			shouldBeExcludedContours.push_back(strongEdgeContours[j]);

			// ���м����� Visualize�Ѵ�
			//cv::drawContours(excludePartsImage, strongEdgeContours, j, color, 2);
		}
		else if (gradientContourLength >= 10 && innerPoints.size() > 0
			&& IsThisContourContainedInROI(strongEdgeContours[j], cv::Size(in_targetImage.cols, in_targetImage.rows), in_ROI)
			&& IsContourInsideCarBody(strongEdgeContours[j], in_carBodyContourPoints))
		{
			// �β��� �⽺ �ĺ��� �׸���
			//cv::drawContours(excludePartsImage, strongEdgeContours, j, cv::Scalar(255, 255, 0), 2);

			unsigned int innerGrayValueMean = 0;
			for (const auto& eachInnerPoint : innerPoints)
			{
				uchar grayValue = copiedGrayImage.at<uchar>(eachInnerPoint);
				innerGrayValueMean += (int)grayValue;
			}
			// �β��� ���� ������ Gray value ���
			innerGrayValueMean /= innerPoints.size();

			// �β��� ���� �⽺�� �Ǳ� ���� ������ �Ʒ��� ����
			// (�β��� ���� ���ο� �ִ� gray value���� ���) - (�� ������ �����ϴ� �ٿ�� �ڽ� ���ο��� gray���) >= 30 
			// ��հ��� ���� ���� ���� == �β��� �⽺��
			auto boundingRect = cv::boundingRect(strongEdgeContours[j]);
			auto boundingRectMean = cv::sum(copiedGrayImage(boundingRect))[0];
			boundingRectMean /= boundingRect.area();

			// ���� ���ߵ���, ����� ���� ���� ����, �� ��ũ��ġ���̴�.. �����Ѵ� 
			if (std::abs(innerGrayValueMean - boundingRectMean) >= 30)
			{
				innerPoints.clear();
				GetPointsInContour(strongEdgeContours[j], 1, innerPoints);

				for (auto eachInnerPoint : innerPoints)
				{
					scratchCandidates.push_back(eachInnerPoint);
				}
			}
		}
	}

	// Segmentation�� �� �̷���� ���
	if (in_carBodySegmentedImage.data)
	{
		std::vector<cv::Point2f> corners;
		std::vector<std::vector<cv::Point>> tempForCarBodyVisualize;

		cv::Mat pointRecordedMat(in_targetImage.rows, in_targetImage.cols, CV_8UC1, cv::Scalar(0));
		cv::goodFeaturesToTrack(copiedGrayImage(in_ROI), corners, 1500, 0.005, 0.5, cv::Mat(), 3, false);

		// �� ������ �׷��ش�.
		tempForCarBodyVisualize.push_back(in_carBodyContourPoints);
		//cv::drawContours(excludePartsImage, tempForCarBodyVisualize, -1, cv::Scalar(255, 0, 0), 2);

		// �ڳʵ��� ROI ���ο��� ���������Ƿ�, �ٽ� compensate ���ش�.
		for (auto& eachPoint : corners)
		{
			eachPoint.x += in_ROI.x;
			eachPoint.y += in_ROI.y;
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
			// �ڳ����� ���� ��ü ���ο� �ְų� ��ó�� �־����.
			auto distanceFromSegmentedBoundary = cv::pointPolygonTest(in_carBodyContourPoints, point, true);

			// ��ũ��ġ�� ROI ���ο� �����ϰ� ���ÿ� ������� Gradient ���� �־����.
			if (gradientForCornerDetection.at<uchar>(point) >= 8 && IsThisPointInROI(in_ROI, point) == true)
			{
				// ROI �߿����� �ܰ��� �ƴϾ�߸� ��
				if (IsThisPointNearByROI(in_ROI, point, 13) == false)
				{
					if ((distanceFromSegmentedBoundary > 1 && !IsThisPointCloseToOneOfContours(shouldBeExcludedContours, point, 5) && !IsThisPointInsideOneOfContours(shouldBeExcludedContours, point)
						|| (distanceFromSegmentedBoundary >= -2 && distanceFromSegmentedBoundary <= -1)))
					{
						scratchCandidates.push_back(point);
					}
				}
			}
		}

		// isolated �Ǿ� �ִ� ��ũ��ġ �ĺ� ������ ���� �ϱ� ���ؼ�, �̹����� �� ��ũ��ġ ����Ʈ�� ����Ѵ�.
		for (auto& eachScratchCandidatePoint : scratchCandidates)
		{
			pointRecordedMat.at<uchar>(eachScratchCandidatePoint) = 255;
		}

		// 
		auto marginWidth_ROI = in_ROI.width / 4;
		auto marginHeight_ROI = in_ROI.height / 4;

		// ��ũ��ġ ����Ʈ �ĺ������� ��ȸ�ϸ鼭, ��ũ��ġ�� ������ ���̵鸸 �����ش�. 
		for (auto& eachPoint : scratchCandidates)
		{
			auto x_left = std::abs(eachPoint.x - in_ROI.x);
			auto x_right = std::abs((in_ROI.x + in_ROI.width) - eachPoint.x);

			auto y_upper = std::abs(eachPoint.y - in_ROI.y);
			auto y_bottom = std::abs((in_ROI.y + in_ROI.height) - eachPoint.y);

			auto distanceFromROI_x = (x_left < x_right) ? x_left : x_right;
			auto distanceFromROI_y = (y_upper < y_bottom) ? y_upper : y_bottom;

			// ROI���� ���濡 ��ġ�� �༮�� (isolated�� �ڳ�)�� ���ο� ó���� �ʿ��ϴ�.
			if (distanceFromROI_x < marginWidth_ROI || distanceFromROI_y < marginHeight_ROI)
			{
				int count = 0;
				for (int rowIndex = eachPoint.y - 3; rowIndex < eachPoint.y + 3; ++rowIndex)
				{
					for (int colIndex = eachPoint.x - 3; colIndex < eachPoint.x + 3; ++colIndex)
					{
						if (pointRecordedMat.at<uchar>(rowIndex, colIndex) > 0)
						{
							count++;
						}
					}
				}

				// ���濡 ��ġ�� ���ֺ��� �ٸ� �ڳʰ� ������ isolate�� �ƴ϶�� ����
				if (count > 2)
				{
					out_scratchPoints.push_back(eachPoint);
				}
			}
			// ������ �ƴ� �߾Ӻ����� ���� ���� ������ �������� �ʰ� ����
			else
			{
				out_scratchPoints.push_back(eachPoint);
			}
		}

		//cv::imshow("Binary Image", in_carBodySegmentedImage);
		//cv::imshow("�� �ܰ��� + ���м� + �β��� �⽺", excludePartsImage);
	}
}
void ConstructScratchClusters(const std::vector<cv::Point2f>& in_scratchPoints, std::vector<Cluster_DBSCAN>& out_clusters)
{
	std::list<Point_DBSCAN*> pointsForDBSCAN;
	GeneratePointsForDBSCAN(in_scratchPoints, pointsForDBSCAN);
	PerformDBSCAN(pointsForDBSCAN, 10, 1, out_clusters);
}
void GetMeanStdDevFromPoints(const std::vector<cv::Point2f>& in_scratchPoints, cv::Point2f &out_meanPoint, std::vector<float>& out_stddev)
{
	std::vector<double> mean, var;
	cv::meanStdDev(in_scratchPoints, mean, var);

	out_meanPoint.x = mean[0];
	out_meanPoint.y = mean[1];
	out_stddev[0] = var[0];
	out_stddev[1] = var[1];
}
double GetSkewness(double in_xStdDev, double in_yStdDev)
{
	double skewness = (in_yStdDev / in_xStdDev);

	return skewness;
}
void GetBoundingBoxOfScratchPoints(const cv::Size& in_imageSize, const std::vector<cv::Point2f>& in_scratchPoints, bool in_bExcludingOutlier, cv::Rect& out_boundingBox,
	const unsigned int in_outlierTolerance)
{
	cv::Mat recordMat(in_imageSize.height, in_imageSize.width, CV_8UC1, cv::Scalar::all(0));
	std::vector<cv::Point2f> scratchPointsAfterPostProcessing;
	std::vector<double> xCoords;
	std::vector<double> yCoords;

	for (auto& eachPoint : in_scratchPoints)
	{
		recordMat.at<uchar>(eachPoint) = 255;
	}

	if (in_bExcludingOutlier)
	{
		for (int index = 0; index < in_scratchPoints.size(); ++index)
		{
			cv::Point currentPt;
			currentPt.x = (unsigned int)in_scratchPoints[index].x;
			currentPt.y = (unsigned int)in_scratchPoints[index].y;

			int nOtherScratchPoints = 0;
			for (auto rowIndex = currentPt.y - in_outlierTolerance; rowIndex < currentPt.y + in_outlierTolerance; ++rowIndex)
			{
				for (auto colIndex = currentPt.x - in_outlierTolerance; colIndex < currentPt.x + in_outlierTolerance; ++colIndex)
				{
					if (recordMat.at<uchar>(rowIndex, colIndex) == 255)
					{
						nOtherScratchPoints++;
					}
				}
			}

			if (nOtherScratchPoints >= 2)
			{
				// �� ���� �ƿ����̾��� ==> ���� ���
				scratchPointsAfterPostProcessing.push_back(in_scratchPoints[index]);
			}
		}
	}

	else
	{
		scratchPointsAfterPostProcessing = in_scratchPoints;
	}

	xCoords.resize(scratchPointsAfterPostProcessing.size());
	yCoords.resize(scratchPointsAfterPostProcessing.size());

	for (int index = 0; index < scratchPointsAfterPostProcessing.size(); ++index)
	{
		xCoords[index] = scratchPointsAfterPostProcessing[index].x;
		yCoords[index] = scratchPointsAfterPostProcessing[index].y;
	}

	if (scratchPointsAfterPostProcessing.size() > 0)
	{
		std::sort(xCoords.begin(), xCoords.end());
		std::sort(yCoords.begin(), yCoords.end());

		out_boundingBox.x = (int)xCoords[0];
		out_boundingBox.width = (int)(xCoords.back() - xCoords.front());

		out_boundingBox.y = (int)yCoords[0];
		out_boundingBox.height = (int)(yCoords.back() - yCoords.front());
	}
	else
	{

	}
}

#pragma optimize("gpsy", off)
bool ExtractCarBody(const cv::Mat& in_srcImage, const cv::Rect in_ROI, cv::Mat& out_carBodyBinaryImage, std::vector<cv::Point>& out_carBodyContour,
	const AlgorithmParameter& in_parameter, AlgorithmResult& out_finalParameter)
{
	cv::Mat originalImage, copiedGrayImage, originalHSVImage, filteredImageMat_luv, luvOriginalImageMat, filteredImageInBGR, filteredImageInHSV, labelMap, edgeGradientMap;

	in_srcImage.copyTo(originalImage);											// �Է¹��� �̹����� deep copy�ؿ�.

	cv::cvtColor(originalImage, copiedGrayImage, CV_BGR2GRAY);					// �Է¹��� �̹��� �׷��̽����� ȭ
	cv::cvtColor(originalImage, luvOriginalImageMat, CV_BGR2Luv);				// �����̹��� Color Space ��ȯ (BGR -> Luv)
	cv::cvtColor(originalImage, originalHSVImage, CV_BGR2HSV);

	const int kTotalPixels = originalImage.total();								// �� �ȼ��� ����
	const int kHueIntervals = 9;												// Hue histogram ���� �� �����, Bin�� ����
	const int kSatIntervals = 16;												// Saturation histogram ���� �� �����, Bin�� ����
																				// ROI ����
	const int kROIWidth = in_ROI.width;
	const int kROIHeight = in_ROI.height;

	const int kTotalPixelsInROI = in_ROI.area();									// ROI���ο� �����ϴ� �� �ȼ���
	const double kHighThresholdToHighSatImage = 0.50;							// ��ä�� �̹����̱� ���� Saturation ���� ����, ROI�����ȼ��� 80%(=0.65)�� ��ä�� => ��ä���̹����̴�.
	const double kLowThresholdToHaveHighSatImage = 0.3;							// ��ä�� ����(ex. �ϴû�)�� �����ϴ� �̹����� Saturation ������ 30%(=0.3) �����̴�.

	cv::Mat hsvPlanes[3];
	cv::Mat LUVPlanes[3];
	std::vector<int> hueArray(kHueIntervals);
	std::vector<int> satArray(kSatIntervals);

	cv::pyrMeanShiftFiltering(luvOriginalImageMat, filteredImageMat_luv, 10, 16);
	std::unordered_map<int, MeanShiftCluster> clusters; 						// Cluster ����, Key = Label, Value = Cluster
	// ROI ���ο����� Clustering�� ����
	PerformClustering(filteredImageMat_luv, in_ROI, 100, labelMap, clusters);

	const int kN = 5;
	std::vector<MeanShiftCluster> topNClusters;
	GetTopNClusters(clusters, kN, topNClusters);		// �Ը� ū top (kN)�� Ŭ������ ����
	const cv::Point3i seedClusterColor = topNClusters[0].GetHSVColor();

	cv::cvtColor(filteredImageMat_luv, filteredImageInBGR, CV_Luv2BGR);
	cv::cvtColor(filteredImageInBGR, filteredImageInHSV, CV_BGR2HSV);

	cv::split(filteredImageMat_luv, LUVPlanes);
	cv::split(filteredImageInHSV, hsvPlanes);

	cv::Mat colorMapOfHue;
	cv::applyColorMap(hsvPlanes[0], colorMapOfHue, COLORMAP_HOT);

	cv::Mat colorMapOfSat;
	cv::applyColorMap(hsvPlanes[1], colorMapOfSat, COLORMAP_HOT);

	uchar biggestClusterHValue = seedClusterColor.x; //clusters[bigClusterLabel].GetHSVColor().x;
	uchar biggestClusterVValue = seedClusterColor.z;  //clusters[bigClusterLabel].GetHSVColor().z;

	// �̹����� ���� ������ �ľ��ϴµ� ���
	// Hue����, Saturation ���� ���
	for (int rowIndex = in_ROI.y; rowIndex < in_ROI.y + in_ROI.height; ++rowIndex)
	{
		for (int colIndex = in_ROI.x; colIndex < in_ROI.x + in_ROI.width; ++colIndex)
		{
			hueArray[(int)(hsvPlanes[0].at<uchar>(rowIndex, colIndex) / (180 / kHueIntervals))]++;
			satArray[(int)(hsvPlanes[1].at<uchar>(rowIndex, colIndex) / (256 / kSatIntervals))]++;
		}
	}

	// 0.65�� �Ѿ�� �̰� ��� �����̾�.
	// �̹������� ��κ��� �ȼ��� ��ä����. (0.65�� ��ü�̹��� �ȼ� �� 65%�� 0�� ����� ä���̴�.)
	float lowSaturationPixelRatio = (float)(satArray[0] + satArray[1] + satArray[2]) / kTotalPixelsInROI;
	cv::Mat lastBinaryImage(originalImage.rows, originalImage.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat whiteInRoiMat;
	lastBinaryImage(in_ROI) = 255;

	bool bImageHasLowSaturationCarColor = (lowSaturationPixelRatio > kHighThresholdToHighSatImage);
	bool bImageHasCertainColor = (lowSaturationPixelRatio < 0.3);

	// ��� ���� ������ ����.
	if (bImageHasLowSaturationCarColor)
	{
		cv::Mat saturationBinaryImage;
		cv::Mat valueBinaryImage;
		int currentUpperValue = 0;
		int num_highValuePixels = 0;
		uchar upperValueBound = 0;
		uchar lowerValueBound = 0;

		// top N Ŭ�����Ϳ��� ���ʷ� ��� �κ��� ã�ƾ��Ѵ�.
		for (int i = 0; i < topNClusters.size(); ++i)
		{
			auto currentHSVColor = topNClusters[i].GetHSVColor();
			if (currentHSVColor.z >= 80)
			{
				biggestClusterVValue = currentHSVColor.z;
				break;
			}
		}

		// top N Ŭ�����Ϳ��� �ſ� ���� ������ ��ŭ �ִ��� ����Ѵ�.
		for (int i = 0; i < topNClusters.size(); ++i)
		{
			auto currentHSVColor = topNClusters[i].GetHSVColor(); 
			if (currentHSVColor.z >= 200)
			{
				num_highValuePixels += topNClusters[i].GetTotalPoints();
				if (currentUpperValue <= currentHSVColor.z)
				{
					currentUpperValue = currentHSVColor.z;
				}
			}
		}

		// �ſ� ���� �༮�� 10%���� ROI�� �����Ѵ�? => ��� ��ο� �κ��� �����ϴ� ������.
		// �׷��Ƿ� �ִ� Value���� ���� ���� Value�� �����ؾ��Ѵ�.
		if ((float)num_highValuePixels / in_ROI.area() >= 0.1)
		{
			upperValueBound = (uchar)currentUpperValue;
		}
		else
		{
			upperValueBound = biggestClusterVValue + 60;
		}
		lowerValueBound = 80;

		// ���� ä���� ���� �κ��� �ɷ����� �ϹǷ� 0���� 4���� ����
		cv::inRange(hsvPlanes[1], 0, 40, saturationBinaryImage);
		// �ռ� ������ �� ������ �� �ɷ���.
		cv::inRange(hsvPlanes[2], lowerValueBound, upperValueBound, valueBinaryImage);

		lastBinaryImage = lastBinaryImage & saturationBinaryImage & valueBinaryImage;
	}

	// ä�� (���� �ִ�!) �����̸� (������ ����, �ϴû�����)
	else if (bImageHasCertainColor)
	{
		cv::Mat hueThresholdedImage;
		const float kThresholdPercentageToBeMajorHue = 0.6;

		// �̹������� ���� �ֵ� Hue���� �������� ������.
		auto maxHueIndex = FindMaxIndexInArray<int>(hueArray, hueArray.size());
		// �� �ֵ� Hue���� 60%�̻��� �����ϴ� Major�� Hue������ �Ǵ�
		bool bIsThisHueMajority = (float)(hueArray[maxHueIndex] / (in_ROI.area())) > kThresholdPercentageToBeMajorHue;

		cv::Mat hueBinaryImage;
		cv::Mat valBinaryImage;

		// To reduce noise.
		//cv::inRange(hsvPlanes[0], maxHueIndex * (180 / kHueIntervals), (maxHueIndex + 1) * (180 / kHueIntervals), hueBinaryImage);
		//cv::medianBlur(hueThresholdedImage, hueThresholdedImage, 5);

		cv::inRange(hsvPlanes[0], biggestClusterHValue - 20, biggestClusterHValue + 20, hueBinaryImage);
		cv::inRange(hsvPlanes[2], biggestClusterVValue - 30, biggestClusterVValue + 30, valBinaryImage);

		lastBinaryImage = lastBinaryImage & hueBinaryImage & valBinaryImage;
	}

	else
	{
		lastBinaryImage.release();
	}

	if (lastBinaryImage.data)
	{
		// Segmented �̹����� median blur
		cv::medianBlur(lastBinaryImage, lastBinaryImage, 7);

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

		out_carBodyBinaryImage = lastBinaryImage;
		out_carBodyContour = carBodyContours[currentMaxIndex];
	}
	return true;
}
#pragma optimize("gpsy", on)

void DrawPrincipalAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
	double angle;
	double hypotenuse;
	angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
	//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
	// Here we lengthen the arrow by a factor of scale
	q.x = (int)(p.x - scale * hypotenuse * cos(angle));
	q.y = (int)(p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 2, CV_AA);
	// create the arrow hooks
	p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
	line(img, p, q, colour, 2, CV_AA);
	p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
	line(img, p, q, colour, 2, CV_AA);
}
double ComputePCA_VisualizeItsResult(const cv::Mat& img, const std::vector<cv::Point2f>& in_pts, double &out_largeEigValue, double &out_smallEigValue, cv::Mat& out_drawnImage)
{
	cv::Mat data_pts((int)in_pts.size(), 2, CV_64FC1);
	cv::Mat test;

	img.copyTo(out_drawnImage);

	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = in_pts[i].x;
		data_pts.at<double>(i, 1) = in_pts[i].y;
		cv::circle(out_drawnImage, in_pts[i], 1, cv::Scalar(0, 0, 255), 2);
	}

	cv::PCA pcaAnalysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);
	std::vector<cv::Point2d> eigen_vecs(2);

	//Store the center of the object
	Point cntr = Point(static_cast<int>(pcaAnalysis.mean.at<double>(0, 0)),
		static_cast<int>(pcaAnalysis.mean.at<double>(0, 1)));

	auto meanX = pcaAnalysis.mean.at<double>(0, 0);
	auto meanY = pcaAnalysis.mean.at<double>(0, 1);

	out_largeEigValue = pcaAnalysis.eigenvalues.at<double>(0, 0);
	out_smallEigValue = pcaAnalysis.eigenvalues.at<double>(0, 1);

	eigen_vecs[0] = cv::Point2d(pcaAnalysis.eigenvectors.at<double>(0, 0), pcaAnalysis.eigenvectors.at<double>(0, 1));
	eigen_vecs[1] = cv::Point2d(pcaAnalysis.eigenvectors.at<double>(1, 0), pcaAnalysis.eigenvectors.at<double>(1, 1));

	// Draw the principal components
	circle(test, cntr, 3, Scalar(255, 0, 255), 2);
	Point p1 = cntr + 0.2 * Point(static_cast<int>(eigen_vecs[0].x * out_largeEigValue), static_cast<int>(eigen_vecs[0].y * out_largeEigValue));
	Point p2 = cntr - 0.2 * Point(static_cast<int>(eigen_vecs[1].x * out_smallEigValue), static_cast<int>(eigen_vecs[1].y * out_smallEigValue));
	DrawPrincipalAxis(out_drawnImage, cntr, p1, Scalar(0, 255, 0), 1);
	DrawPrincipalAxis(out_drawnImage, cntr, p2, Scalar(255, 255, 0), 5);

	return std::atan2(eigen_vecs[0].y, eigen_vecs[0].x) * 180 / 3.141592;
}
void ExtractImageDescriptorFromMat(const cv::Mat& in_targetImage, ImageDescriptor& out_analyzeResult, bool in_bShowExperimentResult)
{
	// TODO: Add your control notification handler code here
	AlgorithmParameter param;
	AlgorithmResult result;

	// ROI �������� ���׸�Ʈ, ��ũ��ġ ���� �� �̹��� ��ũ���� ��� (feature vector)
	if (in_targetImage.data != nullptr)
	{
		cv::Mat forClusterResultDp;
		in_targetImage.copyTo(forClusterResultDp);

		// ROI �����
		const int kROIParameter_Dividier = 8;										// ROI�� ���� ��, Width, Height�� ���� �� ������� ��Ÿ��. 
		const int kWidthMargin = in_targetImage.cols / kROIParameter_Dividier;
		const int kHeightMargin = in_targetImage.rows / kROIParameter_Dividier;
		const int kROIWidth = in_targetImage.cols - (2 * kWidthMargin);
		const int kROIHeight = in_targetImage.rows - (2 * kHeightMargin);
		const cv::Rect kROI(kWidthMargin, kHeightMargin, kROIWidth, kROIHeight);

		cv::Mat carBodyBinaryImage;
		cv::Mat largestClusterImage;
		cv::Mat finalResultMat;
		cv::Rect effectiveBoundingBox;
		std::vector<cv::Point> carBodyContourPoints;
		std::vector<cv::Point2f> scratchPoints;
		std::vector<Cluster_DBSCAN> scratchClusters;

		// ��ü ���׸����̼�
		ExtractCarBody(in_targetImage, kROI, carBodyBinaryImage, carBodyContourPoints, param, result);

		// ����� ��ü �������� ��ũ��ġ ���� ���� 
		DetectScratchPointsFromExtractionResult(in_targetImage, kROI, carBodyBinaryImage, carBodyContourPoints, scratchPoints);
		ImageDescriptor& currentImageDescriptor = out_analyzeResult;

		// Global Feature ����
		currentImageDescriptor.m_totalNumberOfPointsInROI = scratchPoints.size();
		GetMeanStdDevFromPoints(scratchPoints, currentImageDescriptor.m_globalMeanPosition, currentImageDescriptor.m_globalStdDev);
		currentImageDescriptor.m_globalSkewness = GetSkewness(currentImageDescriptor.m_globalStdDev[0], currentImageDescriptor.m_globalStdDev[1]);
		GetBoundingBoxOfScratchPoints(cv::Size(in_targetImage.cols, in_targetImage.rows), scratchPoints, false, effectiveBoundingBox);
		currentImageDescriptor.m_globalDensityInEffectiveROI = ((double)scratchPoints.size() * 100) / (effectiveBoundingBox.width * effectiveBoundingBox.height);

		// �е� ��� �׷��� ����
		ConstructScratchClusters(scratchPoints, scratchClusters);

		// �е� ��� �׷����� �� ��� Ŭ�����Ͱ� ��� 1���̻� ���� �� largest cluster�� ���� feature�� �̴´�.
		if (scratchClusters.size() != 0)
		{
			// ���� ū �Ը��� Ŭ�����͸� ���´�.
			// �Ը� ������� Ŭ�����͵��� Sorting
			std::sort(scratchClusters.begin(), scratchClusters.end(), [](const Cluster_DBSCAN& a, const Cluster_DBSCAN& b) {return a.GetSize() >= b.GetSize(); });
			std::vector<cv::Point2f> largestCluster;
			scratchClusters[0].GetVectorVersion(largestCluster);

			for (auto& eachCluster : scratchClusters)
			{
				if (eachCluster.GetSize() >= 50)
				{
					currentImageDescriptor.m_numberOfDenseClusters++;
				}
			}


			// ���� ū �Ը��� Ŭ�����Ͱ� ������� ǥ�����ֱ� ���ѿ뵵
			for (auto& eachPoint : largestCluster)
			{
				cv::circle(forClusterResultDp, eachPoint, 1, cv::Scalar(0, 255, 0), 2);
			}

			double largeEigValue = 0, smallEigValue = 0, eigValueRatio = 0;
			float degree = ComputePCA_VisualizeItsResult(in_targetImage, largestCluster, largeEigValue, smallEigValue, largestClusterImage);

			currentImageDescriptor.m_largestClusterEigenvalueRatio = largeEigValue / smallEigValue;
			currentImageDescriptor.m_largestClusterOrientation = degree;
			currentImageDescriptor.m_largestClusterSmallEigenValue = smallEigValue;
			currentImageDescriptor.m_largestClusterLargeEigenValue = largeEigValue;

			// ���� ū �Ը��� Ŭ�������� ���, �л��� ����
			GetMeanStdDevFromPoints(largestCluster, currentImageDescriptor.m_largestClusterMeanPosition, currentImageDescriptor.m_largestClusterStdDev);
			currentImageDescriptor.m_largestClusterSkewness = GetSkewness(currentImageDescriptor.m_largestClusterStdDev[0], currentImageDescriptor.m_largestClusterStdDev[1]);
			currentImageDescriptor.m_totalNumberOfPointsOfLargestCluster = largestCluster.size();
		}

		cv::Rect boundingBox;

		// 
		if (in_bShowExperimentResult)
		{
			in_targetImage.copyTo(finalResultMat);
			for (auto& point : scratchPoints)
			{
				cv::circle(finalResultMat, point, 1, cv::Scalar(0, 0, 255), 2);
			}

			cv::rectangle(finalResultMat, effectiveBoundingBox, cv::Scalar(255, 255, 0), 2);
			cv::rectangle(finalResultMat, kROI, cv::Scalar(0, 255, 0), 2);
			cv::rectangle(finalResultMat, boundingBox, cv::Scalar(255, 0, 0), 2);

			cv::imshow("Car Body Segmented Image", carBodyBinaryImage);
			if (largestClusterImage.data)
			{
				cv::imshow("Largest Cluster Result", largestClusterImage);
			}
			
			std::vector<std::vector<cv::Point>> tempForCarBodyContour;
			tempForCarBodyContour.push_back(carBodyContourPoints);
			cv::drawContours(finalResultMat, tempForCarBodyContour, -1, cv::Scalar(0, 255 ,255), 2);
			cv::imshow("Scratch Detection Result", finalResultMat);
		}
	}
}