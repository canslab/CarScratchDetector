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

	// label map 할당
	cv::Mat labelMap(kInputImageHeight, kInputImageWidth, CV_32SC1, cv::Scalar::all(NOT_CLUSTER_LABEL));
	out_labelMap = labelMap;

	// Boundary를 고려하기 위해, ROI 지정
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
				currentClusterMagnitude = cv::sum(roiMat)[0];

				//int nPointsInThisCluster = findIndexColumnVector.rows;
				// if # of elements in a cluster is less than a certian number (0.5% of total number of pixels), -1 is assigned to that pixel
				if (currentClusterMagnitude > in_thresholdToBeCluster)
				{
					// 클러스터를 이루는 점들의 위치를 findIndexColumnVector가 가지고 있음
					cv::findNonZero(roiMat, findIndexColumnVector);

					// label map 만들기
					for (int i = 0; i < findIndexColumnVector.rows; ++i)
					{
						auto& pt = findIndexColumnVector.at<cv::Point>(i, 0);
						out_labelMap.at<int>(pt) = clusterIndex;
					}

					// 클러스터 컨테이너(unordered_map)에 Cluster를 등록
					out_clusters[clusterIndex] = MeanShiftCluster();
					MeanShiftCluster& eachCluster = out_clusters[clusterIndex];
					//eachCluster

					// cluster에 본인이 몇 번째 레이블인지 저장.
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
	// SeedCluster (차체 중 일부)를 중심으로 색상차이가 얼마 안나는 나머지 Cluster들과 merging 작업을 시작한다.
	MeanShiftCluster &seedCluster = in_updatedClusterList[in_seedIndex];
	cv::Point3i seedClusterHSVColor = seedCluster.GetHSVColor();
	cv::Point3i seedClusterLuvColor = seedCluster.GetLuvColor();

	// 씨드클러스터와 합쳐질 클러스터들의 레이블을 담는 Set
	std::set<int> toBeMergedClusterIndices;
	std::set<int> toBePreservedClusterIndicies;

	// 씨드클러스터와 합쳐질 녀석들을 차례대로 순회하며 확인한다.
	for (const auto& eachCluster : in_updatedClusterList)
	{
		// 씨드클러스터는 순회할 필요가 없다.
		if (eachCluster.second.GetLabel() == in_seedIndex)
		{
			continue;
		}

		const cv::Point3i& eachClusterHSVColor = eachCluster.second.GetHSVColor();
		const cv::Point3i& eachClusterLuvColor = eachCluster.second.GetLuvColor();

		// Merging Criteria로 사용할 것임. (HSV값 차이, Luv값 차이)
		auto hueDiff = std::abs(seedClusterHSVColor.x - eachClusterHSVColor.x);
		auto satDiff = std::abs(seedClusterHSVColor.y - eachClusterHSVColor.y);
		auto valDiff = std::abs(seedClusterHSVColor.z - eachClusterHSVColor.z);

		auto lDiff = std::abs(seedClusterLuvColor.x - eachClusterLuvColor.x);
		auto uDiff = std::abs(seedClusterLuvColor.y - eachClusterLuvColor.y);
		auto vDiff = std::abs(seedClusterLuvColor.z - eachClusterLuvColor.z);

		// Merge해도 좋은가? 기록
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

	// 씨드클러스터를 기반으로 Merging 작업 수행
	for (auto& shouldBeMergedClusterLabel : toBeMergedClusterIndices)
	{
		seedCluster.Consume(in_updatedClusterList[shouldBeMergedClusterLabel]);
	}

	// Merging후 최종 클러스터들 목록
	std::unordered_map<int, MeanShiftCluster> finalClusterList;
	// 보존되야할 녀석들을 최종 클러스터 목록에 저장한다.
	for (auto& shouldBePreservedClusterLabel : toBePreservedClusterIndicies)
	{
		finalClusterList[shouldBePreservedClusterLabel] = in_updatedClusterList[shouldBePreservedClusterLabel];
	}
	// 씨드도 저장한다.
	finalClusterList[seedCluster.GetLabel()] = seedCluster;

	// 결과물로 돌려줘야 하기 때문에 Copy한다
	in_updatedClusterList = finalClusterList;
}

void Find_TopN_BiggestClusters(const std::unordered_map<int, MeanShiftCluster>& in_clusters, const int in_N, std::vector<int>& out_labels)
{
	//규모가 큰 #(kNumberOfCandidateClusters)개의 클러스터 레이블을 취득한다.
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
	// 이미지 중심점, 클러스터 후부별 크기 저장하는 벡터, 클러스터 후보별 gaussian weight 계수 계산.
	// Gaussian weight는 이미지 중심점으로부터 멀어지면 멀어질수록 커진다. (exponential에 minums 지수승이므로, 이 값이 커진단 말은 score가 낮아진다는 소리)
	// (즉, 멀리 떨어지면 떨어질 수록 seed가 될 수 없음)
	cv::Point2d imageCenter(in_imageSize.width / 2, in_imageSize.height / 2);	// 이미지 중심
	std::vector<double> candidateClusterSize(in_numberOfCandidates);
	std::vector<double> candidateClusterWeights(in_numberOfCandidates);
	std::vector<double> candidateAdequacyScoresToBeSeed(in_numberOfCandidates);

	// 각 Cluster들이 이미지 중심으로부터 얼마나 먼지 계산하여, 멀면 멀수록 작은 Guassian Weight를 부여해서
	// 클러스터의 중요도를 떨어뜨린다 (SeedCluster일 가능성을 낮춘다)
	auto largerLength = (in_imageSize.width > in_imageSize.height) ? in_imageSize.width : in_imageSize.height;
	const double expCoefficient = -12.5 / pow(largerLength, 2);
	for (int i = 0; i < in_numberOfCandidates; ++i)
	{
		const MeanShiftCluster& currentCandidateCluster = in_clusters.at(in_candidateClusterLabels[i]);
		const int currentCandidateClusterSize = currentCandidateCluster.GetTotalPoints();
		const auto& currentCandidateClusterPoints = currentCandidateCluster.GetPointsArray();

		candidateClusterSize[i] = currentCandidateClusterSize;

		double averageDistanceFromCenter = 0.0;
		// weight를 계산할 때, 클러스터에 속하는 픽셀을 random하게 몇 점 샘플링한다.
		for (int sampleIndex = 0; sampleIndex < in_numberOfRandomSamples; ++sampleIndex)
		{
			cv::Point2d tempPoint = currentCandidateClusterPoints[std::rand() % currentCandidateClusterSize];
			averageDistanceFromCenter += cv::norm(tempPoint - imageCenter);
		}
		averageDistanceFromCenter /= in_numberOfRandomSamples;

		candidateClusterWeights[i] = exp(expCoefficient * pow(averageDistanceFromCenter, 2));
		candidateAdequacyScoresToBeSeed[i] = candidateClusterWeights[i] * candidateClusterSize[i];
	}

	// 위에서 구한 Measure (Gaussian-based Score)를 바탕으로 seedCluster를 찾는다.
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

	// out_edgeMap은 gradient magnitude map과 같다.
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
		// 이미지 사이즈 조정
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
	cv::Mat excludePartsImage;			// 제외해야하는 부분들을 표시한 이미지
	cv::Mat scratchDPImage;				// 스크래치를 표시하기 위한 이미지
	std::vector<std::vector<cv::Point>> shouldBeExcludedContours;
	std::vector<std::vector<cv::Point>> strongEdgeContours;
	std::vector<cv::Point2f> scratchCandidates;

	//in_targetImage.copyTo(excludePartsImage);
	cv::cvtColor(in_targetImage, copiedGrayImage, CV_BGR2GRAY);

	// 원래 그레디언트 맵, 기스를 얻기 위한 그레디언트 맵 (grad_y only)
	CaclculateGradientMap(in_targetImage, edgeGradientMap, 0.5, 0.5);
	CaclculateGradientMap(in_targetImage, gradientForCornerDetection, 0.3, 0.7);
	out_scratchPoints.clear();

	// 구분선등 기스영역에서 제외해야하는 Connected Component 들을 저장.
	cv::threshold(edgeGradientMap, highThresholdedGradientMap, 55, 255, THRESH_BINARY);
	gradientForCornerDetection = gradientForCornerDetection & in_carBodySegmentedImage;

	// 시각화
	//cv::imshow("그레디언트 맵", edgeGradientMap);
	//cv::imshow("y Gradient 그레디언트 맵 (90-255 사이)", gradientForCornerDetection);

	// Gradient Map에서 Connected Component를 계산
	cv::findContours(highThresholdedGradientMap, strongEdgeContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// 차 구분선 추출 & 두껍께 까진 기스 영역 추출
	for (int j = 0; j < strongEdgeContours.size(); j++)
	{
		std::vector<std::vector<cv::Point>> hull(1);
		cv::convexHull(cv::Mat(strongEdgeContours[j]), hull[0]);

		double gradientContourLength = cv::arcLength(hull[0], true);

		std::vector<cv::Point> innerPoints;
		GetPointsInContour(strongEdgeContours[j], 2.5, innerPoints);

		// 그레디언트 맵에서 충분히 길이가 긴 녀석은 차 구분선일 가능성이 크므로 제외해야 하는 윤곽선 목록(excludedContourLength)에 등록
		if (gradientContourLength >= 500 || (!IsThisContourContainedInROI(strongEdgeContours[j], cv::Size(in_targetImage.cols, in_targetImage.rows), in_ROI)))
		{
			cv::Scalar color(0, 200, 200);

			// 구분선들을 저장한다.
			shouldBeExcludedContours.push_back(strongEdgeContours[j]);

			// 구분선들을 Visualize한다
			//cv::drawContours(excludePartsImage, strongEdgeContours, j, color, 2);
		}
		else if (gradientContourLength >= 10 && innerPoints.size() > 0
			&& IsThisContourContainedInROI(strongEdgeContours[j], cv::Size(in_targetImage.cols, in_targetImage.rows), in_ROI)
			&& IsContourInsideCarBody(strongEdgeContours[j], in_carBodyContourPoints))
		{
			// 두꺼운 기스 후보군 그리기
			//cv::drawContours(excludePartsImage, strongEdgeContours, j, cv::Scalar(255, 255, 0), 2);

			unsigned int innerGrayValueMean = 0;
			for (const auto& eachInnerPoint : innerPoints)
			{
				uchar grayValue = copiedGrayImage.at<uchar>(eachInnerPoint);
				innerGrayValueMean += (int)grayValue;
			}
			// 두꺼운 영역 내부의 Gray value 평균
			innerGrayValueMean /= innerPoints.size();

			// 두껍께 까진 기스가 되기 위한 조건은 아래와 같다
			// (두꺼운 영역 내부에 있는 gray value값의 평균) - (그 영역을 포함하는 바운딩 박스 내부에서 gray평균) >= 30 
			// 평균값이 많이 차이 난다 == 두꺼운 기스다
			auto boundingRect = cv::boundingRect(strongEdgeContours[j]);
			auto boundingRectMean = cv::sum(copiedGrayImage(boundingRect))[0];
			boundingRectMean /= boundingRect.area();

			// 위에 말했듯이, 평균이 많이 차이 나면, 다 스크래치들이다.. 저장한다 
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

	// Segmentation이 잘 이루어진 경우
	if (in_carBodySegmentedImage.data)
	{
		std::vector<cv::Point2f> corners;
		std::vector<std::vector<cv::Point>> tempForCarBodyVisualize;

		cv::Mat pointRecordedMat(in_targetImage.rows, in_targetImage.cols, CV_8UC1, cv::Scalar(0));
		cv::goodFeaturesToTrack(copiedGrayImage(in_ROI), corners, 1500, 0.005, 0.5, cv::Mat(), 3, false);

		// 차 영역을 그려준다.
		tempForCarBodyVisualize.push_back(in_carBodyContourPoints);
		//cv::drawContours(excludePartsImage, tempForCarBodyVisualize, -1, cv::Scalar(255, 0, 0), 2);

		// 코너들은 ROI 내부에서 검출했으므로, 다시 compensate 해준다.
		for (auto& eachPoint : corners)
		{
			eachPoint.x += in_ROI.x;
			eachPoint.y += in_ROI.y;
		}

		// 검출한 코너들을 순회하며, Scratch 가능성이 있는 코너들을 점으로 표시함.
		for (auto& point : corners)
		{
			// Specular 판단, 코너점 주위 3 x 3 영역을 살피며, 밝기값이 210을 넘어가는 코너점은 Specular Point로 간주
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

			// 현재 코너점의 3 x 3 영역에 1개 이상의 포화된 픽셀이 존재한다 => Specular라고 판단
			if (nSpecularPoints >= 1)
			{
				continue;
			}

			// 코너점 중에서 Gradient 가 어느정도 크기가 되어서 의미있어야 하며
			// 코너점은 메인 차체 내부에 있거나 근처에 있어야함.
			auto distanceFromSegmentedBoundary = cv::pointPolygonTest(in_carBodyContourPoints, point, true);

			// 스크래치가 ROI 내부에 존재하고 동시에 어느정도 Gradient 값이 있어야함.
			if (gradientForCornerDetection.at<uchar>(point) >= 8 && IsThisPointInROI(in_ROI, point) == true)
			{
				// ROI 중에서도 외곽이 아니어야만 함
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

		// isolated 되어 있는 스크래치 후보 점들을 제거 하기 위해서, 이미지에 각 스크래치 포인트를 기록한다.
		for (auto& eachScratchCandidatePoint : scratchCandidates)
		{
			pointRecordedMat.at<uchar>(eachScratchCandidatePoint) = 255;
		}

		// 
		auto marginWidth_ROI = in_ROI.width / 4;
		auto marginHeight_ROI = in_ROI.height / 4;

		// 스크래치 포인트 후보군들을 순회하면서, 스크래치로 적합한 아이들만 합쳐준다. 
		for (auto& eachPoint : scratchCandidates)
		{
			auto x_left = std::abs(eachPoint.x - in_ROI.x);
			auto x_right = std::abs((in_ROI.x + in_ROI.width) - eachPoint.x);

			auto y_upper = std::abs(eachPoint.y - in_ROI.y);
			auto y_bottom = std::abs((in_ROI.y + in_ROI.height) - eachPoint.y);

			auto distanceFromROI_x = (x_left < x_right) ? x_left : x_right;
			auto distanceFromROI_y = (y_upper < y_bottom) ? y_upper : y_bottom;

			// ROI에서 변방에 위치한 녀석들 (isolated된 코너)은 새로운 처리가 필요하다.
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

				// 변방에 위치한 점주변에 다른 코너가 있으면 isolate가 아니라고 간주
				if (count > 2)
				{
					out_scratchPoints.push_back(eachPoint);
				}
			}
			// 변방이 아닌 중앙부위에 있을 때는 묻지도 따지지도 않고 포함
			else
			{
				out_scratchPoints.push_back(eachPoint);
			}
		}

		//cv::imshow("Binary Image", in_carBodySegmentedImage);
		//cv::imshow("차 외곽선 + 구분선 + 두꺼운 기스", excludePartsImage);
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
				// 이 점은 아웃라이어임 ==> 제거 대상
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

	in_srcImage.copyTo(originalImage);											// 입력받은 이미지를 deep copy해옴.

	cv::cvtColor(originalImage, copiedGrayImage, CV_BGR2GRAY);					// 입력받은 이미지 그레이스케일 화
	cv::cvtColor(originalImage, luvOriginalImageMat, CV_BGR2Luv);				// 원본이미지 Color Space 변환 (BGR -> Luv)
	cv::cvtColor(originalImage, originalHSVImage, CV_BGR2HSV);

	const int kTotalPixels = originalImage.total();								// 총 픽셀수 저장
	const int kHueIntervals = 9;												// Hue histogram 만들 때 사용할, Bin의 갯수
	const int kSatIntervals = 16;												// Saturation histogram 만들 때 사용할, Bin의 갯수
																				// ROI 설정
	const int kROIWidth = in_ROI.width;
	const int kROIHeight = in_ROI.height;

	const int kTotalPixelsInROI = in_ROI.area();									// ROI내부에 존재하는 총 픽셀수
	const double kHighThresholdToHighSatImage = 0.50;							// 무채색 이미지이기 위한 Saturation 기준 비율, ROI내부픽셀의 80%(=0.65)가 저채도 => 저채도이미지이다.
	const double kLowThresholdToHaveHighSatImage = 0.3;							// 유채색 차량(ex. 하늘색)을 포함하는 이미지는 Saturation 비율이 30%(=0.3) 이하이다.

	cv::Mat hsvPlanes[3];
	cv::Mat LUVPlanes[3];
	std::vector<int> hueArray(kHueIntervals);
	std::vector<int> satArray(kSatIntervals);

	cv::pyrMeanShiftFiltering(luvOriginalImageMat, filteredImageMat_luv, 10, 16);
	std::unordered_map<int, MeanShiftCluster> clusters; 						// Cluster 모음, Key = Label, Value = Cluster
	// ROI 내부에서만 Clustering을 수행
	PerformClustering(filteredImageMat_luv, in_ROI, 100, labelMap, clusters);

	const int kN = 5;
	std::vector<MeanShiftCluster> topNClusters;
	GetTopNClusters(clusters, kN, topNClusters);		// 규모가 큰 top (kN)개 클러스터 얻어옴
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

	// 이미지의 색상 분포를 파악하는데 사용
	// Hue분포, Saturation 분포 계산
	for (int rowIndex = in_ROI.y; rowIndex < in_ROI.y + in_ROI.height; ++rowIndex)
	{
		for (int colIndex = in_ROI.x; colIndex < in_ROI.x + in_ROI.width; ++colIndex)
		{
			hueArray[(int)(hsvPlanes[0].at<uchar>(rowIndex, colIndex) / (180 / kHueIntervals))]++;
			satArray[(int)(hsvPlanes[1].at<uchar>(rowIndex, colIndex) / (256 / kSatIntervals))]++;
		}
	}

	// 0.65를 넘어서면 이건 흰색 차량이야.
	// 이미지에서 대부분의 픽셀이 무채색임. (0.65란 전체이미지 픽셀 중 65%가 0에 가까운 채도이다.)
	float lowSaturationPixelRatio = (float)(satArray[0] + satArray[1] + satArray[2]) / kTotalPixelsInROI;
	cv::Mat lastBinaryImage(originalImage.rows, originalImage.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat whiteInRoiMat;
	lastBinaryImage(in_ROI) = 255;

	bool bImageHasLowSaturationCarColor = (lowSaturationPixelRatio > kHighThresholdToHighSatImage);
	bool bImageHasCertainColor = (lowSaturationPixelRatio < 0.3);

	// 흰색 차량 검출을 위함.
	if (bImageHasLowSaturationCarColor)
	{
		cv::Mat saturationBinaryImage;
		cv::Mat valueBinaryImage;
		int currentUpperValue = 0;
		int num_highValuePixels = 0;
		uchar upperValueBound = 0;
		uchar lowerValueBound = 0;

		// top N 클러스터에서 최초로 흰색 부분을 찾아야한다.
		for (int i = 0; i < topNClusters.size(); ++i)
		{
			auto currentHSVColor = topNClusters[i].GetHSVColor();
			if (currentHSVColor.z >= 80)
			{
				biggestClusterVValue = currentHSVColor.z;
				break;
			}
		}

		// top N 클러스터에서 매우 밝은 영역이 얼만큼 있는지 계산한다.
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

		// 매우 밝은 녀석이 10%정도 ROI에 존재한다? => 밝고 어두운 부분이 공존하는 차량임.
		// 그러므로 최대 Value값을 아주 밝은 Value로 설정해야한다.
		if ((float)num_highValuePixels / in_ROI.area() >= 0.1)
		{
			upperValueBound = (uchar)currentUpperValue;
		}
		else
		{
			upperValueBound = biggestClusterVValue + 60;
		}
		lowerValueBound = 80;

		// 먼저 채도가 낮은 부분을 걸러내야 하므로 0에서 4으로 설정
		cv::inRange(hsvPlanes[1], 0, 40, saturationBinaryImage);
		// 앞서 설정한 명도 값으로 더 걸러냄.
		cv::inRange(hsvPlanes[2], lowerValueBound, upperValueBound, valueBinaryImage);

		lastBinaryImage = lastBinaryImage & saturationBinaryImage & valueBinaryImage;
	}

	// 채색 (색이 있는!) 차량이면 (새빨강 차량, 하늘색차량)
	else if (bImageHasCertainColor)
	{
		cv::Mat hueThresholdedImage;
		const float kThresholdPercentageToBeMajorHue = 0.6;

		// 이미지에서 가장 주된 Hue값은 무엇인지 구하자.
		auto maxHueIndex = FindMaxIndexInArray<int>(hueArray, hueArray.size());
		// 그 주된 Hue값이 60%이상을 차지하는 Major한 Hue값인지 판단
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
		// Segmented 이미지를 median blur
		cv::medianBlur(lastBinaryImage, lastBinaryImage, 7);

		std::vector<std::vector<cv::Point>> carBodyContours;
		cv::findContours(lastBinaryImage, carBodyContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// Contour중 가장 긴 녀석이 차체 프레임이라고 할 수 있음
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

	// ROI 영역에서 세그멘트, 스크래치 검출 후 이미지 디스크립터 계산 (feature vector)
	if (in_targetImage.data != nullptr)
	{
		cv::Mat forClusterResultDp;
		in_targetImage.copyTo(forClusterResultDp);

		// ROI 만들기
		const int kROIParameter_Dividier = 8;										// ROI를 만들 때, Width, Height를 각각 몇 등분할지 나타냄. 
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

		// 차체 세그멘테이션
		ExtractCarBody(in_targetImage, kROI, carBodyBinaryImage, carBodyContourPoints, param, result);

		// 검출된 차체 영역에서 스크래치 검출 수행 
		DetectScratchPointsFromExtractionResult(in_targetImage, kROI, carBodyBinaryImage, carBodyContourPoints, scratchPoints);
		ImageDescriptor& currentImageDescriptor = out_analyzeResult;

		// Global Feature 저장
		currentImageDescriptor.m_totalNumberOfPointsInROI = scratchPoints.size();
		GetMeanStdDevFromPoints(scratchPoints, currentImageDescriptor.m_globalMeanPosition, currentImageDescriptor.m_globalStdDev);
		currentImageDescriptor.m_globalSkewness = GetSkewness(currentImageDescriptor.m_globalStdDev[0], currentImageDescriptor.m_globalStdDev[1]);
		GetBoundingBoxOfScratchPoints(cv::Size(in_targetImage.cols, in_targetImage.rows), scratchPoints, false, effectiveBoundingBox);
		currentImageDescriptor.m_globalDensityInEffectiveROI = ((double)scratchPoints.size() * 100) / (effectiveBoundingBox.width * effectiveBoundingBox.height);

		// 밀도 기반 그룹핑 수행
		ConstructScratchClusters(scratchPoints, scratchClusters);

		// 밀도 기반 그룹핑을 한 결과 클러스터가 적어도 1개이상 나올 때 largest cluster에 관한 feature를 뽑는다.
		if (scratchClusters.size() != 0)
		{
			// 가장 큰 규모의 클러스터를 얻어온다.
			// 규모 순서대로 클러스터들을 Sorting
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


			// 가장 큰 규모의 클러스터가 어디인지 표시해주기 위한용도
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

			// 가장 큰 규모의 클러스터의 평균, 분산을 구함
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