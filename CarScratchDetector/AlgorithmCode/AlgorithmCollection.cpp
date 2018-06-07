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

	in_srcImage.copyTo(originalImage);											// 입력받은 이미지를 deep copy해옴.

	cv::cvtColor(originalImage, copiedGrayImage, CV_BGR2GRAY);					// 입력받은 이미지 그레이스케일 화
	cv::cvtColor(originalImage, luvOriginalImageMat, CV_BGR2Luv);				// 원본이미지 Color Space 변환 (BGR -> Luv)
	cv::cvtColor(originalImage, originalHSVImage, CV_BGR2HSV);

	const int kTotalPixels = originalImage.total();								// 총 픽셀수 저장
	const double sp = in_parameter.m_spatialBandwidth;							// Mean Shift Filtering을 위한 spatial radius
	const double sr = in_parameter.m_colorBandwidth;							// Mean Shift Filtering을 위한 range (color) radius

	const int kHueIntervals = 9;												// Hue histogram 만들 때 사용할, Bin의 갯수
	const int kSatIntervals = 16;												// Saturation histogram 만들 때 사용할, Bin의 갯수

	// ROI 설정
	const int kROIParameter_Dividier = 8;										// ROI를 만들 때, Width, Height를 각각 몇 등분할지 나타냄. 
	const int kWidthMargin = originalImage.cols / kROIParameter_Dividier;
	const int kHeightMargin = originalImage.rows / kROIParameter_Dividier;
	const int kROIWidth = originalImage.cols - (2 * kWidthMargin);
	const int kROIHeight = originalImage.rows - (2 * kHeightMargin);
	const cv::Rect kROI(kWidthMargin, kHeightMargin, kROIWidth, kROIHeight);

	const int kTotalPixelsInROI = kROI.area();									// ROI내부에 존재하는 총 픽셀수
	const double kHighThresholdToHighSatImage = 0.65;							// 무채색 이미지이기 위한 Saturation 기준 비율, ROI내부픽셀의 80%(=0.65)가 저채도 => 저채도이미지이다.
	const double kLowThresholdToHaveHighSatImage = 0.3;							// 유채색 차량(ex. 하늘색)을 포함하는 이미지는 Saturation 비율이 30%(=0.3) 이하이다.

	std::unordered_map<int, MeanShiftCluster> clusters; 						// Cluster 모음, Key = Label, Value = Cluster
	cv::pyrMeanShiftFiltering(luvOriginalImageMat, filteredImageMat_luv, sp, sr);
	cv::cvtColor(filteredImageMat_luv, filteredImageInBGR, CV_Luv2BGR);
	cv::cvtColor(filteredImageInBGR, filteredImageInHSV, CV_BGR2HSV);

	cv::Mat highThresholdedGradientMap;
	cv::Mat testImage;
	cv::Mat yGradientMap;
	std::vector<std::vector<cv::Point>> shouldBeExcludedContours;
	originalImage.copyTo(testImage);

	// 원래 그레디언트 맵, 기스를 얻기 위한 그레디언트 맵 (grad_y only)
	CaclculateGradientMap(originalImage, edgeGradientMap, 0.5, 0.5);
	CaclculateGradientMap(originalImage, yGradientMap, 0, 1);
	cv::medianBlur(edgeGradientMap, edgeGradientMap, 5);

	cv::imshow("그레디언트 맵", edgeGradientMap);

	// 구분선등 기스영역에서 제외해야하는 Connected Component 들을 저장.
	{
		cv::threshold(edgeGradientMap, highThresholdedGradientMap, 60, 255, THRESH_BINARY);
		cv::imshow("High Thresholded 그레디언트 맵 (90-255 사이)", yGradientMap);
		std::vector<std::vector<cv::Point>> strongEdgeContours;
		cv::findContours(highThresholdedGradientMap, strongEdgeContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		for (int j = 0; j < strongEdgeContours.size(); j++)
		{
			double gradientContourLength = cv::arcLength(strongEdgeContours[j], true);

			// 구분선이라고 확신을 할 수 없지만, 외각 30 픽셀 밖의 영역에 걸쳐져있다면 ==> 제외한다 (어차피 여기서는 기스가 생길 수 없다는 가정)
	/*		int marginWidth = (int)((double)originalImage.cols / 8);
			int marginHeight = (int)((double)originalImage.rows / 8);*/
			//cv::Rect roi(0 + marginWidth, 0 + marginHeight, originalImage.cols - (2 * marginWidth), originalImage.rows - (2 * marginHeight));

			// 그레디언트 맵에서 충분히 길이가 긴 녀석은 차 구분선일 가능성이 크므로 제외해야 하는 윤곽선 목록(excludedContourLength)에 등록
			if (gradientContourLength >= 500 || (gradientContourLength >= 10 && !IsThisContourInROI(strongEdgeContours[j], cv::Size(originalImage.cols, originalImage.rows), kROI)))
			{
				shouldBeExcludedContours.push_back(strongEdgeContours[j]);
				cv::drawContours(testImage, strongEdgeContours, j, cv::Scalar(rand() % 256, rand() % 256, rand() % 256), 2);
			}
		}
	}
	cv::imshow("구분선 및 외각지역을 침범하고 있는 윤곽선들 ==> 코너점 검출 제외 대상", testImage);

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


	// 이미지의 색상 분포를 파악하는데 사용
	// Hue분포, Saturation 분포 계산
	for (int rowIndex = (int)((double)originalImage.rows / kROIParameter_Dividier) - 1; rowIndex < (int)((double)originalImage.rows * (kROIParameter_Dividier - 1) / kROIParameter_Dividier); ++rowIndex)
	{
		for (int colIndex = (int)((double)originalImage.cols / kROIParameter_Dividier) - 1; colIndex < (int)((double)originalImage.cols * (kROIParameter_Dividier - 1) / kROIParameter_Dividier); ++colIndex)
		{
			hueArray[(int)(hsvPlanes[0].at<uchar>(rowIndex, colIndex) / (180 / kHueIntervals))]++;
			satArray[(int)(hsvPlanes[1].at<uchar>(rowIndex, colIndex) / (256 / kSatIntervals))]++;
		}
	}

	// 0.65를 넘어서면 이건 흰색 차량이야.
	// 이미지에서 대부분의 픽셀이 무채색임. (0.65란 전체이미지 픽셀 중 65%가 0에 가까운 채도이다.)
	float lowSaturationPixelRatio = (float)(satArray[0] + satArray[1] + satArray[2]) / kTotalPixelsInROI;
	cv::Mat lastBinaryImage(originalImage.rows, originalImage.cols, CV_8UC1, cv::Scalar::all(0));
	bool bImageHasLowSaturationCarColor = (lowSaturationPixelRatio > kHighThresholdToHighSatImage);
	bool bImageHasCertainColor = (lowSaturationPixelRatio < 0.3);
	if (bImageHasLowSaturationCarColor)
	{
		cv::Mat saturationBinaryImage;
		cv::Mat valueBinaryImage;

		cv::inRange(hsvPlanes[1], 0, 40, saturationBinaryImage);
		// 명도가 90에서 255사이인 영역을 뽑아내자.
		cv::threshold(hsvPlanes[2], valueBinaryImage, 90, 255, CV_THRESH_BINARY);
		// 이미지에서 채도가 낮으면서 명도는 일정 값 이상하는 영역을 가려낸다.
		// 흰색차량(무채색)의 프레임은 보통 명도가 일정 값 이상이면서 채도가 낮다.
		lastBinaryImage = saturationBinaryImage & valueBinaryImage;

		cv::medianBlur(lastBinaryImage, lastBinaryImage, 7);
	}

	// 채색 (색이 있는!) 차량이면 (새빨강 차량, 하늘색차량)
	else if (bImageHasCertainColor)
	{
		cv::Mat hueThresholdedImage;
		const float kThresholdPercentageToBeMajorHue = 0.6;

		// 이미지에서 가장 주된 Hue값은 무엇인지 구하자.
		auto maxHueIndex = FindMaxIndexInArray<int>(hueArray, hueArray.size());
		// 그 주된 Hue값이 60%이상을 차지하는 Major한 Hue값인지 판단
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

	// 잘 구분이 되서 차 프레임을 걸러낸 경우 (==lastBinaryImage가 존재하는경우)
	if (lastBinaryImage.data)
	{
		// 차 프레임을 포함하는 이진화 영상 표시
		cv::imshow("Binary Image", lastBinaryImage);

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
		std::vector<cv::Point2f> corners;

		cv::goodFeaturesToTrack(copiedGrayImage(kROI), corners, 1200, 0.01, 1, cv::Mat(), 3, false, 0.04);
		for (auto& point : corners)
		{
			point.x += kROI.x;
			point.y += kROI.y;
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
			// 코너점은 메인차체 내부에 있거나 근처에 있어야함.
			if (yGradientMap.at<uchar>(point) >= 12 && (cv::pointPolygonTest(carBodyContours[currentMaxIndex], point, false) > 0))
			{
				// 코너점이 차 구분선 근처에 있지 않는 경우만 의미있음
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