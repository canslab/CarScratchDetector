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
					out_clusters[clusterIndex] = Cluster();
					Cluster& eachCluster = out_clusters[clusterIndex];
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

/**************************************************/
/****       Utility Functions' Implementation *****/
/**************************************************/

void ThresholdImageWithinCertainInterval(cv::Mat& in_givenImage, std::vector<int>& in_range, bool bInversion, cv::Mat& out_binaryImage)
{
	if (out_binaryImage.data)
	{
		out_binaryImage.release();
	}

	out_binaryImage = cv::Mat(in_givenImage.rows, in_givenImage.cols, CV_8UC1, cv::Scalar::all(0));
	for (auto rowIndex = 0; rowIndex < in_givenImage.rows; ++rowIndex)
	{
		for (auto colIndex = 0; colIndex < in_givenImage.cols; ++colIndex)
		{
			uchar currentValue = in_givenImage.at<uchar>(rowIndex, colIndex);
			if (currentValue >= in_range[0] && currentValue <= in_range[1])
			{
				out_binaryImage.at<uchar>(rowIndex, colIndex) = ~bInversion;
			}
		}
	}
}

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
void CaclculateEdgeMap(const cv::Mat &in_imageMat, cv::Mat& out_edgeMap)
{
	// out_edgeMap은 gradient magnitude map과 같다.
	cv::Mat copiedBlurImage;
	GaussianBlur(in_imageMat, copiedBlurImage, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);
	cv::cvtColor(copiedBlurImage, copiedBlurImage, CV_BGR2GRAY);

	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;
	//cv::Mat grad;

	cv::Sobel(copiedBlurImage, grad_x, CV_16S, 1, 0, 3);
	cv::Sobel(copiedBlurImage, grad_y, CV_16S, 0, 1, 3);

	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	cv::addWeighted(abs_grad_x, 0.2, abs_grad_y, 0.8, 0, out_edgeMap);

	cv::imshow("Gradient Image", out_edgeMap);
	//cv::inRange(grad, 100, 220, grad);
}
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
void FindPossibleDefectAreasUsingBlobDetection(const cv::Mat & in_imageMat, const std::vector<cv::Point>& out_centerPointsOfPossibleAreas)
{
	cv::Mat testImage;
	cv::SimpleBlobDetector::Params param;

	cv::cvtColor(in_imageMat, testImage, COLOR_BGR2GRAY);
	float totalPixels = in_imageMat.rows * in_imageMat.cols;

	// Change thresholds
	param.minThreshold = 10;
	param.maxThreshold = 200;

	param.filterByColor = true;
	param.blobColor = 0;

	// Filter by Area.
	param.filterByArea = true;
	param.minArea = 10;
	param.maxArea = 200;

	// Filter by Circularity
	//param.filterByCircularity = true;
	//param.minCircularity = 0.1;

	//// Filter by Convexity
	//param.filterByConvexity = true;
	//param.minConvexity = 0.87;

	// Filter by Inertia
	param.filterByInertia = true;
	param.minInertiaRatio = 0;
	param.maxInertiaRatio = 0.5;
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(param);

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

#pragma optimize("gpsy", off)
void UpdateLabelMap(cv::Mat & inout_labelMap, const std::unordered_map<int, Cluster>& in_clusters)
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
#pragma optimize("gpsy", on)

#pragma optimize("gpsy", off)
bool ExtractCarBody(const cv::Mat & in_srcImage, const AlgorithmParameter& in_parameter, AlgorithmResult& out_result)
{
	cv::Mat filteredImageMat_luv;												// 원본이미지의 민 쉬프트 필터링된 버젼
	cv::Mat originalImage;														// 원본이미지
	cv::Mat originalHSVImage;
	cv::Mat luvOriginalImageMat;												// 원본이미지의 LUV format
	cv::Mat labelMap;															// 레이블맵

	in_srcImage.copyTo(originalImage);											// 입력받은 이미지를 deep copy해옴.
	cv::cvtColor(originalImage, luvOriginalImageMat, CV_BGR2Luv);				// 원본이미지 Color Space 변환 (BGR -> Luv)
	cv::cvtColor(originalImage, originalHSVImage, CV_BGR2HSV);

	cv::Point2d imageCenter(originalImage.cols / 2, originalImage.rows / 2);	// 이미지 중심
	int kTotalPixels = originalImage.total();									// 총 픽셀수 저장

	double sp = in_parameter.GetSpatialBandwidth();								// Mean Shift Filtering을 위한 spatial radius
	double sr = in_parameter.GetColorBandwidth();								// Mean Shift Filtering을 위한 range (color) radius

	std::unordered_map<int, Cluster> clusters; 									// Cluster 모음, Key = Label, Value = Cluster

	////////////////////////////////////////////////////////////////////
	//							                                      //
	// 00.  Mean Shift Clustering & Find Largest Cluster (Car Frame)  //
	//							                                      //
	////////////////////////////////////////////////////////////////////
	// Mean Shift Filtering 코드 (OpenCV)
	cv::pyrMeanShiftFiltering(luvOriginalImageMat, filteredImageMat_luv, sp, sr);
	int minThresholdToBeCluster = (int)(originalImage.rows * originalImage.cols * 0.02);
	PerformClustering(filteredImageMat_luv, cv::Rect(0, 0, originalImage.cols, originalImage.rows), minThresholdToBeCluster, labelMap, clusters);

	const int kNumberOfCandidateClusters = 4;
	const int kNumberOfRandomSamples = (minThresholdToBeCluster < 200) ? minThresholdToBeCluster : 200;
	std::vector<int> candidateClusterLabels(kNumberOfCandidateClusters);
	std::vector<double> candidateClusterMagnitudes(kNumberOfCandidateClusters);
	std::vector<double> candidateClusterWeights(kNumberOfCandidateClusters);
	std::vector<double> candidateClusterScores(kNumberOfCandidateClusters);

	// 규모가 큰 #(kNumberOfCandidateClusters)개의 클러스터 레이블을 취득한다.
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

	// 각 Cluster들이 이미지 중심으로부터 얼마나 먼지 계산하여, 멀면 멀수록 작은 Guassian Weight를 부여해서
	// 클러스터의 중요도를 떨어뜨린다 (SeedCluster일 가능성을 낮춘다)
	auto largerLength = (originalImage.cols > originalImage.rows) ? originalImage.cols : originalImage.rows;
	const double expCoefficient = -12.5 / pow(largerLength, 2);
	for (int i = 0; i < kNumberOfCandidateClusters; ++i)
	{
		Cluster& currentCandidateCluster = clusters[candidateClusterLabels[i]];
		int currentCandidateClusterSize = currentCandidateCluster.GetTotalPoints();
		const auto& currentCandidateClusterPoints = currentCandidateCluster.GetPointsArray();

		candidateClusterMagnitudes[i] = currentCandidateClusterSize;

		double averageDistanceFromCenter = 0.0;
		// weight를 계산할 때, 클러스터에 속하는 픽셀을 random하게 몇 점 샘플링한다.
		for (int sampleIndex = 0; sampleIndex < kNumberOfRandomSamples; ++sampleIndex)
		{
			cv::Point2d tempPoint = currentCandidateClusterPoints[std::rand() % currentCandidateClusterSize];
			averageDistanceFromCenter += cv::norm(tempPoint - imageCenter);
		}
		averageDistanceFromCenter /= kNumberOfRandomSamples;

		candidateClusterWeights[i] = exp(expCoefficient * pow(averageDistanceFromCenter, 2));
		candidateClusterScores[i] = candidateClusterWeights[i] * candidateClusterMagnitudes[i];
	}

#if true
	cv::Mat filteredImageInBGR;
	cv::Mat filteredImageInHSV;
	cv::cvtColor(filteredImageMat_luv, filteredImageInBGR, CV_Luv2BGR);
	cv::cvtColor(filteredImageInBGR, filteredImageInHSV, CV_BGR2HSV);
#endif

	/////////////////////////////////
	////						   //
	//// 02.  Color Merging		   //
	////						   //
	/////////////////////////////////

	// 위에서 구한 Measure (Gaussian-based Score)를 바탕으로 seedCluster를 찾는다.
	int seedClusterIndex = candidateClusterLabels[FindMaxIndexInArray<double>(candidateClusterScores, kNumberOfCandidateClusters)];

	// 가장 점수가 높은 클러스터를 SeedCluster(차체 중 일부)이다. 
	// 이제 이 클러스터를 중심으로 색상차이가 얼마 안나는 나머지 Cluster들과 Merging 작업을 시작한다.
	Cluster &seedCluster = clusters[seedClusterIndex];
	cv::Point3i seedClusterHSVColor = seedCluster.GetHSVColor();
	cv::Point3i seedClusterLuvColor = seedCluster.GetLuvColor();

	// 씨드클러스터와 합쳐질 클러스터들의 레이블을 담는 Set
	std::set<int> toBeMergedClusterIndices;
	std::set<int> toBePreservedClusterIndicies;

	// 씨드클러스터와 합쳐질 녀석들을 차례대로 순회하며 확인한다.
	for (const auto& eachCluster : clusters)
	{
		// 씨드클러스터는 순회할 필요가 없다.
		if (eachCluster.second.GetLabel() == seedClusterIndex)
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

	// Merging후 최종 클러스터들 목록
	std::unordered_map<int, Cluster> finalClusterList;
	// Merging 되야될 녀석들을 차례대로 씨드클러스터에 넣어준다.
	for (auto& shouldBeMergedClusterLabel : toBeMergedClusterIndices)
	{
		seedCluster.Consume(clusters[shouldBeMergedClusterLabel]);
	}

	// 보존되야할 녀석들을 최종 클러스터 목록에 저장한다.
	for (auto& shouldBePreservedClusterLabel : toBePreservedClusterIndicies)
	{
		finalClusterList[shouldBePreservedClusterLabel] = clusters[shouldBePreservedClusterLabel];
	}
	// 씨드도 저장한다.
	finalClusterList[seedCluster.GetLabel()] = seedCluster;
	UpdateLabelMap(labelMap, finalClusterList);

	cv::Mat edgeMat;
	CaclculateEdgeMap(originalImage, edgeMat);

	DrawOuterContourOfCluster(originalImage, seedCluster, cv::Scalar(255, 255, 0));
	cv::imshow("Contour Image", originalImage);

	//FindPossibleDefectAreas(originalImage)

	// 레이블맵을 컬러매핑해서 Visualize한다
	cv::Mat colorLabelMap;
	cv::normalize(labelMap, colorLabelMap, 0, 255, NORM_MINMAX);
	colorLabelMap.convertTo(colorLabelMap, CV_8UC1);
	cv::applyColorMap(colorLabelMap, colorLabelMap, COLORMAP_JET);
	cv::imshow("Color Map", colorLabelMap);
	
	return true;
}
#pragma optimize("gpsy", on)